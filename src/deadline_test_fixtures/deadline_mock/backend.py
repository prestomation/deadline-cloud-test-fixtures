# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Observable HTTP mock for Deadline submitter tests."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from typing import Any, Callable, ClassVar
from urllib.parse import parse_qs, urlparse

import botocore.session
from botocore.exceptions import ClientError
from botocore.model import ServiceModel
from botocore.validate import ParamValidator
from openjd.model import (
    DocumentType,
    ParameterValue,
    ParameterValueType,
    create_job,
    decode_environment_template,
    decode_job_template,
    document_string_to_object,
)

from .scenario import _FAKE_PRINCIPAL, MockDeadlineScenario
from .worker_scheduler import MockWorkerScheduler

API_PREFIX = "/2023-10-12"
ADMIN_STATE_PATH = "/__deadline_test_fixtures__/state"
ADMIN_RESET_PATH = "/__deadline_test_fixtures__/reset"

_INT_QUERY_PARAMS = {"maxResults", "itemOffset", "pageSize"}

# HTTP status for each modeled error code. botocore reads the code from the
# x-amzn-errortype header; the status keeps the wire format close to the service.
_ERROR_STATUS = {
    "AccessDeniedException": 403,
    "ConflictException": 409,
    "InternalServerErrorException": 500,
    "InternalServerException": 500,
    "ResourceNotFoundException": 404,
    "ServiceQuotaExceededException": 402,
    "ThrottlingException": 429,
    "ValidationException": 400,
}
_MAX_BODY_BYTES = 10 * 1024 * 1024

# CreateJob's JobParameter union members, mapped to Open Job Description value types.
_PARAMETER_VALUE_TYPES = {
    "string": ParameterValueType.STRING,
    "path": ParameterValueType.PATH,
    "int": ParameterValueType.INT,
    "float": ParameterValueType.FLOAT,
}


def route(method: str, path: str, operation: str) -> Callable:
    """Associate a backend method with a Deadline REST-JSON route."""

    def decorator(function: Callable) -> Callable:
        # B010: setattr is deliberate. `Callable` has no `__http_route__` attribute,
        # so a direct assignment fails mypy's attr-defined check.
        http_route = (method, f"{API_PREFIX}{path}", operation)
        setattr(function, "__http_route__", http_route)  # noqa: B010
        return function

    return decorator


def _resource_not_found(resource_type: str, resource_id: str, operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": (
                    f"Resource of type {resource_type} with id {resource_id} does not exist."
                ),
            }
        },
        operation,
    )


def _validation_exception(message: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "ValidationException", "Message": message}}, operation)


def _parse_document(document: str, document_type: str | None) -> dict[str, Any]:
    """Parse a CreateJob or queue environment template string; raises ValueError."""
    return document_string_to_object(
        document=document,
        document_type=DocumentType.JSON if document_type == "JSON" else DocumentType.YAML,
    )


def _declared_extensions(document: dict[str, Any]) -> list[str]:
    # Accept whichever extensions the template declares; openjd-model still rejects
    # names it does not know. Reading them from the template rather than from
    # openjd-model's extension enum keeps this working across openjd-model versions.
    extensions = document.get("extensions")
    return list(extensions) if isinstance(extensions, list) else []


def _parameter_value(name: str, value: Any) -> ParameterValue:
    """Convert a CreateJob JobParameter such as ``{"int": "3"}``; raises ValueError."""
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(
            f"Parameter {name} must have exactly one of {sorted(_PARAMETER_VALUE_TYPES)}"
        )
    ((kind, raw),) = value.items()
    if kind not in _PARAMETER_VALUE_TYPES:
        raise ValueError(f"Parameter {name} has unknown value type {kind!r}")
    return ParameterValue(type=_PARAMETER_VALUE_TYPES[kind], value=str(raw))


class MockDeadlineBackend:
    """In-memory resources and observability for the mock service."""

    def __init__(
        self,
        scenario: MockDeadlineScenario | None = None,
        *,
        validate_responses: bool = True,
    ) -> None:
        self.scenario = scenario or MockDeadlineScenario()
        self.validate_responses = validate_responses
        self.response_delay_s = self.scenario.response_delay_s
        self._lock = threading.Lock()
        self.log_callback: Callable[[str], None] | None = None
        self.farms: dict[str, dict[str, Any]] = {}
        self.queues: dict[tuple[str, str], dict[str, Any]] = {}
        self.queue_environments: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.storage_profiles: dict[tuple[str, str], list[dict[str, Any]]] = {}
        # Created by CreateJob, keyed by (farm ID, queue ID, job ID). Each entry holds
        # the full CreateJob request (template, parameters, ...) plus the fields the
        # service assigns, so tests can assert on exactly what was submitted.
        self.jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.call_counts: dict[str, int] = {}
        self.request_log: list[tuple[str, str, str]] = []
        self.unmatched_requests: list[tuple[str, str]] = []
        # Errors to return instead of calling a route, keyed by operation name.
        self.injected_errors: dict[str, list[ClientError]] = {}
        # Extra seconds to wait before answering, keyed by operation name.
        self.operation_delays: dict[str, float] = {}
        self.worker_scheduler = self._new_worker_scheduler()
        self.scenario.seed(self)

    @property
    def farm_id(self) -> str:
        return self.scenario.farm_id

    @property
    def queue_id(self) -> str:
        return self.scenario.queue_id

    def _new_worker_scheduler(self) -> MockWorkerScheduler:
        queue = self.scenario.queue
        role_arn = queue.get("roleArn")
        return MockWorkerScheduler(
            queue_role_arns={str(queue["queueId"]): str(role_arn)} if role_arn else {}
        )

    def reset(self) -> None:
        """Restore scenario resources, discard created jobs and workers, and clear
        request observability."""
        with self._lock:
            self.call_counts.clear()
            self.request_log.clear()
            self.unmatched_requests.clear()
            self.injected_errors.clear()
            self.operation_delays.clear()
            self.jobs.clear()
            self.worker_scheduler = self._new_worker_scheduler()
            self.scenario.seed(self)

    def inject_error(
        self,
        operation: str,
        code: str,
        *,
        count: int = 1,
        message: str = "Injected by the test",
        **fields: Any,
    ) -> None:
        """Make the next ``count`` calls to ``operation`` fail with ``code``.

        ``fields`` become members of the error body, for example
        ``reason="CONCURRENT_MODIFICATION"`` on a ``ConflictException`` or
        ``retryAfterSeconds=1`` on a ``ThrottlingException``. The route is not called,
        so the request has no effect, as with a real failed request.
        """
        error = ClientError({"Error": {"Code": code, "Message": message, **fields}}, operation)
        with self._lock:
            self.injected_errors.setdefault(operation, []).extend([error] * count)

    def _take_injected_error(self, operation: str) -> ClientError | None:
        with self._lock:
            pending = self.injected_errors.get(operation)
            return pending.pop(0) if pending else None

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable state for remote assertions."""
        with self._lock:
            return {
                "identifiers": {
                    "farm_id": self.farm_id,
                    "queue_id": self.queue_id,
                },
                "call_counts": dict(self.call_counts),
                "request_log": [list(request) for request in self.request_log],
                "unmatched_requests": [list(request) for request in self.unmatched_requests],
                "resources": {
                    "farms": list(self.farms.values()),
                    "queues": list(self.queues.values()),
                    "queue_environments": [
                        environment
                        for environments in self.queue_environments.values()
                        for environment in environments
                    ],
                    "storage_profiles": [
                        profile
                        for profiles in self.storage_profiles.values()
                        for profile in profiles
                    ],
                    "jobs": list(self.jobs.values()),
                },
                "worker_protocol": self.worker_scheduler.snapshot(),
            }

    def _log(self, message: str) -> None:
        if self.log_callback is not None:
            try:
                self.log_callback(message)
            except Exception:  # noqa: S110
                # Observability is best-effort and must not break request handling.
                pass

    def _queue_key(self, farm_id: str, queue_id: str, operation: str) -> tuple[str, str]:
        key = (farm_id, queue_id)
        if key not in self.queues:
            raise _resource_not_found("queue", queue_id, operation)
        return key

    @route("GET", "/farms", "ListFarms")
    def list_farms(self, **kwargs: Any) -> dict[str, Any]:
        return {"farms": list(self.farms.values())}

    @route("GET", "/farms/{farmId}", "GetFarm")
    def get_farm(self, *, farmId: str) -> dict[str, Any]:
        try:
            return dict(self.farms[farmId])
        except KeyError:
            raise _resource_not_found("farm", farmId, "GetFarm")

    @route("GET", "/farms/{farmId}/queues", "ListQueues")
    def list_queues(self, *, farmId: str, **kwargs: Any) -> dict[str, Any]:
        if farmId not in self.farms:
            raise _resource_not_found("farm", farmId, "ListQueues")
        return {
            "queues": [
                queue
                for (candidate_farm, _), queue in self.queues.items()
                if candidate_farm == farmId
            ]
        }

    @route("GET", "/farms/{farmId}/queues/{queueId}", "GetQueue")
    def get_queue(self, *, farmId: str, queueId: str) -> dict[str, Any]:
        key = self._queue_key(farmId, queueId, "GetQueue")
        return dict(self.queues[key])

    @route(
        "GET",
        "/farms/{farmId}/queues/{queueId}/user-roles",
        "AssumeQueueRoleForUser",
    )
    def assume_queue_role_for_user(self, *, farmId: str, queueId: str) -> dict[str, Any]:
        self._queue_key(farmId, queueId, "AssumeQueueRoleForUser")
        return {
            "credentials": {
                "accessKeyId": "testing",
                "secretAccessKey": "testing",
                "sessionToken": "testing",
                "expiration": datetime.now(timezone.utc) + timedelta(hours=1),
            }
        }

    @route(
        "GET",
        "/farms/{farmId}/queues/{queueId}/storage-profiles",
        "ListStorageProfilesForQueue",
    )
    def list_storage_profiles_for_queue(
        self, *, farmId: str, queueId: str, **kwargs: Any
    ) -> dict[str, Any]:
        key = self._queue_key(farmId, queueId, "ListStorageProfilesForQueue")
        return {"storageProfiles": list(self.storage_profiles.get(key, ()))}

    @route(
        "GET",
        "/farms/{farmId}/queues/{queueId}/environments",
        "ListQueueEnvironments",
    )
    def list_queue_environments(
        self, *, farmId: str, queueId: str, **kwargs: Any
    ) -> dict[str, Any]:
        key = self._queue_key(farmId, queueId, "ListQueueEnvironments")
        return {"environments": list(self.queue_environments.get(key, ()))}

    @route(
        "GET",
        "/farms/{farmId}/queues/{queueId}/environments/{queueEnvironmentId}",
        "GetQueueEnvironment",
    )
    def get_queue_environment(
        self,
        *,
        farmId: str,
        queueId: str,
        queueEnvironmentId: str,
    ) -> dict[str, Any]:
        key = self._queue_key(farmId, queueId, "GetQueueEnvironment")
        for environment in self.queue_environments.get(key, ()):
            if environment.get("queueEnvironmentId") == queueEnvironmentId:
                return dict(environment)
        raise _resource_not_found("queueEnvironment", queueEnvironmentId, "GetQueueEnvironment")

    def _job_name(
        self,
        key: tuple[str, str],
        template: str,
        template_type: str | None,
        parameters: dict[str, Any],
    ) -> str:
        """Return the job's name, first validating the submission when the scenario asks.

        Validation mirrors the service: the template must decode as an Open Job
        Description job template, and the parameter values must be accepted by it
        together with the queue's environment templates (which define parameters
        such as CondaPackages).
        """
        validate = self.scenario.validate_job_templates
        try:
            document = _parse_document(template, template_type)
            if not validate:
                return str(document.get("name", ""))
            job_template = decode_job_template(
                template=document, supported_extensions=_declared_extensions(document)
            )
            environment_templates = []
            for environment in self.queue_environments.get(key, ()):
                if "template" not in environment:
                    continue
                environment_document = _parse_document(
                    environment["template"], environment.get("templateType")
                )
                environment_templates.append(
                    decode_environment_template(
                        template=environment_document,
                        supported_extensions=_declared_extensions(environment_document),
                    )
                )
            job = create_job(
                job_template=job_template,
                job_parameter_values={
                    name: _parameter_value(name, value) for name, value in parameters.items()
                },
                environment_templates=environment_templates,
            )
        except ValueError as error:
            if not validate:
                return ""
            raise _validation_exception(
                f"Invalid job template or parameters: {error}", "CreateJob"
            ) from error
        return job.name

    @route("POST", "/farms/{farmId}/queues/{queueId}/jobs", "CreateJob")
    def create_job(
        self, *, farmId: str, queueId: str, priority: int, **kwargs: Any
    ) -> dict[str, Any]:
        key = self._queue_key(farmId, queueId, "CreateJob")
        template = kwargs.get("template")
        if template is None:
            raise _validation_exception(
                "The mock Deadline service only supports CreateJob with a template, "
                "not sourceJobId.",
                "CreateJob",
            )
        name = self._job_name(
            key, template, kwargs.get("templateType"), kwargs.get("parameters") or {}
        )
        job_id = f"job-{uuid.uuid4().hex}"
        job = {
            **kwargs,
            "farmId": farmId,
            "queueId": queueId,
            "jobId": job_id,
            # GetJob requires a non-empty name; the job ID stands in when validation is
            # off and the template has no readable name.
            "name": kwargs.get("nameOverride") or name or job_id,
            "priority": priority,
            # Created instantly, so the client's wait for CREATE_COMPLETE ends on its
            # first GetJob.
            "lifecycleStatus": "CREATE_COMPLETE",
            "lifecycleStatusMessage": "Job creation completed successfully",
            "taskRunStatus": kwargs.get("targetTaskRunStatus") or "READY",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "createdBy": _FAKE_PRINCIPAL,
        }
        with self._lock:
            self.jobs[(farmId, queueId, job_id)] = job
        return {"jobId": job_id}

    @route("GET", "/farms/{farmId}/queues/{queueId}/jobs/{jobId}", "GetJob")
    def get_job(self, *, farmId: str, queueId: str, jobId: str) -> dict[str, Any]:
        self._queue_key(farmId, queueId, "GetJob")
        try:
            return dict(self.jobs[(farmId, queueId, jobId)])
        except KeyError:
            raise _resource_not_found("job", jobId, "GetJob")

    # Worker APIs. The worker protocol state lives in self.worker_scheduler.

    @route("POST", "/farms/{farmId}/fleets/{fleetId}/workers", "CreateWorker")
    def create_worker(self, *, farmId: str, fleetId: str, **kwargs: Any) -> dict[str, Any]:
        return self.worker_scheduler.create_worker(farmId=farmId, fleetId=fleetId, **kwargs)

    @route("PATCH", "/farms/{farmId}/fleets/{fleetId}/workers/{workerId}", "UpdateWorker")
    def update_worker(
        self, *, farmId: str, fleetId: str, workerId: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.worker_scheduler.update_worker(
            farmId=farmId, fleetId=fleetId, workerId=workerId, **kwargs
        )

    @route("DELETE", "/farms/{farmId}/fleets/{fleetId}/workers/{workerId}", "DeleteWorker")
    def delete_worker(self, *, farmId: str, fleetId: str, workerId: str) -> dict[str, Any]:
        return self.worker_scheduler.delete_worker(
            farmId=farmId, fleetId=fleetId, workerId=workerId
        )

    @route(
        "PATCH",
        "/farms/{farmId}/fleets/{fleetId}/workers/{workerId}/schedule",
        "UpdateWorkerSchedule",
    )
    def update_worker_schedule(
        self, *, farmId: str, fleetId: str, workerId: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.worker_scheduler.update_worker_schedule(
            farmId=farmId,
            fleetId=fleetId,
            workerId=workerId,
            updatedSessionActions=kwargs.get("updatedSessionActions"),
        )

    @route(
        "POST",
        "/farms/{farmId}/fleets/{fleetId}/workers/{workerId}/batchGetJobEntity",
        "BatchGetJobEntity",
    )
    def batch_get_job_entity(
        self, *, farmId: str, fleetId: str, workerId: str, identifiers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self.worker_scheduler.batch_get_job_entity(
            farmId=farmId, fleetId=fleetId, workerId=workerId, identifiers=identifiers
        )

    @route(
        "GET",
        "/farms/{farmId}/fleets/{fleetId}/workers/{workerId}/fleet-roles",
        "AssumeFleetRoleForWorker",
    )
    def assume_fleet_role_for_worker(
        self, *, farmId: str, fleetId: str, workerId: str
    ) -> dict[str, Any]:
        return self.worker_scheduler.assume_fleet_role_for_worker(
            farmId=farmId, fleetId=fleetId, workerId=workerId
        )

    @route(
        "GET",
        "/farms/{farmId}/fleets/{fleetId}/workers/{workerId}/queue-roles",
        "AssumeQueueRoleForWorker",
    )
    def assume_queue_role_for_worker(
        self, *, farmId: str, fleetId: str, workerId: str, queueId: str
    ) -> dict[str, Any]:
        return self.worker_scheduler.assume_queue_role_for_worker(
            farmId=farmId, fleetId=fleetId, workerId=workerId, queueId=queueId
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Type {type(value)} is not JSON serializable")


def _discover_routes(backend: MockDeadlineBackend) -> list[tuple[str, re.Pattern, Callable, str]]:
    routes = []
    for name in dir(backend):
        function = getattr(backend, name)
        route_info = getattr(function, "__http_route__", None)
        if route_info is None:
            continue
        method, path, operation = route_info
        pattern = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path) + "$")
        routes.append((method, pattern, function, operation))
    return routes


class _ResponseValidator:
    """Filter responses to the installed botocore model, then validate them."""

    _TYPE_DEFAULTS: ClassVar[dict[str, Any]] = {
        "string": "",
        "integer": 0,
        "long": 0,
        "float": 0.0,
        "double": 0.0,
        "boolean": False,
        "timestamp": datetime(1970, 1, 1, tzinfo=timezone.utc),
        "list": [],
        "map": {},
        "structure": {},
    }

    def __init__(self) -> None:
        session = botocore.session.get_session()
        loader = session.get_component("data_loader")
        self._model = ServiceModel(loader.load_service_model("deadline", "service-2"))
        self._validator = ParamValidator()

    def _filter(self, shape: Any, value: Any) -> Any:
        if shape is None or value is None:
            return value
        if getattr(shape, "is_document_type", False):
            # A document (such as a step or environment template) is free-form JSON.
            return value
        if shape.type_name == "structure":
            filtered = {
                key: self._filter(shape.members[key], child)
                for key, child in value.items()
                if key in shape.members
            }
            for required in getattr(shape, "required_members", ()):
                if required not in filtered and required in shape.members:
                    filtered[required] = self._TYPE_DEFAULTS.get(shape.members[required].type_name)
            return filtered
        if shape.type_name == "list":
            return [self._filter(shape.member, child) for child in value]
        if shape.type_name == "map":
            return {key: self._filter(shape.value, child) for key, child in value.items()}
        return value

    def filter_and_validate(self, operation_name: str, response: dict[str, Any]) -> dict[str, Any]:
        output_shape = self._model.operation_model(operation_name).output_shape
        if output_shape is None:
            return response
        filtered = self._filter(output_shape, response)
        report = self._validator.validate(filtered, output_shape)
        if report.has_errors():
            raise ValueError(
                f"Mock response for {operation_name} failed validation: "
                f"{report.generate_report()}"
            )
        return filtered


def _make_handler(
    routes: list[tuple[str, re.Pattern, Callable, str]],
    validator: _ResponseValidator | None,
    backend: MockDeadlineBackend,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, body: Any, *, error_code: str | None = None) -> None:
            payload = json.dumps(body, default=_json_default).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if error_code:
                    self.send_header("x-amzn-errortype", error_code)
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # DCC clients can abandon in-flight requests while closing or reloading.
                pass

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                self.close_connection = True

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            if method == "GET" and parsed.path == ADMIN_STATE_PATH:
                self._send_json(200, backend.snapshot())
                return
            if method == "POST" and parsed.path == ADMIN_RESET_PATH:
                backend.reset()
                self._send_json(200, backend.snapshot())
                return

            for route_method, pattern, function, operation in routes:
                match = pattern.match(parsed.path)
                if route_method != method or match is None:
                    continue
                delay = backend.response_delay_s + backend.operation_delays.get(operation, 0.0)
                if delay:
                    time.sleep(delay)
                with backend._lock:
                    backend.call_counts[operation] = backend.call_counts.get(operation, 0) + 1
                    backend.request_log.append((method, parsed.path, operation))
                try:
                    arguments: dict[str, Any] = dict(match.groupdict())
                    for key, values in parse_qs(parsed.query).items():
                        arguments[key] = int(values[0]) if key in _INT_QUERY_PARAMS else values[0]
                    length = int(self.headers.get("Content-Length", 0))
                    if length > _MAX_BODY_BYTES:
                        self._send_json(413, {"message": "Payload too large"})
                        return
                    if length:
                        body = json.loads(self.rfile.read(length))
                        if not isinstance(body, dict):
                            raise ValueError("Request body must be a JSON object")
                        arguments.update(body)
                except (TypeError, ValueError) as error:
                    self._send_json(
                        400,
                        {"message": f"Invalid request: {error}"},
                        error_code="ValidationException",
                    )
                    return
                try:
                    injected = backend._take_injected_error(operation)
                    if injected is not None:
                        raise injected
                    result = function(**arguments)
                    if validator is not None:
                        result = validator.filter_and_validate(operation, result)
                    backend._log(f"served {operation} ({method} {parsed.path}) -> 200")
                    self._send_json(200, result)
                except ClientError as error:
                    details = error.response["Error"]
                    code = details.get("Code", "InternalServerException")
                    status = _ERROR_STATUS.get(code, 400)
                    # Modeled members such as a ConflictException's reason and
                    # resourceId go in the body, where botocore parses them.
                    body = {
                        key: value
                        for key, value in details.items()
                        if key not in ("Code", "Message")
                    }
                    body["message"] = details.get("Message", "")
                    self._send_json(status, body, error_code=code)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    self.close_connection = True
                except Exception as error:
                    traceback.print_exc()
                    try:
                        self._send_json(
                            500,
                            {"message": str(error)},
                            error_code="InternalServerException",
                        )
                    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                        self.close_connection = True
                return

            with backend._lock:
                backend.unmatched_requests.append((method, parsed.path))
            backend._log(f"UNMATCHED {method} {parsed.path} -> 404")
            sys.stderr.write(f"[mock-deadline] 404 NO ROUTE {method} {parsed.path}\n")
            sys.stderr.flush()
            self._send_json(
                404,
                {"message": f"No route for {method} {parsed.path}"},
                error_code="NotFoundException",
            )

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_PATCH(self) -> None:
            self._dispatch("PATCH")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

    return Handler


class _LoopbackHTTPServer(ThreadingHTTPServer):
    """Bind the offline server without HTTPServer's reverse-DNS lookup."""

    def server_bind(self) -> None:
        if self.server_address[0] != "127.0.0.1":
            raise ValueError("The offline mock must bind to IPv4 loopback")
        TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = int(self.server_address[1])


def start_server(
    backend: MockDeadlineBackend,
    port: int = 0,
) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    """Serve a mock Deadline backend over REST-JSON."""
    routes = _discover_routes(backend)
    validator = _ResponseValidator() if backend.validate_responses else None
    server = _LoopbackHTTPServer(
        ("127.0.0.1", port),
        _make_handler(routes, validator, backend),
    )
    actual_port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{actual_port}", thread
