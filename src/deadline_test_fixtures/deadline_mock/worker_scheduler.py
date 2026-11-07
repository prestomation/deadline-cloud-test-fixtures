# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Service side of the Deadline Cloud worker protocol, for hermetic worker tests.

:class:`MockWorkerScheduler` holds what the service knows about workers, sessions and
session actions. It answers the worker APIs (``CreateWorker``, ``UpdateWorker``,
``UpdateWorkerSchedule``, ``BatchGetJobEntity``, ``AssumeFleetRoleForWorker``,
``AssumeQueueRoleForWorker`` and ``DeleteWorker``) through
:class:`~deadline_test_fixtures.deadline_mock.MockDeadlineBackend`.

It does not decide on its own what a worker runs. A test decides, by calling
:meth:`MockWorkerScheduler.open_session`, :meth:`~MockWorkerScheduler.add_actions`,
:meth:`~MockWorkerScheduler.cancel_action`, :meth:`~MockWorkerScheduler.close_session`
and :meth:`~MockWorkerScheduler.drain`. The scheduler then sends those decisions to the
worker in ``UpdateWorkerSchedule`` responses, in the same shape the service uses.

Every report that a worker sends is checked against the rules in the worker agent's
``docs/dev/worker_api_protocol.md``. A broken rule is added to
:attr:`MockWorkerScheduler.violations` with the rule's name, so a test can assert that
the list is empty. The scheduler never raises for a broken rule: the worker keeps
running, so one run can find more than one problem.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from botocore.exceptions import ClientError

ActionKind = Literal["envEnter", "envExit", "taskRun", "syncInputJobAttachments"]

TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELED", "NEVER_ATTEMPTED"})
"""Values of ``completedStatus`` in ``UpdateWorkerSchedule`` requests."""

_UNSUCCESSFUL = frozenset({"FAILED", "INTERRUPTED", "CANCELED"})
_TERMINAL_QUEUE_ROLE_ERRORS = frozenset({"ResourceNotFoundException"})
"""AssumeQueueRoleForWorker errors after which the protocol says to fail the session.
For AccessDeniedException and most conflicts the worker may also run without
credentials, so those are not checked."""
_ONLINE_STATUSES = frozenset({"STARTED", "STOPPING"})
_WORKER_STATUS_ORDER = ("CREATED", "STARTED", "STOPPING", "STOPPED")

_FAKE_CREDENTIALS_KEY = "ASIAWORKERPROTOCOLMOCK"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_datetime(value: Any) -> datetime | None:
    """Parse a timestamp from a REST-JSON request body (epoch seconds or ISO 8601)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _client_error(operation: str, code: str, message: str, **extra: Any) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message, **extra}}, operation)


def _status_conflict(operation: str, worker_id: str, status: str) -> ClientError:
    return _client_error(
        operation,
        "ConflictException",
        f"Worker {worker_id} is {status}.",
        reason="STATUS_CONFLICT",
        resourceId=worker_id,
        resourceType="worker",
        context={"status": status},
    )


@dataclass(frozen=True)
class ProtocolViolation:
    """One broken rule of the worker protocol.

    ``rule`` is a short, stable name for the rule, for example
    ``never-attempted-has-times``. ``detail`` says what the worker did.
    """

    rule: str
    detail: str
    session_id: str | None = None
    session_action_id: str | None = None

    def __str__(self) -> str:
        where = "/".join(part for part in (self.session_id, self.session_action_id) if part)
        return f"[{self.rule}] {where}: {self.detail}" if where else f"[{self.rule}] {self.detail}"


@dataclass
class MockSessionAction:
    """What the service knows about one session action."""

    session_action_id: str
    kind: ActionKind
    definition: dict[str, Any]
    status: str = "ASSIGNED"
    """``ASSIGNED`` (sent, nothing reported), ``RUNNING`` (reported without a
    ``completedStatus``), or a value of :data:`TERMINAL_STATUSES`."""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    process_exit_code: int | None = None
    progress_message: str | None = None
    canceled_by_service: bool = False
    first_sent_in_call: int | None = None
    """Number of the ``UpdateWorkerSchedule`` call whose response first held this action."""
    never_attempted_received_at: datetime | None = None
    reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def environment_id(self) -> str | None:
        if self.kind in ("envEnter", "envExit"):
            return str(self.definition["environmentId"])
        return None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def was_attempted(self) -> bool:
        """True if the worker started the action, whatever its outcome."""
        return self.status not in ("ASSIGNED", "NEVER_ATTEMPTED")

    def worker_could_report(self, worker: MockWorker) -> bool:
        """True if the worker made a schedule call after it first received this action."""
        return self.first_sent_in_call is not None and (
            self.first_sent_in_call < worker.schedule_calls - 1
        )

    def to_assignment(self) -> dict[str, Any]:
        return {
            "sessionActionId": self.session_action_id,
            "definition": {self.kind: self.definition},
        }


@dataclass
class MockWorkerSession:
    """What the service knows about one session on one worker."""

    session_id: str
    worker_key: tuple[str, str, str]
    queue_id: str
    job_id: str
    actions: list[MockSessionAction] = field(default_factory=list)
    log_configuration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] | None = None
    closing: bool = False
    ended: bool = False
    opened_at: datetime = field(default_factory=_now)
    queue_role_error: str | None = None
    """Error code of a failed ``AssumeQueueRoleForWorker`` call made while this session
    was waiting to start. After a terminal error the session must not run."""
    accepts_work: bool = True
    """False once the service canceled work in, or closed, the session. After that
    the service only adds ``envExit`` actions."""
    blocked_at: int | None = None
    """Index of the first non-``envExit`` action that ended unsuccessfully. After it,
    the service may only add ``envExit`` actions."""

    def index_of(self, session_action_id: str) -> int:
        for index, action in enumerate(self.actions):
            if action.session_action_id == session_action_id:
                return index
        raise KeyError(session_action_id)

    @property
    def all_terminal(self) -> bool:
        return all(action.is_terminal for action in self.actions)

    def entered_environment_ids(self) -> list[str]:
        """Environments whose ``envEnter`` was attempted, in the order entered."""
        return [
            str(action.environment_id)
            for action in self.actions
            if action.kind == "envEnter" and action.was_attempted
        ]

    def environments_needing_exit(self) -> list[str]:
        """Environments whose ``envEnter`` started a process or succeeded.

        A worker may report an ``envEnter`` as ``FAILED`` without running it, for example
        when it cannot fetch ``jobDetails`` for a new session. Such a report has no
        ``processExitCode``, and nothing needs to be exited.
        """
        return [
            str(action.environment_id)
            for action in self.actions
            if action.kind == "envEnter"
            and (action.status == "SUCCEEDED" or action.process_exit_code is not None)
        ]

    def queued_exit_ids(self) -> set[str]:
        return {str(action.environment_id) for action in self.actions if action.kind == "envExit"}


@dataclass
class MockWorker:
    farm_id: str
    fleet_id: str
    worker_id: str
    status: str = "CREATED"
    status_history: list[str] = field(default_factory=lambda: ["CREATED"])
    host_properties: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None
    drain_requested: bool = False
    stop_sent: bool = False
    schedule_calls: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.farm_id, self.fleet_id, self.worker_id)


class MockWorkerScheduler:
    """The service side of the worker protocol. See the module docstring."""

    def __init__(
        self,
        *,
        update_interval_seconds: int = 1,
        queue_role_arns: Mapping[str, str] | None = None,
        log_group_name: str = "/aws/deadline/mock",
    ) -> None:
        self.update_interval_seconds = update_interval_seconds
        self.queue_role_arns = dict(queue_role_arns or {})
        self.log_group_name = log_group_name
        self._lock = threading.RLock()
        self._schedule_called = threading.Condition(self._lock)
        self.workers: dict[tuple[str, str, str], MockWorker] = {}
        self.sessions: dict[str, MockWorkerSession] = {}
        self._action_index: dict[str, str] = {}
        self.job_entities: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.violations: list[ProtocolViolation] = []
        self._entity_errors: dict[tuple[str, str, str], list[str]] = {}
        self._queue_role_errors: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        # (worker key, job ID, entity kind, entity ID) -> list of (time, outcome code).
        self.entity_fetches: dict[tuple[tuple[str, str, str], str, str, str], list] = {}
        self.queue_role_calls: list[tuple[str, str, str]] = []
        """(worker ID, queue ID, outcome) for every AssumeQueueRoleForWorker call."""

    # ------------------------------------------------------------------
    # Test-side API: register job data and decide what the worker runs.
    # ------------------------------------------------------------------

    def add_job(
        self,
        *,
        job_id: str,
        job_details: Mapping[str, Any],
        steps: Mapping[str, Mapping[str, Any]] | None = None,
        environments: Mapping[str, Mapping[str, Any]] | None = None,
        job_attachment_details: Mapping[str, Any] | None = None,
    ) -> None:
        """Register the entities that ``BatchGetJobEntity`` returns for a job.

        ``job_details``, each value of ``steps`` (keyed by step ID) and each value of
        ``environments`` (keyed by environment ID) are the entity bodies as the service
        returns them. ``jobId``, ``stepId`` and ``environmentId`` are filled in.
        """
        with self._lock:
            self.job_entities[("jobDetails", job_id, job_id)] = {**job_details, "jobId": job_id}
            for step_id, step in (steps or {}).items():
                self.job_entities[("stepDetails", job_id, step_id)] = {
                    "dependencies": [],
                    **step,
                    "jobId": job_id,
                    "stepId": step_id,
                }
            for environment_id, environment in (environments or {}).items():
                self.job_entities[("environmentDetails", job_id, environment_id)] = {
                    **environment,
                    "jobId": job_id,
                    "environmentId": environment_id,
                }
            if job_attachment_details is not None:
                self.job_entities[("jobAttachmentDetails", job_id, job_id)] = {
                    **job_attachment_details,
                    "jobId": job_id,
                }

    def register_worker(
        self, *, farm_id: str, fleet_id: str, worker_id: str | None = None, status: str = "CREATED"
    ) -> str:
        """Create a worker without a ``CreateWorker`` call. Returns its ID."""
        worker_id = worker_id or f"worker-{uuid.uuid4().hex}"
        with self._lock:
            worker = MockWorker(farm_id=farm_id, fleet_id=fleet_id, worker_id=worker_id)
            worker.status = status
            worker.status_history = [status]
            self.workers[worker.key] = worker
        return worker_id

    def open_session(
        self,
        *,
        worker_id: str,
        queue_id: str,
        job_id: str,
        actions: Iterable[tuple[ActionKind, Mapping[str, Any]]] = (),
        log_configuration: Mapping[str, Any] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Assign a new session to a worker. Returns the session ID.

        Each action is ``(kind, definition)``, for example
        ``("envEnter", {"environmentId": "JOB:job-1:Env"})``.
        """
        with self._lock:
            worker = self._worker_by_id(worker_id)
            session_id = f"session-{uuid.uuid4().hex}"
            log_stream = f"{session_id}"
            self.sessions[session_id] = MockWorkerSession(
                session_id=session_id,
                worker_key=worker.key,
                queue_id=queue_id,
                job_id=job_id,
                log_configuration=dict(
                    log_configuration
                    or {
                        "logDriver": "awslogs",
                        "options": {
                            "logGroupName": self.log_group_name,
                            "logStreamName": log_stream,
                        },
                        "parameters": {"interval": "15"},
                    }
                ),
                metadata=dict(metadata) if metadata is not None else None,
            )
            self.add_actions(session_id, actions)
            return session_id

    def add_actions(
        self, session_id: str, actions: Iterable[tuple[ActionKind, Mapping[str, Any]]]
    ) -> list[str]:
        """Append actions to a session. Returns their session action IDs.

        Raises ``ValueError`` if the protocol forbids the service from adding the action:
        after an unsuccessful action, the service may only add ``envExit`` actions.
        """
        with self._lock:
            session = self.sessions[session_id]
            if session.ended:
                raise ValueError(f"Session {session_id} has ended")
            added = []
            for kind, definition in actions:
                if kind != "envExit" and (
                    session.blocked_at is not None or not session.accepts_work
                ):
                    raise ValueError(
                        f"Session {session_id} is closed, canceled or had an unsuccessful "
                        f"action; the service may only add envExit actions, not {kind}"
                    )
                action_id = f"sessionaction-{session_id[len('session-') :]}-{len(session.actions)}"
                session.actions.append(
                    MockSessionAction(
                        session_action_id=action_id, kind=kind, definition=dict(definition)
                    )
                )
                self._action_index[action_id] = session_id
                added.append(action_id)
            return added

    def cancel_action(self, session_action_id: str) -> list[str]:
        """Cancel an action the way the service does. Returns the IDs it cancels.

        Canceling a non-``envExit`` action also cancels every later non-``envExit``
        action in the session. Canceling an ``envExit`` cancels only that action.
        Actions that already ended are left alone.
        """
        with self._lock:
            session = self.sessions[self._action_index[session_action_id]]
            index = session.index_of(session_action_id)
            target = session.actions[index]
            if target.kind == "envExit":
                chosen = [target]
            else:
                session.accepts_work = False
                chosen = [target] + [
                    action for action in session.actions[index + 1 :] if action.kind != "envExit"
                ]
            canceled = []
            for action in chosen:
                if not action.is_terminal:
                    action.canceled_by_service = True
                    canceled.append(action.session_action_id)
            return canceled

    def close_session(self, session_id: str) -> list[str]:
        """End a session once its work is done. Returns the IDs of added ``envExit`` actions.

        Like the service, the mock queues ``envExit`` actions only once the worker has
        reported every ``envEnter``: one for each environment whose ``envEnter`` was
        attempted, inner-most first. Until then this returns an empty list, and the
        exits are queued when the last ``envEnter`` report arrives. The session leaves
        ``UpdateWorkerSchedule`` responses once every action has ended.
        """
        with self._lock:
            session = self.sessions[session_id]
            session.closing = True
            session.accepts_work = False
            return self._queue_closing_exits(session)

    def _queue_closing_exits(self, session: MockWorkerSession) -> list[str]:
        if not session.closing or any(
            action.kind == "envEnter" and not action.is_terminal for action in session.actions
        ):
            return []
        return self._queue_exits(session, session.entered_environment_ids())

    def drain(self, worker_id: str, *, immediate: bool) -> None:
        """Start a service-initiated drain.

        An immediate drain cancels all active work. An eventual drain lets assigned work
        finish. Either way every session is closed, and once the worker has no sessions
        the next ``UpdateWorkerSchedule`` response sets ``desiredWorkerStatus=STOPPED``.
        """
        with self._lock:
            worker = self._worker_by_id(worker_id)
            worker.drain_requested = True
            for session in self._sessions_for(worker.key):
                if immediate:
                    for action in session.actions:
                        if action.kind != "envExit" and not action.is_terminal:
                            self.cancel_action(action.session_action_id)
                            break
                self.close_session(session.session_id)

    def wait_for_schedule_calls(self, worker_id: str, count: int, timeout: float) -> bool:
        """Wait until the worker made ``count`` more ``UpdateWorkerSchedule`` calls."""
        with self._schedule_called:
            worker = self._worker_by_id(worker_id)
            target = worker.schedule_calls + count
            return self._schedule_called.wait_for(
                lambda: worker.schedule_calls >= target, timeout=timeout
            )

    def wait_until(self, predicate: Any, timeout: float) -> bool:
        """Wait until ``predicate(self)`` is true; it is checked after each schedule call."""
        with self._schedule_called:
            return self._schedule_called.wait_for(lambda: predicate(self), timeout=timeout)

    def action(self, session_action_id: str) -> MockSessionAction:
        with self._lock:
            session = self.sessions[self._action_index[session_action_id]]
            return session.actions[session.index_of(session_action_id)]

    def inject_entity_error(
        self, *, kind: str, job_id: str, entity_id: str, code: str, count: int = 1
    ) -> None:
        """Make ``BatchGetJobEntity`` return ``code`` in ``errors`` for one entity.

        ``kind`` is ``jobDetails``, ``stepDetails``, ``environmentDetails`` or
        ``jobAttachmentDetails``. ``entity_id`` is the job, step or environment ID.
        The error is returned for the next ``count`` requests of that entity.
        """
        with self._lock:
            self._entity_errors.setdefault((kind, job_id, entity_id), []).extend([code] * count)

    def inject_queue_role_error(
        self, queue_id: str, code: str, *, count: int = 1, **fields: Any
    ) -> None:
        """Make the next ``count`` ``AssumeQueueRoleForWorker`` calls for a queue fail.

        ``fields`` become members of the error, for example ``reason="STATUS_CONFLICT"``
        and ``resourceId=...`` for a ``ConflictException``.
        """
        with self._lock:
            self._queue_role_errors.setdefault(queue_id, []).extend([(code, fields)] * count)

    def mark_not_responding(self, worker_id: str) -> None:
        """Set the worker to ``NOT_RESPONDING``, as the service does when heartbeats stop.

        Later worker calls get a ``STATUS_CONFLICT``. The service takes the work back,
        so the worker's sessions end without further reports.
        """
        with self._lock:
            worker = self._worker_by_id(worker_id)
            worker.status = "NOT_RESPONDING"
            worker.status_history.append("NOT_RESPONDING")
            for session in self._sessions_for(worker.key):
                session.ended = True

    def forget_worker(self, worker_id: str) -> None:
        """Delete the worker on the service side. Later worker calls get a 404."""
        with self._lock:
            worker = self._worker_by_id(worker_id)
            for session in self._sessions_for(worker.key):
                session.ended = True
            del self.workers[worker.key]

    def check_session_complete(self, session_id: str, *, worker_drained: bool = False) -> None:
        """Add a violation for each rule a finished session breaks.

        Call this when the test believes the session is over. ``worker_drained`` is
        true when the worker ran a worker-initiated drain, which may skip ``envExit``.
        """
        with self._lock:
            session = self.sessions[session_id]
            worker = self.workers.get(session.worker_key)
            for action in session.actions:
                if not action.is_terminal:
                    if worker_drained and worker and not action.worker_could_report(worker):
                        # Sent in the response to the worker's last call, so the worker
                        # never had a chance to report it.
                        continue
                    self._violate(
                        "action-never-completed",
                        f"{action.kind} is still {action.status}",
                        session,
                        action,
                    )
            if worker_drained:
                return
            exited = {
                str(action.environment_id)
                for action in session.actions
                if action.kind == "envExit" and action.was_attempted
            }
            canceled_exits = {
                str(action.environment_id)
                for action in session.actions
                if action.kind == "envExit" and action.canceled_by_service
            }
            for environment_id in session.environments_needing_exit():
                if environment_id not in exited and environment_id not in canceled_exits:
                    self._violate(
                        "entered-environment-not-exited",
                        f"environment {environment_id} was entered but its envExit never ran",
                        session,
                    )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "workers": [
                    {
                        "farmId": worker.farm_id,
                        "fleetId": worker.fleet_id,
                        "workerId": worker.worker_id,
                        "status": worker.status,
                        "statusHistory": list(worker.status_history),
                    }
                    for worker in self.workers.values()
                ],
                "sessions": [
                    {
                        "sessionId": session.session_id,
                        "workerId": session.worker_key[2],
                        "jobId": session.job_id,
                        "ended": session.ended,
                        "actions": [
                            {
                                "sessionActionId": action.session_action_id,
                                "kind": action.kind,
                                "status": action.status,
                                "canceledByService": action.canceled_by_service,
                            }
                            for action in session.actions
                        ],
                    }
                    for session in self.sessions.values()
                ],
                "violations": [str(violation) for violation in self.violations],
            }

    # ------------------------------------------------------------------
    # Worker APIs, called by MockDeadlineBackend.
    # ------------------------------------------------------------------

    def create_worker(self, *, farmId: str, fleetId: str, **kwargs: Any) -> dict[str, Any]:
        worker_id = self.register_worker(farm_id=farmId, fleet_id=fleetId)
        with self._lock:
            self.workers[(farmId, fleetId, worker_id)].host_properties = kwargs.get(
                "hostProperties"
            )
        return {"workerId": worker_id}

    def update_worker(
        self, *, farmId: str, fleetId: str, workerId: str, **kwargs: Any
    ) -> dict[str, Any]:
        with self._lock:
            worker = self._worker(farmId, fleetId, workerId, "UpdateWorker")
            if "capabilities" in kwargs:
                worker.capabilities = kwargs["capabilities"]
            if "hostProperties" in kwargs:
                worker.host_properties = kwargs["hostProperties"]
            status = kwargs.get("status")
            response: dict[str, Any] = {}
            if status is None:
                return response
            if status == "STARTED" and worker.status not in ("CREATED", "STOPPED", "STARTED"):
                raise _status_conflict("UpdateWorker", workerId, worker.status)
            if status == "STOPPED":
                for session in self._sessions_for(worker.key):
                    for action in session.actions:
                        if not action.is_terminal and action.worker_could_report(worker):
                            self._violate(
                                "stopped-with-open-actions",
                                f"worker set STOPPED while {action.kind} is {action.status}",
                                session,
                                action,
                            )
                    session.ended = True
            if status == "STARTED":
                worker.drain_requested = False
                worker.stop_sent = False
                response["log"] = {
                    "logDriver": "awslogs",
                    "options": {
                        "logGroupName": self.log_group_name,
                        "logStreamName": workerId,
                    },
                    "parameters": {},
                }
            worker.status = status
            worker.status_history.append(status)
            return response

    def delete_worker(self, *, farmId: str, fleetId: str, workerId: str) -> dict[str, Any]:
        with self._lock:
            worker = self._worker(farmId, fleetId, workerId, "DeleteWorker")
            if worker.status in _ONLINE_STATUSES:
                raise _status_conflict("DeleteWorker", workerId, worker.status)
            del self.workers[worker.key]
            return {}

    def assume_fleet_role_for_worker(
        self, *, farmId: str, fleetId: str, workerId: str
    ) -> dict[str, Any]:
        with self._lock:
            self._worker(farmId, fleetId, workerId, "AssumeFleetRoleForWorker")
        return {"credentials": self._credentials()}

    def assume_queue_role_for_worker(
        self, *, farmId: str, fleetId: str, workerId: str, queueId: str
    ) -> dict[str, Any]:
        with self._lock:
            try:
                worker = self._worker(farmId, fleetId, workerId, "AssumeQueueRoleForWorker")
                if worker.status not in _ONLINE_STATUSES:
                    raise _status_conflict("AssumeQueueRoleForWorker", workerId, worker.status)
                sessions = [
                    session
                    for session in self._sessions_for(worker.key)
                    if session.queue_id == queueId
                ]
                if not sessions:
                    raise _client_error(
                        "AssumeQueueRoleForWorker",
                        "AccessDeniedException",
                        f"Worker {workerId} has no session for queue {queueId}",
                    )
                pending = self._queue_role_errors.get(queueId)
                if pending:
                    code, fields = pending.pop(0)
                    # Blame only sessions the worker has received and not started.
                    for session in sessions:
                        received = any(a.first_sent_in_call is not None for a in session.actions)
                        started = any(a.was_attempted for a in session.actions)
                        if received and not started:
                            session.queue_role_error = code
                    raise _client_error(
                        "AssumeQueueRoleForWorker", code, "Injected by the test", **fields
                    )
            except ClientError as error:
                self.queue_role_calls.append(
                    (workerId, queueId, str(error.response["Error"]["Code"]))
                )
                raise
            self.queue_role_calls.append((workerId, queueId, "OK"))
            # A later successful call gives the waiting sessions their credentials.
            for session in sessions:
                session.queue_role_error = None
        if queueId not in self.queue_role_arns:
            return {}
        return {"credentials": self._credentials(f"ASIAQUEUE{queueId[-8:].upper()}")}

    def batch_get_job_entity(
        self, *, farmId: str, fleetId: str, workerId: str, identifiers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        with self._lock:
            worker = self._worker(farmId, fleetId, workerId, "BatchGetJobEntity")
            jobs = {session.job_id for session in self._sessions_for(worker.key)}
            entities: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for identifier in identifiers:
                ((kind, fields),) = identifier.items()
                job_id = fields["jobId"]
                entity_id = fields.get("stepId") or fields.get("environmentId") or job_id
                entity = self.job_entities.get((kind, job_id, entity_id))
                injected = self._entity_errors.get((kind, job_id, entity_id))
                if job_id not in jobs:
                    code, message = "AccessDeniedException", "No session for this job"
                elif injected:
                    code, message = injected.pop(0), "Injected by the test"
                elif entity is None:
                    code, message = "ResourceNotFoundException", f"No {kind} for {entity_id}"
                else:
                    entities.append({kind: dict(entity)})
                    self._record_fetch(worker, kind, job_id, entity_id, "OK")
                    continue
                self._record_fetch(worker, kind, job_id, entity_id, code)
                errors.append({kind: {**fields, "code": code, "message": message}})
            return {"entities": entities, "errors": errors}

    def update_worker_schedule(
        self,
        *,
        farmId: str,
        fleetId: str,
        workerId: str,
        updatedSessionActions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            try:
                worker = self._worker(farmId, fleetId, workerId, "UpdateWorkerSchedule")
                if worker.status not in _ONLINE_STATUSES:
                    raise _status_conflict("UpdateWorkerSchedule", workerId, worker.status)
                if worker.stop_sent:
                    self._violate(
                        "schedule-after-stop",
                        "UpdateWorkerSchedule called after desiredWorkerStatus=STOPPED",
                    )
                self._apply_updates(worker, dict(updatedSessionActions or {}))
                return self._schedule_for(worker)
            finally:
                worker_or_none = self.workers.get((farmId, fleetId, workerId))
                if worker_or_none is not None:
                    worker_or_none.schedule_calls += 1
                self._schedule_called.notify_all()

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _record_fetch(
        self, worker: MockWorker, kind: str, job_id: str, entity_id: str, outcome: str
    ) -> None:
        self.entity_fetches.setdefault((worker.key, job_id, kind, entity_id), []).append(
            (_now(), outcome)
        )

    def _entity_outcome_before(
        self, session: MockWorkerSession, kind: str, entity_id: str, when: datetime
    ) -> str | None:
        """How the worker's fetches of an entity went between the session opening and ``when``.

        Returns ``"OK"`` if any fetch succeeded, the last error code if all failed, or
        ``None`` if there was no fetch. The worker caches entities per session, but the
        service cannot tell which session a fetch is for, so one success is enough.
        """
        fetches = self.entity_fetches.get((session.worker_key, session.job_id, kind, entity_id))
        outcome = None
        for fetched_at, result in fetches or ():
            if session.opened_at <= fetched_at <= when + timedelta(seconds=1):
                if result == "OK":
                    return "OK"
                outcome = result
        return outcome

    def _check_entities(
        self,
        session: MockWorkerSession,
        action: MockSessionAction,
        completed: str | None,
        started_at: datetime | None,
    ) -> None:
        """An action whose details could not be fetched must not run or succeed."""
        if completed not in (None, "SUCCEEDED") or action.kind == "envExit":
            return
        when = started_at or _now()
        needed = [("jobDetails", session.job_id)]
        if action.kind == "taskRun":
            needed.append(("stepDetails", str(action.definition["stepId"])))
        elif action.kind == "envEnter":
            needed.append(("environmentDetails", str(action.environment_id)))
        for kind, entity_id in needed:
            outcome = self._entity_outcome_before(session, kind, entity_id, when)
            if outcome not in (None, "OK", "MaxPayloadSizeExceeded"):
                self._violate(
                    "ran-despite-entity-error",
                    f"{action.kind} was {completed or 'RUNNING'} though its {kind} "
                    f"fetch returned {outcome}",
                    session,
                    action,
                )

    def _violate(
        self,
        rule: str,
        detail: str,
        session: MockWorkerSession | None = None,
        action: MockSessionAction | None = None,
    ) -> None:
        self.violations.append(
            ProtocolViolation(
                rule=rule,
                detail=detail,
                session_id=session.session_id if session else None,
                session_action_id=action.session_action_id if action else None,
            )
        )

    def _credentials(self, access_key_id: str = _FAKE_CREDENTIALS_KEY) -> dict[str, Any]:
        """Credentials whose access key ID says whose they are, so a test can tell
        fleet credentials and each queue's credentials apart."""
        return {
            "accessKeyId": access_key_id,
            "secretAccessKey": "mock-secret",
            "sessionToken": "mock-session-token",
            "expiration": _now() + timedelta(hours=1),
        }

    def _worker_by_id(self, worker_id: str) -> MockWorker:
        for worker in self.workers.values():
            if worker.worker_id == worker_id:
                return worker
        raise KeyError(worker_id)

    def _worker(self, farm_id: str, fleet_id: str, worker_id: str, operation: str) -> MockWorker:
        worker = self.workers.get((farm_id, fleet_id, worker_id))
        if worker is None:
            raise _client_error(
                operation,
                "ResourceNotFoundException",
                f"Resource of type worker with id {worker_id} does not exist.",
                resourceId=worker_id,
                resourceType="worker",
            )
        return worker

    def _sessions_for(self, worker_key: tuple[str, str, str]) -> list[MockWorkerSession]:
        return [
            session
            for session in self.sessions.values()
            if session.worker_key == worker_key and not session.ended
        ]

    def _queue_exits(self, session: MockWorkerSession, environment_ids: list[str]) -> list[str]:
        missing = [
            environment_id
            for environment_id in reversed(environment_ids)
            if environment_id not in session.queued_exit_ids()
        ]
        return self.add_actions(
            session.session_id,
            [("envExit", {"environmentId": environment_id}) for environment_id in missing],
        )

    def _apply_updates(self, worker: MockWorker, updates: dict[str, Mapping[str, Any]]) -> None:
        received_at = _now()
        by_session: dict[str, list[tuple[int, MockSessionAction, Mapping[str, Any]]]] = {}
        for action_id, update in updates.items():
            session_id = self._action_index.get(action_id)
            session = self.sessions.get(session_id) if session_id else None
            if session is None or session.worker_key != worker.key:
                self._violate(
                    "unknown-session-action",
                    f"worker reported {action_id}, which was never assigned to it",
                )
                continue
            index = session.index_of(action_id)
            by_session.setdefault(session.session_id, []).append(
                (index, session.actions[index], update)
            )
        for session_id, entries in by_session.items():
            session = self.sessions[session_id]
            # Apply in session order, so a report that ends action N and a report for
            # action N+1 in the same request are judged in the order they happened.
            for index, action, update in sorted(entries, key=lambda entry: entry[0]):
                self._apply_update(session, index, action, update, received_at)

    def _apply_update(
        self,
        session: MockWorkerSession,
        index: int,
        action: MockSessionAction,
        update: Mapping[str, Any],
        received_at: datetime,
    ) -> None:
        action.reports.append(dict(update))
        completed = update.get("completedStatus")
        started_at = _as_datetime(update.get("startedAt"))
        ended_at = _as_datetime(update.get("endedAt"))

        if session.ended:
            self._violate(
                "report-after-session-ended",
                "worker reported an action of a session the service already ended",
                session,
                action,
            )
        if action.is_terminal:
            if completed != action.status:
                self._violate(
                    "terminal-status-changed",
                    f"action was {action.status}, then reported as {completed or 'RUNNING'}",
                    session,
                    action,
                )
            else:
                self._violate(
                    "completion-reported-twice",
                    f"{completed} was reported in more than one successful request",
                    session,
                    action,
                )
            return
        if completed is not None and completed not in TERMINAL_STATUSES:
            self._violate(
                "unknown-completed-status", f"completedStatus={completed}", session, action
            )
            return

        if completed == "NEVER_ATTEMPTED":
            if started_at is not None or ended_at is not None:
                self._violate(
                    "never-attempted-has-times",
                    "NEVER_ATTEMPTED must not report startedAt or endedAt",
                    session,
                    action,
                )
            for earlier in session.actions[:index]:
                if earlier.status == "RUNNING":
                    self._violate(
                        "never-attempted-before-earlier-action-ended",
                        f"reported NEVER_ATTEMPTED while earlier {earlier.kind} "
                        f"{earlier.session_action_id} is still running",
                        session,
                        action,
                    )
            action.never_attempted_received_at = received_at
        else:
            for earlier in session.actions[:index]:
                if not earlier.is_terminal:
                    self._violate(
                        "ran-out-of-order",
                        f"{action.kind} ran before earlier {earlier.kind} "
                        f"{earlier.session_action_id} ended (it is {earlier.status})",
                        session,
                        action,
                    )
            if (
                completed in (None, "SUCCEEDED")
                and started_at is None
                and action.started_at is None
            ):
                self._violate(
                    "missing-started-at",
                    f"a {completed or 'RUNNING'} report has no startedAt",
                    session,
                    action,
                )
            if started_at and ended_at and ended_at < started_at:
                self._violate(
                    "ended-before-started", "endedAt is before startedAt", session, action
                )
            if ended_at is not None:
                # An action that was still running when a later action was reported as
                # NEVER_ATTEMPTED breaks the cancel rule, even if the worker never sent a
                # RUNNING report for it.
                for later in session.actions[index + 1 :]:
                    if (
                        later.never_attempted_received_at is not None
                        and later.canceled_by_service
                        and ended_at > later.never_attempted_received_at
                    ):
                        self._violate(
                            "never-attempted-before-earlier-action-ended",
                            f"NEVER_ATTEMPTED was reported before earlier {action.kind} "
                            f"{action.session_action_id} ended",
                            session,
                            later,
                        )
            self._check_allowed_after_failure(session, index, action, completed)
            if action.status == "ASSIGNED":
                self._check_entities(session, action, completed, started_at)
                if (
                    session.queue_role_error in _TERMINAL_QUEUE_ROLE_ERRORS
                    and completed in (None, "SUCCEEDED")
                    and action.kind != "envExit"
                ):
                    self._violate(
                        "ran-after-queue-role-error",
                        f"{action.kind} ran though AssumeQueueRoleForWorker returned "
                        f"{session.queue_role_error}",
                        session,
                        action,
                    )

        if completed is None:
            action.status = "RUNNING"
        else:
            action.status = completed
        action.started_at = started_at or action.started_at
        action.ended_at = ended_at or action.ended_at
        if "processExitCode" in update:
            action.process_exit_code = update["processExitCode"]
        if "progressMessage" in update:
            action.progress_message = update["progressMessage"]

        if completed in _UNSUCCESSFUL and action.kind != "envExit" and session.blocked_at is None:
            session.blocked_at = index
            # The service may queue envExit actions for every environment that ran.
            environment_ids = session.entered_environment_ids()
            if action.kind == "envEnter" and action.environment_id not in environment_ids:
                environment_ids.append(str(action.environment_id))
            self._queue_exits(session, environment_ids)

    def _check_allowed_after_failure(
        self,
        session: MockWorkerSession,
        index: int,
        action: MockSessionAction,
        completed: str | None,
    ) -> None:
        """An attempted action after an unsuccessful one must be an allowed envExit."""
        if session.blocked_at is None or index < session.blocked_at:
            return
        if action.kind != "envExit":
            blocker = session.actions[session.blocked_at]
            ran = completed in (None, "SUCCEEDED") or "processExitCode" in action.reports[-1]
            if not ran:
                rule = "rejected-instead-of-never-attempted"
            elif blocker.process_exit_code is None:
                # The blocking action failed without running a process, as when the
                # worker cannot set up the session.
                rule = "ran-after-setup-failure"
            else:
                rule = "ran-after-unsuccessful-action"
            self._violate(
                rule,
                f"{action.kind} was {completed or 'RUNNING'} after action "
                f"{session.blocked_at} ended {blocker.status}",
                session,
                action,
            )
            return
        if action.environment_id not in session.entered_environment_ids():
            self._violate(
                "exit-without-enter",
                f"envExit ran for {action.environment_id}, which was never entered",
                session,
                action,
            )

    def _schedule_for(self, worker: MockWorker) -> dict[str, Any]:
        assigned: dict[str, Any] = {}
        cancels: dict[str, list[str]] = {}
        for session in self._sessions_for(worker.key):
            self._queue_closing_exits(session)
            if session.closing and session.all_terminal:
                session.ended = True
                continue
            pending = [action for action in session.actions if not action.is_terminal]
            for action in pending:
                if action.first_sent_in_call is None:
                    action.first_sent_in_call = worker.schedule_calls
            assignment: dict[str, Any] = {
                "queueId": session.queue_id,
                "jobId": session.job_id,
                # Canceled actions stay in the list until the worker reports them. The
                # worker replaces its queue with this list, so dropping a canceled action
                # here would drop it from the worker before it is reported.
                "sessionActions": [action.to_assignment() for action in pending],
                "logConfiguration": dict(session.log_configuration),
            }
            if session.metadata is not None:
                assignment["metadata"] = dict(session.metadata)
            assigned[session.session_id] = assignment
            canceled = [
                action.session_action_id for action in pending if action.canceled_by_service
            ]
            if canceled:
                cancels[session.session_id] = canceled
        response: dict[str, Any] = {
            "assignedSessions": assigned,
            "cancelSessionActions": cancels,
            "updateIntervalSeconds": self.update_interval_seconds,
        }
        if worker.drain_requested and not assigned:
            response["desiredWorkerStatus"] = "STOPPED"
            worker.stop_sent = True
        return response


__all__ = [
    "TERMINAL_STATUSES",
    "MockSessionAction",
    "MockWorker",
    "MockWorkerScheduler",
    "MockWorkerSession",
    "ProtocolViolation",
]
