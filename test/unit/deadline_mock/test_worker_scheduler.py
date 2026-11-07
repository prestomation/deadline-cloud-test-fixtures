# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from deadline_test_fixtures.deadline_mock import (
    MockDeadlineBackend,
    MockDeadlineScenario,
    start_server,
)

FLEET_ID = "fleet-0000000000000000000000000000000f"
JOB_ID = "job-00000000000000000000000000000001"
STEP_ID = "step-00000000000000000000000000000001"
ENV_ID = "JOB:job-00000000000000000000000000000001:Env"


@pytest.fixture
def backend() -> MockDeadlineBackend:
    return MockDeadlineBackend(MockDeadlineScenario.without_job_attachments())


@pytest.fixture
def client(backend: MockDeadlineBackend) -> Generator[Any, None, None]:
    server, endpoint, thread = start_server(backend)
    try:
        yield boto3.client(
            "deadline",
            region_name="us-west-2",
            endpoint_url=endpoint,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            # The worker APIs have a "scheduling." host prefix; a loopback endpoint has none.
            config=Config(inject_host_prefix=False, retries={"max_attempts": 1}),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _start_worker(client: Any, backend: MockDeadlineBackend) -> str:
    worker_id = client.create_worker(farmId=backend.farm_id, fleetId=FLEET_ID)["workerId"]
    client.update_worker(
        farmId=backend.farm_id, fleetId=FLEET_ID, workerId=worker_id, status="STARTED"
    )
    return worker_id


def _schedule(
    client: Any, backend: MockDeadlineBackend, worker_id: str, updates: dict | None = None
) -> dict[str, Any]:
    return client.update_worker_schedule(
        farmId=backend.farm_id,
        fleetId=FLEET_ID,
        workerId=worker_id,
        updatedSessionActions=updates or {},
    )


def _times(seconds: float = 1.0) -> tuple[datetime, datetime]:
    start = datetime.now(timezone.utc) - timedelta(seconds=10)
    return start, start + timedelta(seconds=seconds)


def _open(backend: MockDeadlineBackend, worker_id: str, *actions: tuple[str, dict]) -> str:
    return backend.worker_scheduler.open_session(
        worker_id=worker_id,
        queue_id=backend.queue_id,
        job_id=JOB_ID,
        actions=actions,  # type: ignore[arg-type]
    )


def _ids(response: dict[str, Any], session_id: str) -> list[str]:
    return [
        action["sessionActionId"]
        for action in response["assignedSessions"][session_id]["sessionActions"]
    ]


ENTER = ("envEnter", {"environmentId": ENV_ID})
TASK = ("taskRun", {"stepId": STEP_ID, "parameters": {}})


def test_new_worker_gets_an_empty_schedule(client: Any, backend: MockDeadlineBackend) -> None:
    worker_id = _start_worker(client, backend)

    response = _schedule(client, backend, worker_id)

    assert response["assignedSessions"] == {}
    assert response["cancelSessionActions"] == {}
    assert response["updateIntervalSeconds"] == 1
    assert "desiredWorkerStatus" not in response


def test_schedule_for_a_created_worker_is_a_status_conflict(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = client.create_worker(farmId=backend.farm_id, fleetId=FLEET_ID)["workerId"]

    with pytest.raises(ClientError) as raised:
        _schedule(client, backend, worker_id)

    assert raised.value.response["Error"]["Code"] == "ConflictException"
    assert raised.value.response["reason"] == "STATUS_CONFLICT"
    assert raised.value.response["resourceId"] == worker_id


def test_happy_path_session_ends_and_drain_stops_the_worker(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, ENTER, TASK)

    response = _schedule(client, backend, worker_id)
    enter_id, task_id = _ids(response, session_id)
    assert response["assignedSessions"][session_id]["sessionActions"][0]["definition"] == {
        "envEnter": {"environmentId": ENV_ID}
    }
    assert response["assignedSessions"][session_id]["logConfiguration"]["logDriver"] == "awslogs"

    start, end = _times()
    _schedule(
        client,
        backend,
        worker_id,
        {
            enter_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end},
            task_id: {"startedAt": end},
        },
    )
    assert scheduler.action(task_id).status == "RUNNING"

    (exit_id,) = scheduler.close_session(session_id)
    response = _schedule(
        client,
        backend,
        worker_id,
        {task_id: {"completedStatus": "SUCCEEDED", "endedAt": end + timedelta(seconds=1)}},
    )
    assert _ids(response, session_id) == [exit_id]

    _schedule(
        client,
        backend,
        worker_id,
        {exit_id: {"completedStatus": "SUCCEEDED", "startedAt": end, "endedAt": end}},
    )
    scheduler.drain(worker_id, immediate=False)
    response = _schedule(client, backend, worker_id)

    assert response["assignedSessions"] == {}
    assert response["desiredWorkerStatus"] == "STOPPED"
    scheduler.check_session_complete(session_id)
    assert scheduler.violations == []


def test_never_attempted_with_times_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK)
    (task_id,) = _ids(_schedule(client, backend, worker_id), session_id)
    start, end = _times()

    _schedule(
        client,
        backend,
        worker_id,
        {task_id: {"completedStatus": "NEVER_ATTEMPTED", "startedAt": start, "endedAt": end}},
    )

    assert [v.rule for v in backend.worker_scheduler.violations] == ["never-attempted-has-times"]


def test_running_a_later_action_first_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, ENTER, TASK)
    _, task_id = _ids(_schedule(client, backend, worker_id), session_id)
    start, _ = _times()

    _schedule(client, backend, worker_id, {task_id: {"startedAt": start}})

    assert [v.rule for v in backend.worker_scheduler.violations] == ["ran-out-of-order"]


def test_failure_queues_env_exit_and_blocks_new_tasks(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, ENTER, TASK, TASK)
    enter_id, task_1, task_2 = _ids(_schedule(client, backend, worker_id), session_id)
    start, end = _times()

    response = _schedule(
        client,
        backend,
        worker_id,
        {
            enter_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": start},
            task_1: {
                "completedStatus": "FAILED",
                "startedAt": start,
                "endedAt": end,
                "processExitCode": 1,
            },
            task_2: {"completedStatus": "NEVER_ATTEMPTED"},
        },
    )

    (exit_id,) = _ids(response, session_id)
    assert scheduler.action(exit_id).kind == "envExit"
    with pytest.raises(ValueError):
        scheduler.add_actions(session_id, [TASK])  # type: ignore[list-item]
    assert scheduler.violations == []


def test_task_after_failure_is_a_violation(client: Any, backend: MockDeadlineBackend) -> None:
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK, TASK)
    task_1, task_2 = _ids(_schedule(client, backend, worker_id), session_id)
    start, end = _times()

    _schedule(
        client,
        backend,
        worker_id,
        {
            task_1: {
                "completedStatus": "FAILED",
                "startedAt": start,
                "endedAt": start,
                "processExitCode": 1,
            },
            task_2: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end},
        },
    )

    assert [v.rule for v in backend.worker_scheduler.violations] == [
        "ran-after-unsuccessful-action"
    ]


def test_cancel_lists_the_action_and_later_tasks_but_not_env_exit(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, ENTER, TASK, TASK)
    enter_id, task_1, task_2 = _ids(_schedule(client, backend, worker_id), session_id)
    # The envEnter is not reported yet, so closing queues no envExit.
    assert scheduler.close_session(session_id) == []

    canceled = scheduler.cancel_action(task_1)
    start, end = _times()
    response = _schedule(
        client,
        backend,
        worker_id,
        {enter_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end}},
    )

    assert canceled == [task_1, task_2]
    assert response["cancelSessionActions"] == {session_id: [task_1, task_2]}
    (exit_id,) = _ids(response, session_id)[2:]
    assert _ids(response, session_id) == [task_1, task_2, exit_id]
    assert scheduler.action(exit_id).kind == "envExit"


def test_never_attempted_while_earlier_action_runs_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK, TASK)
    task_1, task_2 = _ids(_schedule(client, backend, worker_id), session_id)
    start, _ = _times()
    _schedule(client, backend, worker_id, {task_1: {"startedAt": start}})
    scheduler.cancel_action(task_1)

    _schedule(client, backend, worker_id, {task_2: {"completedStatus": "NEVER_ATTEMPTED"}})

    assert [v.rule for v in scheduler.violations] == ["never-attempted-before-earlier-action-ended"]


def test_unknown_action_and_changed_status_are_violations(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK)
    (task_id,) = _ids(_schedule(client, backend, worker_id), session_id)
    start, end = _times()
    done = {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end}
    _schedule(client, backend, worker_id, {task_id: done})

    _schedule(
        client,
        backend,
        worker_id,
        {task_id: {**done, "completedStatus": "FAILED"}, "sessionaction-nope-0": done},
    )

    assert sorted(v.rule for v in backend.worker_scheduler.violations) == [
        "terminal-status-changed",
        "unknown-session-action",
    ]


def test_stopping_with_open_actions_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    _open(backend, worker_id, TASK)
    ids = {"farmId": backend.farm_id, "fleetId": FLEET_ID, "workerId": worker_id}
    _schedule(client, backend, worker_id)
    # An action first sent in the last response is not a violation: the worker never
    # had a chance to report it.
    client.update_worker(**ids, status="STOPPING")
    client.update_worker(**ids, status="STOPPED")
    assert backend.worker_scheduler.violations == []

    client.update_worker(**ids, status="STARTED")
    _open(backend, worker_id, TASK)
    _schedule(client, backend, worker_id)
    _schedule(client, backend, worker_id)
    client.update_worker(**ids, status="STOPPED")

    assert [v.rule for v in backend.worker_scheduler.violations] == ["stopped-with-open-actions"]


def test_entered_environment_without_exit_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, ENTER)
    (enter_id,) = _ids(_schedule(client, backend, worker_id), session_id)
    start, end = _times()
    _schedule(
        client,
        backend,
        worker_id,
        {enter_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end}},
    )

    scheduler.check_session_complete(session_id)

    assert [v.rule for v in scheduler.violations] == ["entered-environment-not-exited"]


def test_batch_get_job_entity_returns_entities_and_errors(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    scheduler.add_job(
        job_id=JOB_ID,
        job_details={"logGroupName": "/aws/deadline/mock", "schemaVersion": "jobtemplate-2023-09"},
        steps={
            STEP_ID: {
                "schemaVersion": "jobtemplate-2023-09",
                "template": {"name": "Step", "script": {"actions": {"onRun": {"command": "x"}}}},
            }
        },
    )
    worker_id = _start_worker(client, backend)
    _open(backend, worker_id, TASK)

    response = client.batch_get_job_entity(
        farmId=backend.farm_id,
        fleetId=FLEET_ID,
        workerId=worker_id,
        identifiers=[
            {"jobDetails": {"jobId": JOB_ID}},
            {"stepDetails": {"jobId": JOB_ID, "stepId": STEP_ID}},
            {"environmentDetails": {"jobId": JOB_ID, "environmentId": ENV_ID}},
        ],
    )

    assert response["entities"][0]["jobDetails"]["jobId"] == JOB_ID
    assert response["entities"][1]["stepDetails"]["dependencies"] == []
    # Templates are documents and pass through the response filter unchanged.
    assert response["entities"][1]["stepDetails"]["template"]["name"] == "Step"
    assert response["errors"] == [
        {
            "environmentDetails": {
                "jobId": JOB_ID,
                "environmentId": ENV_ID,
                "code": "ResourceNotFoundException",
                "message": f"No environmentDetails for {ENV_ID}",
            }
        }
    ]


def test_queue_role_credentials_need_a_session_on_the_queue(
    client: Any, backend: MockDeadlineBackend
) -> None:
    backend.worker_scheduler.queue_role_arns[backend.queue_id] = "arn:aws:iam::1:role/Queue"
    worker_id = _start_worker(client, backend)
    request = {
        "farmId": backend.farm_id,
        "fleetId": FLEET_ID,
        "workerId": worker_id,
        "queueId": backend.queue_id,
    }

    with pytest.raises(ClientError) as raised:
        client.assume_queue_role_for_worker(**request)
    _open(backend, worker_id, TASK)
    response = client.assume_queue_role_for_worker(**request)

    assert raised.value.response["Error"]["Code"] == "AccessDeniedException"
    assert response["credentials"]["accessKeyId"]


def test_fleet_role_credentials(client: Any, backend: MockDeadlineBackend) -> None:
    worker_id = _start_worker(client, backend)

    response = client.assume_fleet_role_for_worker(
        farmId=backend.farm_id, fleetId=FLEET_ID, workerId=worker_id
    )

    assert response["credentials"]["expiration"] > datetime.now(timezone.utc)


def test_injected_conflict_has_modeled_members_and_skips_the_route(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    backend.inject_error(
        "UpdateWorkerSchedule",
        "ConflictException",
        reason="CONCURRENT_MODIFICATION",
        resourceId=worker_id,
        resourceType="worker",
    )
    calls_before = backend.worker_scheduler.workers[
        (backend.farm_id, FLEET_ID, worker_id)
    ].schedule_calls

    with pytest.raises(ClientError) as raised:
        _schedule(client, backend, worker_id)
    _schedule(client, backend, worker_id)

    error = raised.value.response
    assert error["ResponseMetadata"]["HTTPStatusCode"] == 409
    assert error["reason"] == "CONCURRENT_MODIFICATION"
    worker = backend.worker_scheduler.workers[(backend.farm_id, FLEET_ID, worker_id)]
    assert worker.schedule_calls == calls_before + 1


def test_delete_worker_requires_an_offline_worker(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    ids = {"farmId": backend.farm_id, "fleetId": FLEET_ID, "workerId": worker_id}

    with pytest.raises(ClientError):
        client.delete_worker(**ids)
    client.update_worker(**ids, status="STOPPED")
    client.delete_worker(**ids)

    assert backend.worker_scheduler.workers == {}


def test_snapshot_includes_worker_protocol_state(client: Any, backend: MockDeadlineBackend) -> None:
    worker_id = _start_worker(client, backend)
    _open(backend, worker_id, TASK)

    snapshot = backend.snapshot()["worker_protocol"]

    assert snapshot["workers"][0]["statusHistory"] == ["CREATED", "STARTED"]
    assert snapshot["sessions"][0]["actions"][0]["kind"] == "taskRun"
    assert snapshot["violations"] == []


def _add_job(backend: MockDeadlineBackend) -> None:
    backend.worker_scheduler.add_job(
        job_id=JOB_ID,
        job_details={"logGroupName": "/aws/deadline/mock", "schemaVersion": "jobtemplate-2023-09"},
        steps={STEP_ID: {"schemaVersion": "jobtemplate-2023-09", "template": {"name": "S"}}},
    )


def _get_step(client: Any, backend: MockDeadlineBackend, worker_id: str) -> dict[str, Any]:
    return client.batch_get_job_entity(
        farmId=backend.farm_id,
        fleetId=FLEET_ID,
        workerId=worker_id,
        identifiers=[{"stepDetails": {"jobId": JOB_ID, "stepId": STEP_ID}}],
    )


def test_injected_entity_error_is_returned_once(client: Any, backend: MockDeadlineBackend) -> None:
    _add_job(backend)
    worker_id = _start_worker(client, backend)
    _open(backend, worker_id, TASK)
    backend.worker_scheduler.inject_entity_error(
        kind="stepDetails", job_id=JOB_ID, entity_id=STEP_ID, code="InternalServerException"
    )

    first = _get_step(client, backend, worker_id)
    second = _get_step(client, backend, worker_id)

    assert first["errors"][0]["stepDetails"]["code"] == "InternalServerException"
    assert second["entities"][0]["stepDetails"]["stepId"] == STEP_ID


def test_task_that_succeeds_after_its_step_fetch_failed_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    _add_job(backend)
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK)
    (task_id,) = _ids(_schedule(client, backend, worker_id), session_id)
    backend.worker_scheduler.inject_entity_error(
        kind="stepDetails", job_id=JOB_ID, entity_id=STEP_ID, code="ValidationException"
    )
    _get_step(client, backend, worker_id)
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1)

    _schedule(
        client,
        backend,
        worker_id,
        {task_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end}},
    )

    assert [v.rule for v in backend.worker_scheduler.violations] == ["ran-despite-entity-error"]


def test_queue_role_credentials_name_their_queue(client: Any, backend: MockDeadlineBackend) -> None:
    backend.worker_scheduler.queue_role_arns[backend.queue_id] = "arn:aws:iam::1:role/Queue"
    worker_id = _start_worker(client, backend)
    _open(backend, worker_id, TASK)

    response = client.assume_queue_role_for_worker(
        farmId=backend.farm_id, fleetId=FLEET_ID, workerId=worker_id, queueId=backend.queue_id
    )

    assert response["credentials"]["accessKeyId"] == f"ASIAQUEUE{backend.queue_id[-8:].upper()}"


def test_running_after_queue_role_not_found_is_a_violation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    scheduler.queue_role_arns[backend.queue_id] = "arn:aws:iam::1:role/Queue"
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK)
    (task_id,) = _ids(_schedule(client, backend, worker_id), session_id)
    scheduler.inject_queue_role_error(backend.queue_id, "ResourceNotFoundException")

    with pytest.raises(ClientError):
        client.assume_queue_role_for_worker(
            farmId=backend.farm_id,
            fleetId=FLEET_ID,
            workerId=worker_id,
            queueId=backend.queue_id,
        )
    start, end = _times()
    _schedule(
        client,
        backend,
        worker_id,
        {task_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end}},
    )

    assert scheduler.queue_role_calls[-1][2] == "ResourceNotFoundException"
    assert [v.rule for v in scheduler.violations] == ["ran-after-queue-role-error"]


def test_not_responding_worker_gets_a_status_conflict(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK)
    backend.worker_scheduler.mark_not_responding(worker_id)

    with pytest.raises(ClientError) as raised:
        _schedule(client, backend, worker_id)

    assert raised.value.response["reason"] == "STATUS_CONFLICT"
    assert backend.worker_scheduler.sessions[session_id].ended


def test_forgotten_worker_gets_resource_not_found(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    backend.worker_scheduler.forget_worker(worker_id)

    with pytest.raises(ClientError) as raised:
        _schedule(client, backend, worker_id)

    assert raised.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_operation_delay_slows_only_that_operation(
    client: Any, backend: MockDeadlineBackend
) -> None:
    worker_id = _start_worker(client, backend)
    backend.operation_delays["UpdateWorkerSchedule"] = 0.5

    started = datetime.now(timezone.utc)
    _schedule(client, backend, worker_id)
    slow = datetime.now(timezone.utc) - started
    started = datetime.now(timezone.utc)
    client.assume_fleet_role_for_worker(
        farmId=backend.farm_id, fleetId=FLEET_ID, workerId=worker_id
    )
    fast = datetime.now(timezone.utc) - started

    assert slow >= timedelta(seconds=0.5) > fast


def test_env_enter_failed_without_a_process_needs_no_exit(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, ENTER, TASK)
    enter_id, task_id = _ids(_schedule(client, backend, worker_id), session_id)
    now = datetime.now(timezone.utc)

    response = _schedule(
        client,
        backend,
        worker_id,
        {
            enter_id: {"completedStatus": "FAILED", "startedAt": now, "endedAt": now},
            task_id: {"completedStatus": "NEVER_ATTEMPTED"},
        },
    )
    (exit_id,) = _ids(response, session_id)
    _schedule(client, backend, worker_id, {exit_id: {"completedStatus": "NEVER_ATTEMPTED"}})
    scheduler.check_session_complete(session_id)

    assert scheduler.violations == []


def test_queue_role_error_is_cleared_by_a_later_success(
    client: Any, backend: MockDeadlineBackend
) -> None:
    scheduler = backend.worker_scheduler
    scheduler.queue_role_arns[backend.queue_id] = "arn:aws:iam::1:role/Queue"
    worker_id = _start_worker(client, backend)
    session_id = _open(backend, worker_id, TASK)
    (task_id,) = _ids(_schedule(client, backend, worker_id), session_id)
    scheduler.inject_queue_role_error(backend.queue_id, "ResourceNotFoundException")
    request = {
        "farmId": backend.farm_id,
        "fleetId": FLEET_ID,
        "workerId": worker_id,
        "queueId": backend.queue_id,
    }
    with pytest.raises(ClientError):
        client.assume_queue_role_for_worker(**request)
    client.assume_queue_role_for_worker(**request)
    start, end = _times()

    _schedule(
        client,
        backend,
        worker_id,
        {task_id: {"completedStatus": "SUCCEEDED", "startedAt": start, "endedAt": end}},
    )

    assert scheduler.violations == []
