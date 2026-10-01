"""Regression: ``notify_on_complete`` completions are delivered-or-retried, never silently lost.

A background terminal session (notify_on_complete) whose exit lands while no consumer is
live to drain the in-memory queue — the owning CLI process exited between turns — must
have its completion notification replayed at the next drain of the owning session from
the durable receipt, exactly once. Receipts that were explicitly "never replayed as
notifications" were the defect: the event vanished with zero log lines while the result
sat in the receipt. Un-owned and stale events stay pending for their owner
(delivered-or-retried); a consumed/suppressed completion clears the retry marker so the
replay is never a duplicate turn.
"""

import json
import time

import pytest

from tools import process_registry_results as results_mod
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _bare_registry():
    """A registry with fresh in-memory state, as a new process would have after the
    one that lost the event exited."""
    reg = ProcessRegistry.__new__(ProcessRegistry)
    import queue as queue_mod
    import threading

    reg._lock = threading.Lock()
    reg._running = {}
    reg._finished = {}
    reg.completion_queue = queue_mod.Queue()
    reg._completion_consumed = set()
    reg._poll_observed = set()
    reg._completions_restored = False
    return reg


def _session(**overrides):
    fields = dict(
        id="proc_deadbeef01",
        command="sleep 3600 && echo WORKFLOW_FAILED",
        task_id="task-owner",
        owner_task_id="task-owner",
        session_key="sess-owner",
        started_at=time.time() - 3600.0,
        notify_on_complete=True,
        output_buffer="WORKFLOW_FAILED\n",
    )
    fields.update(overrides)
    return ProcessSession(**fields)


def _receipt_path(hermes_home, session_id):
    return hermes_home / "logs" / "process-results" / f"{session_id}.json"


def _load_receipt(hermes_home, session_id):
    return json.loads(_receipt_path(hermes_home, session_id).read_text(encoding="utf-8-sig"))


def _exit_in_dying_process(session):
    """The incident: the exit is observed, the receipt + in-memory queue write happen,
    and the owning process exits without any consumer draining the queue."""
    reg = _bare_registry()
    session.mark_exited(1)
    reg._running[session.id] = session
    assert reg._move_to_finished(session) is True
    assert not reg.completion_queue.empty()  # queued in memory only — dies with the process


def test_completion_survives_owner_death_and_replays_once(home):
    session = _session()
    _exit_in_dying_process(session)
    receipt = _load_receipt(home, session.id)
    assert receipt["notify_on_complete"] is True

    # A NEW process for the owning session (the next turn after the gap): the drain
    # reconciles the durable receipt into a notification instead of losing it.
    reg = _bare_registry()
    owned = {"sess-owner"}
    drained = reg.drain_notifications(
        session_key="sess-owner", owns_event=lambda evt: evt.get("session_key") in owned)
    assert [evt["session_id"] for evt, _text in drained] == [session.id]
    text = drained[0][1]
    assert "Background process" in text and "WORKFLOW_FAILED" in text
    assert drained[0][0]["restored"] is True

    # Delivered exactly once: the retry marker is cleared and no later drain replays it.
    assert "notification_pending" not in _load_receipt(home, session.id)
    reg2 = _bare_registry()
    assert reg2.drain_notifications(
        session_key="sess-owner", owns_event=lambda evt: evt.get("session_key") in owned) == []


def test_foreign_drain_does_not_steal_and_event_stays_pending(home):
    session = _session()
    _exit_in_dying_process(session)

    # Another session's drain in the new process must not consume the owner's event...
    thief = _bare_registry()
    assert thief.drain_notifications(
        session_key="sess-other",
        owns_event=lambda evt: evt.get("session_key") == "sess-other") == []
    # ...and the receipt remains pending so a later touch by the real owner retries.
    assert _load_receipt(home, session.id)["notification_pending"]["session_id"] == session.id

    # An unfiltered legacy drain cannot prove ownership of a restored completion
    # (fail-closed, same contract as restored delegation events).
    legacy = _bare_registry()
    assert [evt for evt, _text in legacy.drain_notifications()] == []


def test_consumed_completion_is_not_replayed_after_process_death(home):
    """The agent already read the result via wait()/read_log() before its process died:
    the next process must not spend a turn re-announcing output in hand."""
    session = _session()
    reg = _bare_registry()
    session.mark_exited(0)
    reg._running[session.id] = session
    reg._move_to_finished(session)
    reg._completion_consumed.add(session.id)  # wait()/read_log() path
    reg.drain_notifications(
        session_key="sess-owner", owns_event=lambda evt: evt.get("session_key") == "sess-owner")

    fresh = _bare_registry()
    assert fresh.drain_notifications(
        session_key="sess-owner", owns_event=lambda evt: evt.get("session_key") == "sess-owner") == []
    assert "notification_pending" not in _load_receipt(home, session.id)


def test_stale_pending_completion_drops_its_retry_marker(home):
    session = _session(started_at=time.time() - 30 * 86400.0)
    _exit_in_dying_process(session)
    # Age the completion past the replay window (results keep receipts for 7 days; a
    # nobody-waits-on-it-old replay would be a phantom turn).
    pending = _load_receipt(home, session.id)["notification_pending"]
    pending["completed_at"] = time.time() - 30 * 86400.0
    _receipt_path(home, session.id).write_text(
        json.dumps({**_load_receipt(home, session.id), "notification_pending": pending}),
        encoding="utf-8")

    reg = _bare_registry()
    assert reg.drain_notifications(
        session_key="sess-owner", owns_event=lambda evt: evt.get("session_key") == "sess-owner") == []
    assert "notification_pending" not in _load_receipt(home, session.id)


def test_no_notification_persisted_without_notify_contract(home):
    """Sessions without notify_on_complete keep receipts exactly as before: a plain
    server/daemon must never produce a replayed wake."""
    session = _session(notify_on_complete=False)
    reg = _bare_registry()
    session.mark_exited(0)
    reg._running[session.id] = session
    reg._move_to_finished(session)
    assert "notification_pending" not in _load_receipt(home, session.id)
    fresh = _bare_registry()
    assert fresh.drain_notifications() == []


def test_receipt_rewrite_preserves_pending_notification(home):
    """The kill path re-writes a receipt the reader thread already wrote (same exit code
    race the module already handles); the retry marker must survive the rewrite."""
    session = _session()
    results_mod.save_completed_result(
        session, notification={"type": "completion", "session_id": session.id,
                               "session_key": "sess-owner"})
    session.mark_exited(-15, reason="killed", source="process.kill")
    results_mod.save_completed_result(session)  # no notification passed — must not lose the marker
    assert _load_receipt(home, session.id)["notification_pending"]["session_id"] == session.id
