"""Bounded, profile-local receipts for completed terminal processes.

Receipt output/results are read through process_manage and receipts are never
adopted as live PIDs. A receipt whose process had a ``notify_on_complete`` contract
additionally carries the undelivered completion event under ``notification_pending``
until a consumer resolves it: the completion is delivered-or-retried across process
death, replayed by ``restore_pending_process_completions`` at the next drain.
Each producer writes its own file so independent one-shot parents cannot overwrite
each other's results in the running-PID checkpoint.
"""

import json
import logging
import re
import sqlite3
import time

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger("tools.process_registry")

RESULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_RETAINED_RESULTS = 64
_RESULT_FIELDS = (
    "id", "command", "cwd", "task_id", "owner_task_id", "session_key",
    "parent_session_id", "started_at", "exit_code", "completion_reason",
    "termination_source", "notify_on_complete",
)

# The replay window matches async delegation's cap: an old completion nobody has
# touched gets dropped (the receipt stays queryable), never a phantom turn.
_NOTIFICATION_REPLAY_AGE_S = 48 * 3600.0
_NOTIFICATION_PENDING_KEY = "notification_pending"


def _result_paths():
    """Prune by completion time, not start time (jobs can take days)."""
    directory = get_hermes_home() / "logs" / "process-results"
    cutoff = time.time() - RESULT_RETENTION_SECONDS
    retained = []
    for path in directory.glob("proc_*.json"):
        try:
            modified = path.stat().st_mtime
            if modified < cutoff:
                path.unlink(missing_ok=True)
            else:
                retained.append((modified, path))
        except FileNotFoundError:
            continue  # Another producer pruned it.
    retained.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    for _, path in retained[MAX_RETAINED_RESULTS:]:
        path.unlink(missing_ok=True)
    return [path for _, path in retained[:MAX_RETAINED_RESULTS]]


def save_completed_result(session, notification: dict | None = None) -> None:
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        record = {key: getattr(session, key) for key in _RESULT_FIELDS}
        record["output"] = session.output_buffer[-MAX_OUTPUT_CHARS:]
    # Live-output opt-out must not persist raw credentials in durable receipts.
    record["output"] = redact_terminal_output(record["output"], record["command"], force=True)
    record["command"] = redact_sensitive_text(record["command"], code_file=True, force=True)
    directory = get_hermes_home() / "logs" / "process-results"
    try:
        from hermes_constants import assert_named_profile_home_live
        assert_named_profile_home_live(directory)
        # A retry rewrite (e.g. a kill racing the reader thread's receipt) preserves the
        # pending completion unless the caller resolves it explicitly (below).
        pending = _read_pending_notification(directory / f"{session.id}.json")
        if notification is not None:
            payload = dict(notification)
            payload["command"] = redact_sensitive_text(payload.get("command", ""), code_file=True, force=True)
            payload["output"] = redact_terminal_output(
                payload.get("output", ""), notification.get("command", ""), force=True)
            payload.setdefault("completed_at", time.time())
            pending = payload
        if pending is not None:
            record[_NOTIFICATION_PENDING_KEY] = pending
        elif _NOTIFICATION_PENDING_KEY in record:
            del record[_NOTIFICATION_PENDING_KEY]
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_json_write(directory / f"{session.id}.json", record, mode=0o600)
        _result_paths()
    except OSError:
        # Preserve live delivery on disk failure, but never silently claim durability.
        logger.warning("Could not retain completed process result %s", session.id, exc_info=True)


def _read_pending_notification(path) -> dict | None:
    """The retry marker on one receipt file; None when absent or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    pending = data.get(_NOTIFICATION_PENDING_KEY) if isinstance(data, dict) else None
    return pending if isinstance(pending, dict) else None


def clear_pending_notification(session_id: str) -> bool:
    """Resolve a receipt's retry marker once a consumer delivered, skipped (output already
    in hand via wait/log/poll), or deliberately suppressed the completion. Returns True when
    a marker was removed. The result payload itself is untouched and stays queryable."""
    path = get_hermes_home() / "logs" / "process-results" / f"{session_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or _NOTIFICATION_PENDING_KEY not in data:
        return False
    data.pop(_NOTIFICATION_PENDING_KEY)
    try:
        atomic_json_write(path, data, mode=0o600)
    except OSError:
        logger.warning("Could not resolve pending completion notification %s", session_id, exc_info=True)
        return False
    return True


def restore_pending_process_completions(target_queue) -> int:
    """Rehydrate completion notifications that died with their producer process.

    Scans retained receipts for ``notification_pending`` markers and re-queues each event
    stamped ``restored=True`` (in-memory only — never persisted), the same contract as
    delegation replay: a drain without an ownership filter must not adopt a dead session's
    completion. Markers past the replay window are resolved instead of replaying a turn
    nobody is waiting on; markers a consumer resolves are cleared at that decision, so a
    completion is delivered at most once across process deaths. Callers bind the owning
    profile first (receipts are profile-local)."""
    try:
        paths = _result_paths()
    except OSError:
        logger.warning("Could not read retained process results for replay", exc_info=True)
        return 0
    now = time.time()
    restored = 0
    for path in paths:
        pending = _read_pending_notification(path)
        session_id = str(pending.get("session_id") or "") if pending else ""
        if (pending is None or pending.get("type") != "completion"
                or not session_id or session_id != path.stem
                or not re.fullmatch(r"proc_[\w]+", session_id)):
            continue
        age_basis = pending.get("completed_at") or pending.get("started_at")
        if age_basis and (now - float(age_basis)) > _NOTIFICATION_REPLAY_AGE_S:
            logger.warning("Process %s: pending completion notification is %.1fh old (cap %.1fh); "
                           "resolving the replay (result remains queryable).",
                           session_id, (now - float(age_basis)) / 3600.0, _NOTIFICATION_REPLAY_AGE_S / 3600.0)
            clear_pending_notification(session_id)
            continue
        event = dict(pending)
        event["restored"] = True
        target_queue.put(event)
        restored += 1
    return restored


def _owns_result(owner: str, parent: str | None) -> bool:
    if not parent:
        return False
    if owner == parent:
        return True
    from hermes_state import SessionDB

    # Pure lineage read on the hot path of every retained-result load; a writable open here
    # was one more writer handle per call inside the gateway (#100896).
    db = SessionDB(read_only=True)
    try:
        return db.get_compression_tip(parent) == owner
    finally:
        db.close()


def load_completed_results(prefix: str = "") -> dict:
    """Restore read-only snapshots; no process handles, watchers, or queue events."""
    from tools.process_registry import ProcessSession

    from gateway.session_context import get_session_env

    owner = get_session_env("HERMES_SESSION_ID", "")
    if not owner:
        return {}
    results = {}
    try:
        paths = _result_paths()
    except OSError:
        logger.warning("Could not read retained process results", exc_info=True)
        return results
    for path in paths:
        if not path.stem.startswith(prefix):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            if not _owns_result(owner, record.get("parent_session_id")):
                continue
            session = ProcessSession(
                **{key: record[key] for key in _RESULT_FIELDS},
                exited=True, output_buffer=record["output"],
            )
            session._completion_event.set()
            results[session.id] = session
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return results
