"""tasks.json scheduling: dependency order, blocked skipping, roll-up, resume."""

from taskstore import BLOCKED, DONE, IN_PROGRESS, PENDING, TaskStore


def _data():
    return {
        "project": "p",
        "tasks": [
            {
                "id": "T1",
                "title": "t1",
                "subtasks": [
                    {"id": "T1.1", "title": "a", "depends_on": [], "status": "pending"},
                    {"id": "T1.2", "title": "b", "depends_on": ["T1.1"], "status": "pending"},
                ],
            },
            {
                "id": "T2",
                "title": "t2",
                "subtasks": [
                    {"id": "T2.1", "title": "c", "depends_on": ["T1.2"], "status": "pending"},
                ],
            },
        ],
    }


def test_next_runnable_respects_dependencies(workspace):
    s = TaskStore.from_data(workspace, _data())
    assert s.next_runnable()[1]["id"] == "T1.1"
    s.set_status("T1.1", DONE)
    assert s.next_runnable()[1]["id"] == "T1.2"


def test_blocked_makes_dependents_unsatisfiable(workspace):
    s = TaskStore.from_data(workspace, _data())
    s.set_status("T1.1", BLOCKED)
    # T1.2 depends on the blocked T1.1, and T2.1 transitively -> nothing runnable.
    assert s.next_runnable() is None


def test_status_rollup_and_counts(workspace):
    s = TaskStore.from_data(workspace, _data())
    s.set_status("T1.1", DONE)
    s.set_status("T1.2", DONE)
    assert s.tasks[0]["status"] == DONE  # all subtasks done -> task done
    c = s.counts()
    assert c[DONE] == 2 and c[PENDING] == 1


def test_reset_in_progress_for_resume(workspace):
    s = TaskStore.from_data(workspace, _data())
    s.set_status("T1.1", IN_PROGRESS)
    assert s.reset_in_progress() == 1
    assert s.get_subtask("T1.1")["status"] == PENDING


def test_persistence_roundtrip(workspace):
    TaskStore.from_data(workspace, _data())
    assert TaskStore.load(workspace).total_subtasks() == 3
