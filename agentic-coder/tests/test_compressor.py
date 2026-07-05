"""HandoffBuilder budget trimming: the redesign replaced the old standalone
compressor with priority trimming inside the Manager handoff assembly
(context/handoff.py) — the subtask block is never truncated, distant file
summaries and the file tree are dropped/trimmed first when over budget."""

from config import load_config
from context.handoff import HandoffBuilder
from context.manifest import Manifest
from context.summaries import SummaryIndex
from taskstore import TaskStore


def _cfg(max_handoff_tokens: int, always_include=None):
    c = load_config()
    c.context.max_handoff_tokens = max_handoff_tokens
    c.context.always_include = list(always_include) if always_include is not None else []
    return c


def _builder(workspace, cfg):
    summaries = SummaryIndex(workspace)
    manifest = Manifest(workspace)
    return HandoffBuilder(workspace, summaries, manifest, cfg), summaries, manifest


def _store(workspace, subtask, dep_files=None):
    subtasks = []
    if dep_files is not None:
        subtasks.append(
            {"id": "T1.0", "title": "dep", "type": "implement", "files": dep_files, "dependencies": [], "test_command": "pytest"}
        )
    subtasks.append(subtask)
    data = {"project": "p", "tasks": [{"id": "T1", "title": "t1", "subtasks": subtasks}]}
    return TaskStore.from_data(workspace, data)


def test_no_trim_when_under_budget(workspace):
    cfg = _cfg(100_000)
    builder, _, _ = _builder(workspace, cfg)
    subtask = {
        "id": "T1.1", "title": "s", "type": "implement", "intent": "do x",
        "files": ["a.py"], "dependencies": [], "test_command": "pytest",
    }
    store = _store(workspace, subtask)
    handoff = builder.build(subtask, store)
    assert "T1.1" in handoff.user_context
    assert handoff.trimmed == []


def test_trims_distant_file_summaries_over_budget(workspace):
    cfg = _cfg(512, always_include=[])
    builder, summaries, _ = _builder(workspace, cfg)
    summaries.write("dep_a.py", "summary of dependency a " * 150)
    summaries.write("dep_b.py", "summary of dependency b " * 150)
    subtask = {
        "id": "T1.1", "title": "s", "type": "implement", "intent": "do x",
        "files": [], "dependencies": ["T1.0"], "test_command": "pytest",
    }
    store = _store(workspace, subtask, dep_files=["dep_a.py", "dep_b.py"])
    handoff = builder.build(subtask, store)
    assert "T1.1" in handoff.user_context  # the current subtask is never trimmed
    assert handoff.trimmed  # at least one distant summary/tree entry was dropped to fit


def test_tree_dropped_when_no_room(workspace):
    cfg = _cfg(512, always_include=[])
    builder, _, manifest = _builder(workspace, cfg)
    manifest.record("src/app.py", "entrypoint")
    subtask = {
        "id": "T1.1", "title": "s", "type": "implement",
        "intent": "x " * 800,  # alone exceeds the 512-token budget
        "files": [], "dependencies": [], "test_command": "pytest",
    }
    store = _store(workspace, subtask)
    handoff = builder.build(subtask, store)
    assert "file_tree:dropped" in handoff.trimmed
    assert "Workspace file tree" not in handoff.user_context
