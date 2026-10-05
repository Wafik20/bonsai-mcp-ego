"""Progress reporting must observe, never change, tour planning."""
import ast
import json
import os
import time
from pathlib import Path

import pytest

from tests.test_ego_planner import semantic_fixture
from tests.test_ego_planner import tour_plan as tour_fixture


def progress_type():
    source = Path(__file__).resolve().parents[1] / "blender_addon/bonsai_bridge.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "_EgoProgress")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_EgoProgress"]


def test_progress_atomic_snapshots_throttle_and_success(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    path = tmp_path / "tour_progress.json"
    with progress_type()(path) as progress:
        first = path.read_bytes()
        progress("starting", "Throttled update")
        assert path.read_bytes() == first
        progress("routing", "Searching 1 -> 2", search_attempt=1, expanded_nodes=500)
        state = json.loads(path.read_text())
        assert state["status"] == "running"
        assert state["completed"] is None and state["total"] is None
        assert state["details"]["expanded_nodes"] == 500
        clock[0] += .6
        progress("routing", "Searching 1 -> 2", search_attempt=1, expanded_nodes=1000)
        assert json.loads(path.read_text())["details"]["expanded_nodes"] == 1000
        progress("rendering", "Rendered frame 2", 2, 10)
        assert json.loads(path.read_text())["completed"] == 2
    final = json.loads(path.read_text())
    assert final["status"] == "completed"
    assert final["elapsed_seconds"] == .6
    assert final["updated_at"]
    assert list(tmp_path.iterdir()) == [path]


def test_failed_progress_preserves_search_context_and_exception(tmp_path):
    path = tmp_path / "tour_progress.json"
    with pytest.raises(ValueError, match="blocked portal"), progress_type()(path) as progress:
        progress("routing", "Searching", source_space_id=514, target_space_id=355, connector_id=8066)
        raise ValueError("blocked portal")
    state = json.loads(path.read_text())
    assert state["status"] == "failed"
    assert state["stage"] == "routing"
    assert state["message"] == "ValueError: blocked portal"
    assert state["details"]["connector_id"] == 8066


def test_progress_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("original")
    with pytest.raises(FileExistsError), progress_type()(path):
        pytest.fail("Must not enter")
    assert path.read_text() == "original"


def test_progress_io_failure_does_not_abort_work(tmp_path, monkeypatch):
    path = tmp_path / "tour_progress.json"
    replace = os.replace
    with progress_type()(path) as progress:
        def fail(*args):
            raise PermissionError("viewer holds file open")
        monkeypatch.setattr(os, "replace", fail)
        progress("routing", "Searching")
        assert progress.warned
        assert json.loads(path.read_text())["stage"] == "starting"
        assert list(tmp_path.iterdir()) == [path]
        monkeypatch.setattr(os, "replace", replace)
    assert json.loads(path.read_text())["status"] == "completed"


def test_progress_does_not_change_deterministic_path():
    plan = tour_fixture.__wrapped__()
    meshes, semantic = semantic_fixture()
    expected = plan(meshes, semantic, 1.65, 2, 2)
    events = []
    def record(stage, message, completed=None, total=None, **details):
        events.append(dict(stage=stage, message=message, completed=completed, total=total, **details))
    assert plan(meshes, semantic, 1.65, 2, 2, progress=record) == expected
    stages = {event["stage"] for event in events}
    assert {"navigation_geometry", "observations", "routing", "trajectory"} <= stages
    search = [event for event in events if "expanded_nodes" in event]
    assert search
    assert all(event["completed"] is None and event["total"] is None for event in search)
    assert all(event["source_space_id"] != event["target_space_id"] for event in search)
