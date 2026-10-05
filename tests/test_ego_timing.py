"""Timing regression tests for the complete semantic physical tour."""

import ast
import math
from pathlib import Path

import pytest

from tests.test_ego_planner import semantic_fixture
from tests.test_ego_planner import tour_plan as tour_fixture
from tests.test_house_tour import IFC, Entity, boundary, rooms
from tests.test_house_tour import planner as graph_fixture


@pytest.fixture
def tour_plan():
    return tour_fixture.__wrapped__()


@pytest.fixture
def planner():
    return graph_fixture.__wrapped__()


@pytest.fixture
def sample():
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_tour")
    namespace = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["sample_tour"]


def test_speed_holds_vertices_and_full_route(sample):
    path = [(0, 0, 0), (3, 0, 0), (3, 4, 0), (3, 0, 0)]
    samples, frames, timing = sample(path, [0, 2, 3], 1, 10)
    assert [samples[frames[i]] for i in range(len(path))] == path
    assert samples[-1] == path[-1]
    assert timing["movement_seconds"] == 11
    assert timing["turn_seconds"] == 4.5
    assert timing["observation_seconds"] == pytest.approx(2.4)
    for hold in timing["holds"]:
        assert len(set(samples[hold["start_frame"]:hold["end_frame"] + 1])) == 1
    assert len([h for h in timing["holds"] if h["kind"] == "observation"]) == 3
    assert len([h for h in timing["holds"] if h["kind"] == "turn"]) == 2
    longer, _, slower = sample(path, [0, 2, 3], len(samples) + 100, 10)
    assert len(longer) == len(samples) + 100
    assert slower["movement_seconds"] == timing["movement_seconds"]


def test_low_fps_not_limited_by_collision_probe_step(sample):
    samples, _, timing = sample([(0, 0, 0), (3, 0, 0)], [0, 1], 1, 2)
    assert timing["movement_seconds"] == 3
    assert max(math.dist(a, b) for a, b in zip(samples, samples[1:], strict=False)) == .5


def test_frame_limit_fails_without_truncation(sample):
    with pytest.raises(ValueError, match="refusing truncation"):
        sample([(0, 0, 0), (300000, 0, 0)], [0, 1], 1, 1)


def test_seeded_graph_and_physical_frames_repeat(planner, tour_plan):
    a, b, c = rooms(3)
    model = IFC(a, b, c)
    for eid, pair in ((101, (a, b)), (102, (b, c))):
        opening = Entity(eid, "IfcOpeningElement")
        model.entities.extend([opening] + [boundary(eid * 10 + i, room, opening) for i, room in enumerate(pair)])
    meshes, physical = semantic_fixture()
    def run():
        semantic = planner["_house_tour"](model, seed=42)
        for edge in semantic["edges"]:
            edge["portal_triangles"] = next(e["portal_triangles"] for e in physical["edges"] if e["connector_id"] == edge["connector_id"])
        return tour_plan(meshes, semantic, 1.65, 2, 10)
    first = run()
    model.entities.reverse()
    assert first == run()


def test_singleton_camera_must_be_inside_selected_space(tour_plan):
    meshes, semantic = semantic_fixture()
    semantic["component_tours"] = [dict(component_id=0, space_ids=[1], route_space_ids=[1], coverage=1, tour=[dict(space_id=1)])]
    with pytest.raises(ValueError, match="No interior observation point"):
        tour_plan(meshes, semantic, 4.0, 10, 10)


def test_singleton_has_observation_pause(tour_plan):
    meshes, semantic = semantic_fixture()
    semantic["component_tours"] = [dict(component_id=0, space_ids=[1], route_space_ids=[1], coverage=1, tour=[dict(space_id=1)])]
    positions, metadata = tour_plan(meshes, semantic, 1.65, 1, 10)
    assert len(positions) >= 9
    assert len(set(positions)) == 1
    assert set(metadata["frame_space_ids"]) == {1}
    assert metadata["timing"]["observation_seconds"] == .8


def test_yaw_changes_only_while_stationary(sample):
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ego_yaws")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    samples, _, _ = sample([(0, 0, 0), (3, 0, 0), (3, 4, 0), (3, 0, 0)], [0, 2, 3], 1, 10)
    yaws = namespace["_ego_yaws"](samples, 10)
    for i in range(1, len(samples)):
        delta = abs((yaws[i] - yaws[i - 1] + math.pi) % (2 * math.pi) - math.pi)
        assert delta <= math.radians(6) + 1e-8
        if math.dist(samples[i], samples[i - 1]) > 1e-8:
            assert delta < 1e-8
