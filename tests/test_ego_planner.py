"""Pure geometry planner tests: load only its AST, without Blender imports."""

import ast
import math
from pathlib import Path

import pytest


@pytest.fixture
def plan():
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_ego_plan"
    ]
    assert nodes, "Bridge must contain the pure-Python _ego_plan helper"
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_ego_plan"]


def box(lo, hi, cls="IfcSpace"):
    x, y, z = lo
    X, Y, Z = hi
    vertices = [
        (x, y, z),
        (X, y, z),
        (X, Y, z),
        (x, Y, z),
        (x, y, Z),
        (X, y, Z),
        (X, Y, Z),
        (x, Y, Z),
    ]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    triangles = []
    for a, b, c, d in faces:
        triangles.extend(
            [(vertices[a], vertices[b], vertices[c]), (vertices[a], vertices[c], vertices[d])]
        )
    return {"name": cls, "ifc_class": cls, "triangles": triangles}


def room(size=6, height=3):
    return [
        box((0, 0, 0), (size, size, height)),
        box((-0.2, -0.2, -0.2), (size + 0.2, size + 0.2, 0), "IfcSlab"),
    ]


def test_deterministic_forward_smooth_single_storey(plan):
    meshes = room()
    a, meta = plan(meshes, 1.6, 42, 101, 25)
    assert (a, meta) == plan(meshes, 1.6, 42, 101, 25)
    assert len(a) == 101
    assert meta["method"] == "ifc_space"
    assert all(p[2] == 1.6 for p in a)
    heading = meta["heading"]
    velocities = [
        sum((q[i] - p[i]) * heading[i] for i in range(3)) for p, q in zip(a, a[1:], strict=False)
    ]
    assert min(velocities) > 0
    assert velocities[0] < velocities[50] / 10
    assert all(0.3 < p[0] < 5.7 and 0.3 < p[1] < 5.7 for p in a)


@pytest.mark.parametrize("height", [1.5, 1.75])
def test_reject_low_headroom(plan, height):
    with pytest.raises(ValueError):
        plan(room(3, height), 1.6, 1, 20, 20)


def test_reject_empty_open_and_broken_space(plan):
    for meshes in [[], [box((0, 0, -0.2), (5, 5, 0), "IfcSlab")]]:
        with pytest.raises(ValueError):
            plan(meshes, 1.6, 1, 20, 20)
    meshes = room()
    meshes[0]["triangles"].pop()
    with pytest.raises(ValueError, match="watertight"):
        plan(meshes, 1.6, 1, 20, 20)


def test_obstacle_fills_room(plan):
    meshes = room(3) + [box((0.1, 0.1, 0), (2.9, 2.9, 2.5), "IfcFurnishingElement")]
    with pytest.raises(ValueError):
        plan(meshes, 1.6, 1, 20, 20)


def test_wall_not_crossed(plan):
    meshes = room(6) + [box((2.8, 0, 0), (3.2, 6, 2.9), "IfcWall")]
    path, _ = plan(meshes, 1.6, 123, 120, 24)
    assert all(p[0] < 2.5 for p in path) or all(p[0] > 3.5 for p in path)


def test_triangle_shape_not_aabb(plan):
    # The long wall's AABB spans the room but its actual triangles are diagonal.
    meshes = room(8)
    meshes.append(
        {
            "ifc_class": "IfcWall",
            "triangles": [((0, 0, 0), (8, 8, 0), (8, 8, 3)), ((0, 0, 0), (8, 8, 3), (0, 0, 3))],
        }
    )
    path, _ = plan(meshes, 1.6, 3, 50, 25)
    assert all(abs(p[0] - p[1]) / math.sqrt(2) > 0.3 for p in path)


def test_enclosed_floor_fallback(plan):
    meshes = [box((0, 0, -0.2), (5, 5, 0), "IfcSlab"), box((0, 0, 3), (5, 5, 3.2), "IfcSlab")]
    meshes += [
        box((-0.2, 0, 0), (0, 5, 3), "IfcWall"),
        box((5, 0, 0), (5.2, 5, 3), "IfcWall"),
        box((0, -0.2, 0), (5, 0, 3), "IfcWall"),
        box((0, 5, 0), (5, 5.2, 3), "IfcWall"),
    ]
    path, meta = plan(meshes, 1.6, 4, 50, 25)
    assert meta["method"] == "enclosed_floor"
    assert all(p[2] == 1.6 for p in path)


def test_input_limits(plan):
    with pytest.raises(ValueError):
        plan(room(), float("nan"), 1, 50, 25)
    with pytest.raises(ValueError):
        plan(room(), 1.6, 1, 0, 25)
    with pytest.raises(ValueError):
        plan(room(), 1.6, 1, 50, 0)


def test_adjacent_spaces_multi_room_attempt(plan):
    meshes = [
        box((0, 0, 0), (3, 4, 3)),
        box((3, 0, 0), (6, 4, 3)),
        box((0, 0, -0.2), (6, 4, 0), "IfcSlab"),
    ]
    path, meta = plan(meshes, 1.6, 10, 200, 25)
    assert meta["multi_room_attempted"]
    assert meta["visited_space_indices"] == [0, 1]
    assert min(p[0] for p in path) < 3 < max(p[0] for p in path)


def test_closed_partition_blocks_room_transition(plan):
    meshes = [
        box((0, 0, 0), (3, 4, 3)),
        box((3, 0, 0), (6, 4, 3)),
        box((0, 0, -0.2), (6, 4, 0), "IfcSlab"),
        box((2.95, 0, 0), (3.05, 4, 3), "IfcWall"),
    ]
    path, meta = plan(meshes, 1.6, 10, 200, 25)
    assert meta["multi_room_attempted"]
    assert len(meta["visited_space_indices"]) == 1
    assert all(p[0] < 2.65 for p in path) or all(p[0] > 3.35 for p in path)


def test_single_frame(plan):
    path, meta = plan(room(), 1.6, 1, 1, 25)
    assert len(path) == 1
    assert meta["path_length_m"] == 0


def test_graph_bridges_bounded_doorway_gap(plan):
    meshes = [
        box((0, 0, 0), (4, 5, 3)),
        box((4.2, 0, 0), (8.2, 5, 3)),
        box((0, 0, -0.2), (8.2, 5, 0), "IfcSlab"),
        box((0, 0, 3), (8.2, 5, 3.2), "IfcSlab"),
    ]
    meshes += [
        box((-0.2, 0, 0), (0, 5, 3), "IfcWall"),
        box((8.2, 0, 0), (8.4, 5, 3), "IfcWall"),
        box((0, -0.2, 0), (8.2, 0, 3), "IfcWall"),
        box((0, 5, 0), (8.2, 5.2, 3), "IfcWall"),
        box((4, 0, 0), (4.2, 1.7, 3), "IfcWall"),
        box((4, 3.2, 0), (4.2, 5, 3), "IfcWall"),
    ]
    path, meta = plan(meshes, 1.6, 32, 600, 25)
    assert meta["method"] == "enclosed_floor_graph"
    assert meta["visited_space_indices"] == [0, 1]
    assert meta["path_length_m"] > 5
    assert min(p[0] for p in path) < 4 < max(p[0] for p in path)
    assert all(2.0 < p[1] < 2.9 for p in path if 4.0 < p[0] < 4.2)


def test_gap_without_physical_roof_not_bridged(plan):
    meshes = [
        box((0, 0, 0), (3, 4, 3)),
        box((3.2, 0, 0), (6.2, 4, 3)),
        box((0, 0, -0.2), (6.2, 4, 0), "IfcSlab"),
    ]
    path, meta = plan(meshes, 1.6, 10, 200, 25)
    assert len(meta["visited_space_indices"]) == 1


def test_semantic_space_without_physical_support_rejected(plan):
    meshes = [box((0, 0, 0), (5, 5, 3)), box((0, 0, 2.8), (5, 5, 3), "IfcSlab")]
    with pytest.raises(ValueError):
        plan(meshes, 1.6, 1, 50, 25)


@pytest.fixture
def tour_plan():
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ego_tour_plan"]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_ego_tour_plan"]


def semantic_fixture(route=(1, 2, 3, 2, 1)):
    meshes = []
    for sid in (1, 2, 3):
        mesh = box(((sid - 1) * 3, 0, 0), (sid * 3, 3, 3))
        mesh["ifc_id"] = sid
        meshes.append(mesh)
    meshes.append(box((0, 0, -0.2), (9, 3, 0), "IfcSlab"))
    edges = []
    for sid in (1, 2):
        x = sid * 3
        edges.append(dict(source=sid, target=sid + 1, connector_id=100 + sid, kind="opening", portal_triangles=[((x, 0.5, 0), (x, 2.5, 0), (x, 2.5, 3)), ((x, 0.5, 0), (x, 2.5, 3), (x, 0.5, 3))]))
    semantic = dict(graph_validation_status="validated_topology", edges=edges,
                    nodes=[dict(id=sid, classification="tourable_interior") for sid in (1, 2, 3)],
                    component_tours=[dict(component_id=0, space_ids=[1, 2, 3], route_space_ids=list(route), coverage=1, tour=[dict(space_id=sid) for sid in route])])
    return meshes, semantic


def test_semantic_tour_preserves_every_revisit_and_minimum_duration(tour_plan):
    meshes, semantic = semantic_fixture()
    positions, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    assert metadata["tour"] == [1, 2, 3, 2, 1]
    assert metadata["visited_spaces"] == [1, 2, 3]
    assert metadata["tourable_rooms_visited"] == metadata["tourable_rooms_total"] == 3
    assert metadata["coverage"] == 1
    assert metadata["route_length_m"] >= 12
    assert len(positions) > 2
    assert max(math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False)) <= 1.0 / 10 + 1e-9
    for visit in metadata["visits"]:
        frame = visit["frame"]
        assert metadata["frame_space_ids"][frame] == visit["space_id"]
        assert positions[frame][:2] == tuple(visit["observation_point_m"][:2])
    longer, _ = tour_plan(meshes, semantic, 1.65, len(positions) + 20, 10)
    assert len(longer) == len(positions) + 20


def test_semantic_blocking_wall_fails_not_partial_success(tour_plan):
    meshes, semantic = semantic_fixture()
    meshes.append(box((2.95, 0, 0), (3.05, 3, 3), "IfcWall"))
    with pytest.raises(ValueError, match="Required tour transition failed"):
        tour_plan(meshes, semantic, 1.65, 10, 10)


@pytest.mark.parametrize("geometry", ["closed", "malformed", "staged"])
def test_semantic_ignores_all_door_geometry(tour_plan, geometry):
    meshes, semantic = semantic_fixture()
    for edge in semantic["edges"]:
        edge["kind"] = "door"
    expected_positions, expected_metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    for edge in semantic["edges"]:
        x = edge["source"] * 3
        door = dict(box((x - .05, 0, 0), (x + .05, 3, 3), "IfcDoor"),
                    ifc_id=edge["connector_id"])
        if geometry == "malformed":
            door["triangles"] = [[(float("nan"), 0, 0)], None]
        elif geometry == "staged":
            # Neither malformed staged poses nor sweep metadata may influence
            # navigation. The actual opening evidence remains on the edge.
            door.update(closed_triangles=None, door_open_angle_degrees="invalid",
                        door_sweep_bounds_m=[None])
        meshes.append(door)
    positions, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    assert positions == expected_positions
    assert metadata["tour"] == expected_metadata["tour"] == [1, 2, 3, 2, 1]
    assert metadata["coverage"] == 1
    assert metadata.get("human_radius_m", 0) == 0
    assert not metadata.get("door_actions")
    assert metadata["transitions"] == expected_metadata["transitions"]

    # Ignoring the door never grants an exemption for its real wall.
    meshes.append(box((2.95, 0, 0), (3.05, 3, 3), "IfcWall"))
    with pytest.raises(ValueError, match="Required tour transition failed"):
        tour_plan(meshes, semantic, 1.65, 2, 10)


@pytest.mark.parametrize("kind", ["door", "opening"])
def test_semantic_does_not_use_unevidenced_shared_boundary(tour_plan, kind):
    meshes, semantic = semantic_fixture()
    for edge in semantic["edges"]:
        edge["kind"] = kind
        edge["portal_triangles"] = [tuple((x, y + 20, z) for x, y, z in tri) for tri in edge["portal_triangles"]]
    with pytest.raises(ValueError, match="Required tour transition failed"):
        tour_plan(meshes, semantic, 1.65, 10, 10)


def test_semantic_component_and_graph_fail_closed(tour_plan):
    meshes, semantic = semantic_fixture()
    with pytest.raises(ValueError, match="Unknown component_id"):
        tour_plan(meshes, semantic, 1.65, 10, 10, 99)
    semantic["graph_validation_status"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        tour_plan(meshes, semantic, 1.65, 10, 10)


def test_semantic_missing_room_geometry_fails(tour_plan):
    meshes, semantic = semantic_fixture()
    with pytest.raises(ValueError, match="Missing physical support or evaluated spaces"):
        tour_plan(meshes[1:], semantic, 1.65, 10, 10)



def staircase_fixture():
    meshes = [box((0, 0, 0), (3, 3, 3)), box((4.2, 0, 0.8), (7.2, 3, 3.8))]
    meshes[0]["ifc_id"], meshes[1]["ifc_id"] = 1, 2
    meshes.extend([box((0, 0, -0.2), (3, 3, 0), "IfcSlab"), box((4.2, 0, 0.6), (7.2, 3, 0.8), "IfcSlab")])
    flight = dict(ifc_class="IfcStairFlight", ifc_id=100, triangles=[])
    for i in range(4):
        flight["triangles"].extend(box((3 + i * 0.3, 0.5, -0.2), (3.3 + i * 0.3, 2.5, (i + 1) * 0.2), "IfcStairFlight")["triangles"])
    meshes.append(flight)
    semantic = dict(graph_validation_status="validated_topology",
                    edges=[dict(source=1, target=2, connector_id=100, kind="stair", evidence=dict(connector_ids=[100]))],
                    nodes=[dict(id=sid, classification="tourable_interior") for sid in (1, 2)],
                    component_tours=[dict(component_id=0, space_ids=[1, 2], route_space_ids=[1, 2, 1], coverage=1, tour=[dict(space_id=sid) for sid in (1, 2, 1)])])
    return meshes, semantic


def test_semantic_stairs_follow_treads_continuously_both_directions(tour_plan):
    meshes, semantic = staircase_fixture()
    positions, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    assert metadata["tour"] == [1, 2, 1]
    assert max(p[2] for p in positions) == pytest.approx(2.45)
    assert positions[0][2] == positions[-1][2] == 1.65
    assert max(math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False)) <= 1.0 / 10 + 1e-9
    assert any(sid is None for sid in metadata["frame_space_ids"])


def test_semantic_stairs_missing_tread_fails(tour_plan):
    meshes, semantic = staircase_fixture()
    del meshes[-1]["triangles"][12:24]
    with pytest.raises(ValueError, match="Required tour transition failed"):
        tour_plan(meshes, semantic, 1.65, 2, 10)



@pytest.mark.parametrize("kind", ["door", "opening"])
def test_semantic_narrow_real_aperture_needs_no_body_clearance(tour_plan, kind):
    meshes, semantic = semantic_fixture()
    low, high = 1.45, 1.55  # Only 0.10 m wide: no human-radius clearance.
    for edge in semantic["edges"]:
        edge["kind"] = kind
        edge["portal_triangles"] = [tuple((x, low if y == 0.5 else high, z) for x, y, z in tri) for tri in edge["portal_triangles"]]
        x = edge["source"] * 3
        meshes.extend([
            box((x - .05, 0, 0), (x + .05, low, 3), "IfcWall"),
            box((x - .05, high, 0), (x + .05, 3, 3), "IfcWall"),
        ])
    positions, metadata = tour_plan(meshes, semantic, 1.65, 10, 10)
    assert metadata["tour"] == [1, 2, 3, 2, 1]
    assert metadata["coverage"] == 1
    for x in (3, 6):
        crossings = []
        for a, b in zip(positions, positions[1:], strict=False):
            if a[0] != b[0] and min(a[0], b[0]) <= x <= max(a[0], b[0]):
                t = (x - a[0]) / (b[0] - a[0])
                crossings.append(a[1] + t * (b[1] - a[1]))
        assert crossings
        assert all(low < y < high for y in crossings)
    assert max(math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False)) <= .1 + 1e-9


def test_semantic_stair_room_approach_ignores_furniture(tour_plan):
    meshes, semantic = staircase_fixture()
    expected = tour_plan(meshes, semantic, 1.65, 10, 10)
    meshes.append(box((2.1, 0, 0), (2.4, 3, 2.8), "IfcFurnishingElement"))
    assert tour_plan(meshes, semantic, 1.65, 10, 10) == expected


def test_semantic_component_default_is_smallest_id_and_explicit_selects(tour_plan):
    meshes, semantic = semantic_fixture()
    semantic["component_tours"] = [dict(component_id=5, space_ids=[2], route_space_ids=[2], coverage=1, tour=[dict(space_id=2)]), dict(component_id=2, space_ids=[1], route_space_ids=[1], coverage=1, tour=[dict(space_id=1)])]
    _, default = tour_plan(meshes, semantic, 1.65, 10, 10)
    assert default["component_id"] == 2
    assert default["tour"] == [1]
    _, explicit = tour_plan(meshes, semantic, 1.65, 10, 10, 5)
    assert explicit["component_id"] == 5
    assert explicit["tour"] == [2]



@pytest.mark.parametrize("slotted", [False, True])
def test_camera_animation_is_linear_between_validated_samples(slotted):
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_ego_linear_camera_animation"]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    frames = [SimpleNamespace(interpolation="BEZIER") for _ in range(3)]
    curve = SimpleNamespace(keyframe_points=frames)
    action = SimpleNamespace(layers=[SimpleNamespace(strips=[SimpleNamespace(channelbags=[SimpleNamespace(fcurves=[curve])])])]) if slotted else SimpleNamespace(fcurves=[curve])
    namespace["_ego_linear_camera_animation"](action)
    assert all(frame.interpolation == "LINEAR" for frame in frames)
    with pytest.raises(RuntimeError, match="animation curves"):
        namespace["_ego_linear_camera_animation"](SimpleNamespace())



def test_semantic_furniture_does_not_displace_centroid_observation(tour_plan):
    meshes, semantic = semantic_fixture()
    meshes.extend([
        box((1, 1, 0), (1.1, 2, 2.5), "IfcFurnishingElement"),
        box((1.9, 1, 0), (2, 2, 2.5), "IfcFurnishingElement"),
        box((1, 1, 0), (2, 1.1, 2.5), "IfcFurnishingElement"),
        box((1, 1.9, 0), (2, 2, 2.5), "IfcFurnishingElement"),
    ])
    _, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    assert metadata["coverage"] == 1
    assert metadata["observation_points_m"]["1"][:2] == [1.5, 1.5]
    assert metadata["tour"] == [1, 2, 3, 2, 1]


def test_semantic_tries_each_evidenced_portal_patch(tour_plan):
    meshes, semantic = semantic_fixture()
    edge = semantic["edges"][0]
    invalid = [((20, 0, 0), (22, 0, 0), (22, 0, 3)), ((20, 0, 0), (22, 0, 3), (20, 0, 3))]
    edge["portal_variants"] = [invalid, edge["portal_triangles"]]
    _, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    assert metadata["coverage"] == 1
    assert metadata["transitions"][0]["portal_variant"] == 1


def test_semantic_full_room_furniture_does_not_block_or_change_timing(tour_plan):
    meshes, semantic = semantic_fixture()
    expected = tour_plan(meshes, semantic, 1.65, 2, 10)
    meshes.append(box((.01, .01, 0), (8.99, 2.99, 2.9), "IfcFurnishingElement"))
    positions, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    assert (positions, metadata) == expected
    assert metadata["tour"] == [1, 2, 3, 2, 1]
    assert [v["space_id"] for v in metadata["visits"]] == metadata["tour"]
    assert max(math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False)) <= .1 + 1e-9


def test_stair_landing_keeps_continuous_checked_tread_path(tour_plan):
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_ego_tour_plan")
    solve = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == "solve")
    def expose_before_assignment(body, name):
        stop = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        return body[:stop] + [ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))]
    solve.body = expose_before_assignment(solve.body, "start")
    function.body = expose_before_assignment(function.body, "search_attempts")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
    meshes, semantic = staircase_fixture()
    meshes[1] = dict(box((4.075, 0, 1.0), (7.2, 3, 4)), ifc_id=2)
    meshes[3] = box((4.075, 0, 0.6), (7.2, 3, 1.0), "IfcSlab")
    geometry = namespace["_ego_tour_plan"](meshes, semantic, 1.65, 2, 10)
    a, b = (3.75, 1.5, 0.6), (3.9, 1.5, 0.8)
    predicates = geometry["solve"](a, [b], 1, 2, semantic["edges"][0])
    # With no body capsule, this direct tread-to-tread segment is lawful.
    # A forced raise-before-advance vertex is no longer required.
    assert predicates["segment"](a, b)
    link = predicates["connection"](a, b)
    assert link == (a, b)
    assert all(predicates["segment"](u, v) for u, v in zip(link, link[1:], strict=False))
    assert predicates["connection"](b, a) == tuple(reversed(link))

    positions, metadata = tour_plan(meshes, semantic, 1.65, 2, 10)
    feet = [(x, y, z - 1.65) for x, y, z in positions]
    changes = [v[2] - u[2] for u, v in zip(feet, feet[1:], strict=False)]
    assert any(dz > 1e-7 for dz in changes)
    assert any(dz < -1e-7 for dz in changes)
    assert max(math.dist(u, v) for u, v in zip(feet, feet[1:], strict=False)) <= .1 + 1e-9
    assert metadata["route_space_ids"] == [1, 2, 1]
    assert all(predicates["segment"](u, v) for u, v in zip(feet, feet[1:], strict=False))
