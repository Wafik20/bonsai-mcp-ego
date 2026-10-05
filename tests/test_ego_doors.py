"""Leaf-only temporary geometry and failure restoration regressions."""
import ast
import contextlib
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name, namespace=None):
    root = Path(__file__).resolve().parents[1]
    source = root / "tests/fixtures/retired_ego_door_helpers.py"
    tree = ast.parse(source.read_text())
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        source = root / "blender_addon/bonsai_bridge.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"contextlib": contextlib, **(namespace or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def geometry():
    vertices = list(product((0., .8), (.073, .124), (0., 2.032)))
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if sum(vertices[a][k] != vertices[b][k] for k in range(3)) == 1]
    # Independent retained frame component, never moved by leaf helper.
    vertices += [(-.076, 0, 0), (-.076, 0, 2.108)]
    edges += [(8, 9)]
    corners = [(0, .124), (.051, .124), (.051, .924), (0, .924)]
    segments = [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
    return vertices, edges, segments


def test_leaf_only_unique_authored_swing():
    vertices, edges, segments = geometry()
    spec = load("_ego_leaf_geometry")(vertices, edges, .8, 2.032, segments)
    assert spec["indices"] == list(range(8))
    assert spec["hinge"] == [0, .124]
    assert spec["angle_degrees"] == 90
    assert all(-1e-8 <= p[0] <= .05100001 for p in spec["vertices"])
    assert vertices[8:] == [(-.076, 0, 0), (-.076, 0, 2.108)]


@pytest.mark.parametrize("change", ["missing_plan", "wrong_width", "welded_frame", "ambiguous_leaf"])
def test_unsupported_leaf_fails_closed(change):
    vertices, edges, segments = geometry()
    width = .8
    if change == "missing_plan":
        segments = []
    elif change == "wrong_width":
        width = .9
    elif change == "welded_frame":
        edges.append((0, 8))
    else:
        vertices += vertices[:8]
        edges += [(a + 10, b + 10) for a, b in edges if b < 8]
    with pytest.raises(ValueError):
        load("_ego_leaf_geometry")(vertices, edges, width, 2.032, segments)


class Mesh:
    def __init__(self):
        self.vertices = [NS(co=(0, 0, 0)), NS(co=(1, 0, 0))]
        self.users = 0

    def copy(self):
        result = Mesh()
        result.vertices = [NS(co=v.co) for v in self.vertices]
        return result

    def update(self):
        pass


def lifecycle(fail_second=False):
    original = Mesh()
    # Two objects intentionally share one original mesh datablock.
    objects = [NS(data=original, identifier=i) for i in (10, 20)]
    entities = {i: NS(id=lambda i=i: i, is_a=lambda kind: kind == "IfcDoor") for i in (10, 20)}
    removed = []
    def spec(ifc, entity, obj, scale):
        if fail_second and entity.id() == 20:
            raise ValueError("second door rejected")
        return dict(indices=[0], vertices=[(0, 1, 0)], door_id=entity.id())
    namespace = dict(bpy=NS(context=NS(view_layer=NS(update=lambda: None)), data=NS(meshes=NS(remove=removed.append))),
                     _element_for_object=lambda obj: entities[obj.identifier], _ego_door_leaf_spec=spec,
                     _ego_door_motion_record=lambda obj, original, record, scale: record)
    manager = load("_ego_temporary_open_doors", namespace)
    scene = NS(objects=objects)
    ifc = NS(by_id=entities.__getitem__)
    selected = {"route_space_ids": [1, 2, 3]}
    edges = [dict(kind="door", source=1, target=2, connector_id=10),
             dict(kind="door", source=2, target=3, connector_id=20)]
    return manager, scene, ifc, selected, edges, original, removed


@pytest.mark.parametrize("partial", [False, True])
def test_failure_restores_shared_original_and_removes_copies(partial):
    manager, scene, ifc, selected, edges, original, removed = lifecycle(partial)
    with pytest.raises((ValueError, RuntimeError)), manager(scene, ifc, selected, edges, 1):
        assert all(o.data is not original for o in scene.objects)
        assert scene.objects[0].data is not scene.objects[1].data
        assert original.vertices[0].co == (0, 0, 0)
        assert all(o.data.vertices[1].co == (1, 0, 0) for o in scene.objects)
        raise RuntimeError("render failure")
    assert all(o.data is original for o in scene.objects)
    assert original.vertices[0].co == (0, 0, 0)
    assert len(removed) == (1 if partial else 2)


def test_unused_doors_remain_unchanged():
    manager, scene, ifc, selected, edges, original, removed = lifecycle()
    selected["route_space_ids"] = [1, 2]
    with manager(scene, ifc, selected, edges, 1) as records:
        assert [r["door_id"] for r in records] == [10]
        assert scene.objects[1].data is original
    assert len(removed) == 1


@pytest.mark.parametrize("alternative_kind", ["door", "opening"])
def test_ambiguous_connector_selection_fails_before_changes(alternative_kind):
    manager, scene, ifc, selected, edges, original, removed = lifecycle()
    edges.append(dict(kind=alternative_kind, source=1, target=2, connector_id=20))
    with pytest.raises(ValueError, match="Ambiguous"), manager(scene, ifc, selected, edges, 1):
        pytest.fail("ambiguous entry must fail")
    assert all(o.data is original for o in scene.objects)
    assert not removed


@pytest.mark.parametrize("angle", [90, -90])
def test_exact_arc_bounds_include_extrema_and_not_full_circle(angle):
    bounds = load("_ego_door_arc_bounds")
    matrix = [[1, 0, 0, 4], [0, 1, 0, -2], [0, 0, 1, 3], [0, 0, 0, 1]]
    lo, hi = bounds([(1, 0, 0), (1, 0, 2)], (0, 0), angle, matrix, 1)
    assert lo == pytest.approx([4, -2 if angle == 90 else -3, 3])
    assert hi == pytest.approx([5, -1 if angle == 90 else -2, 5])


def staged_functions():
    at_frame = load("_ego_door_angles_at_frame")
    apply = load("_ego_apply_door_angles")
    validate = load("_ego_validate_door_schedule", {"_ego_door_angles_at_frame": at_frame})
    handler = load("_ego_door_frame_handler", {"_ego_door_angles_at_frame": at_frame, "_ego_apply_door_angles": apply})
    return at_frame, apply, validate, handler


def test_signed_rigid_callback_random_frame_order_keeps_frame_vertices():
    at_frame, apply, validate, handler = staged_functions()
    mesh = Mesh()
    mesh.vertices[0].co = (1, 0, 2)
    mesh.vertices[1].co = (9, 9, 9)
    records = [dict(door_id=10, angle_degrees=-90, indices=[0], _object=NS(data=mesh),
                    _original_leaf_vertices=[(1, 0, 2)], _hinge_local=(0, 0))]
    actions = [dict(door_id=10, start_frame=0, end_frame=2, from_angle_degrees=0, to_angle_degrees=-90),
               dict(door_id=10, start_frame=3, end_frame=5, from_angle_degrees=-90, to_angle_degrees=0)]
    navigation = dict(door_actions=actions, frame_door_angles=[{"10": v} for v in (0, -45, -90, -90, -45, 0)])
    validate(records, navigation, 6, 2)
    errors = []
    callback = handler(records, actions, 6, errors)
    for frame in [2, 0, 4, 1, 5, 2]:
        callback(NS(frame_current=frame + 1, frame_subframe=0))
        import math
        angle = math.radians(at_frame(records, actions, frame)[10])
        assert mesh.vertices[0].co == pytest.approx((math.cos(angle), math.sin(angle), 2))
        assert mesh.vertices[1].co == (9, 9, 9)
    assert not errors


@pytest.mark.parametrize("failure", ["short", "overlap", "wrong_state", "missing_door", "wrong_full_open"])
def test_schedule_rejects_unsafe_or_ambiguous_actions(failure):
    _, _, validate, _ = staged_functions()
    records = [dict(door_id=10, angle_degrees=90)]
    actions = [dict(door_id=10, start_frame=0, end_frame=2, from_angle_degrees=0, to_angle_degrees=90)]
    states = [{"10": angle} for angle in (0, 45, 90, 90)]
    if failure == "short":
        actions[0]["end_frame"] = 1
    elif failure == "overlap":
        actions.append(dict(door_id=10, start_frame=1, end_frame=3, from_angle_degrees=90, to_angle_degrees=0))
    elif failure == "wrong_state":
        actions[0]["from_angle_degrees"] = 90
    elif failure == "missing_door":
        records.append(dict(door_id=20, angle_degrees=-90))
    else:
        states[1] = {"10": 90}
    with pytest.raises(ValueError):
        validate(records, dict(door_actions=actions, frame_door_angles=states), 4, 2)


def test_handler_reports_error_instead_of_silently_accepting_bad_frame():
    _, _, _, factory = staged_functions()
    errors = []
    handler = factory([], [], 2, errors)
    with pytest.raises(ValueError):
        handler(NS(frame_current=3, frame_subframe=0))
    assert len(errors) == 1


def test_variable_angles_preserve_nonblocking_open_state():
    at_frame, _, validate, _ = staged_functions()
    records = [dict(door_id=10, angle_degrees=-90), dict(door_id=20, angle_degrees=90)]
    actions = [dict(door_id=10, start_frame=0, end_frame=2, from_angle_degrees=0, to_angle_degrees=-53),
               dict(door_id=20, start_frame=3, end_frame=5, from_angle_degrees=0, to_angle_degrees=67)]
    states = [{str(k): v for k, v in at_frame(records, actions, i).items()} for i in range(6)]
    assert validate(records, dict(door_actions=actions, frame_door_angles=states), 6, 2) == actions
    assert states[-1] == {"10": -53, "20": 67}



@pytest.mark.parametrize("door_hidden", [False, True])
def test_render_hides_doors_and_restores_visibility_after_failure(tmp_path, monkeypatch, door_hidden):
    import os
    import sys

    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=object))
    door = NS(ifc_class="IfcDoor", hide_render=door_hidden)
    wall = NS(ifc_class="IfcWall", hide_render=False)
    render = NS(**dict.fromkeys([
        "engine", "filepath", "resolution_x", "resolution_y", "resolution_percentage",
        "fps", "fps_base", "pixel_aspect_x", "pixel_aspect_y", "use_file_extension",
        "use_sequencer", "use_compositing", "use_border", "use_crop_to_border",
    ]))
    render.image_settings = NS(color_mode="RGBA", file_format="PNG")
    render.ffmpeg = NS(format=None, codec=None, audio_codec=None,
                       constant_rate_factor=None, ffmpeg_preset=None)
    shading = NS(light=None, color_type=None, show_shadows=None,
                 show_cavity=None, show_xray=None)
    frames = []
    scene = NS(objects=[door, wall], render=render, display=NS(shading=shading),
               frame_current=7, frame_subframe=.5,
               frame_set=lambda frame, subframe: frames.append((frame, subframe)))

    def fail_camera_creation(name):
        assert door.hide_render is True
        assert wall.hide_render is False
        raise RuntimeError("camera creation failed")

    planner_calls = []
    def plan(*args):
        planner_calls.append(args)
        return [(1, 1, 1.65)], {}

    namespace = dict(
        os=os,
        bpy=NS(context=NS(evaluated_depsgraph_get=lambda: None),
               data=NS(cameras=NS(new=fail_camera_creation))),
        _ego_evaluated_meshes=lambda *args: [],
        _ego_portal_geometry=lambda *args: [],
        _get_loaded_ifc=lambda: None,
        _ego_tour_plan=plan,
        _ifc_class_for_object=lambda obj: obj.ifc_class,
    )
    args = dict(output_path=tmp_path / "tour.mp4", poses_path=tmp_path / "tour.json",
                width=640, height=480, fps=10, camera_height=1.65,
                frame_count=10, component_id=None)
    render_scene = load("_ego_render_scene", namespace)
    with pytest.raises(RuntimeError, match="camera creation failed"):
        render_scene(args, scene, 1, {"edges": []}, {"route_space_ids": [1]})
    assert len(planner_calls) == 1
    assert door.hide_render is door_hidden
    assert wall.hide_render is False
    assert frames == [(7, .5)]
    assert not args["output_path"].exists()
    assert not args["poses_path"].exists()
