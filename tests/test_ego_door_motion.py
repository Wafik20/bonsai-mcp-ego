"""Pure continuous-sweep tests, without Blender imports."""
import ast
import math
from pathlib import Path

import pytest


def load_helpers():
    root = Path(__file__).resolve().parents[1]
    bridge = root / "blender_addon/bonsai_bridge.py"
    staged = root / "dev/ego_tour_validation/door_motion_helpers.py"
    names = {"_ego_leaf_motion_clear", "_ego_point_inside_triangles"}
    tree = ast.parse(bridge.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        functions = [n for n in ast.parse(staged.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(staged), "exec"), namespace)
    return namespace["_ego_leaf_motion_clear"], namespace["_ego_point_inside_triangles"]


clear, inside = load_helpers()


def box(x0, x1, y0, y1, z0=0., z1=2.):
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [tuple(v[i] for i in indices) for a, b, c, d in quads
            for indices in ((a, b, c), (a, c, d))]


LEAF = box(-1., 0., -.04, 0.)
HINGE = (0., 0., 0.)


def test_quarter_turn_hits_between_clear_endpoints():
    obstacle = box(-.72, -.69, .69, .72, .5, 1.5)
    assert clear(LEAF, HINGE, 0, 0, obstacle)
    assert clear(LEAF, HINGE, -90, -90, obstacle)
    assert not clear(LEAF, HINGE, 0, -90, obstacle)
    assert not clear(LEAF, HINGE, -90, 0, obstacle)


def test_frame_boundary_contact_and_fixed_floor_header_allowed():
    frame = box(0., .1, 0., 1.2)
    floor = box(-2., 2., -2., 2., -.2, 0.)
    header = box(-2., 2., -2., 2., 2., 2.2)
    assert clear(LEAF, HINGE, 0, -90, frame + floor + header)
    assert not clear(LEAF, HINGE, 0, -91, frame)


def test_thin_positive_frame_penetration_rejected():
    assert not clear(LEAF, HINGE, 0, -90, box(-.0001, .1, .4, .6))


def test_contact_tolerance_is_explicit():
    assert clear(LEAF, HINGE, 0, -90, box(-.2e-6, .1, .4, .6))
    assert not clear(LEAF, HINGE, 0, -90, box(-5e-6, .1, .4, .6))


def test_signed_rotations_use_the_correct_side():
    above = box(-.72, -.69, .69, .72, .5, 1.5)
    below = box(-.72, -.69, -.72, -.69, .5, 1.5)
    assert clear(LEAF, HINGE, 0, 90, above)
    assert not clear(LEAF, HINGE, 0, 90, below)
    assert clear(LEAF, HINGE, 0, -90, below)


def test_obstacles_outside_curved_sweep_not_just_endpoint_hull():
    assert clear(LEAF, HINGE, 0, -90, box(-.82, -.80, .80, .82, .5, 1.5))
    assert clear(LEAF, HINGE, 0, -90, box(.0001, .01, .4, .6, .5, 1.5))


def test_subdegree_between_pose_collision():
    narrow = box(-1., 0., -1e-5, 0.)
    angle = math.radians(.23)
    x, y = -.9999 * math.cos(angle), .9999 * math.sin(angle)
    obstacle = box(x - 2e-5, x + 2e-5, y - 2e-5, y + 2e-5, .5, 1.5)
    for pose in (0., -.5, -1.):
        assert clear(narrow, HINGE, pose, pose, obstacle)
    assert not clear(narrow, HINGE, 0, -1, obstacle)


def test_containment_centroid_and_boundary():
    solid = box(-1., 1., -1., 1., -1., 1.)
    assert inside((0., 0., 0.), solid)
    assert inside((1., 0., 0.), solid)
    assert not inside((1.01, 0., 0.), solid)
    assert not inside((2., 2., 2.), solid)
    assert not inside((0., 0., 0.), [])
    assert inside((0., 0., 0.), [tuple(reversed(t)) for t in solid])


@pytest.mark.parametrize("angle", [float("nan"), float("inf")])
def test_nonfinite_motion_fails_closed(angle):
    assert not clear(LEAF, HINGE, 0, angle, [])


def test_translated_world_geometry():
    shift = (10000., -20000., 50.)
    def moved(triangles):
        return [tuple(tuple(p[i] + shift[i] for i in range(3)) for p in tri)
                for tri in triangles]
    assert clear(moved(LEAF), shift, 0, -90, moved(box(0., .1, 0., 1.2)))
    assert not clear(moved(LEAF), shift, 0, -90,
                     moved(box(-.72, -.69, .69, .72, .5, 1.5)))
