"""Historical staging helpers, retained for reference; not used by ego tours."""
import contextlib

def _ego_leaf_geometry(vertices, edges, width, height, plan_segments):
    """Recognize one closed rectangular leaf and its authored 90-degree plan swing.

    Deliberately narrow: unsupported, ambiguous or already-open leaves fail closed.
    Coordinates, dimensions and plan segments must share object-local units.
    """
    import math
    from itertools import product

    tolerance = 0.0002
    adjacency = [set() for _ in vertices]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(range(len(vertices)))
    islands = []
    while remaining:
        todo = [remaining.pop()]
        island = set(todo)
        while todo:
            for neighbor in adjacency[todo.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    island.add(neighbor)
                    todo.append(neighbor)
        islands.append(sorted(island))
    candidates = []
    for island in islands:
        pts = [vertices[i] for i in island]
        lo = [min(p[k] for p in pts) for k in range(3)]
        hi = [max(p[k] for p in pts) for k in range(3)]
        size = [hi[k] - lo[k] for k in range(3)]
        if (len(island) != 8 or abs(size[0] - width) > tolerance or
                abs(size[2] - height) > tolerance or not 0.015 <= size[1] <= 0.12):
            continue
        corners = list(product(*zip(lo, hi, strict=True)))
        if any(sum(math.dist(p, q) < tolerance for p in pts) != 1 for q in corners):
            continue
        candidates.append((island, lo, hi))
    if len(candidates) != 1 or len(islands) < 2:
        raise ValueError("Door leaf is not one unambiguous detached cuboid with retained frame")
    island, lo, hi = candidates[0]
    # Four authored straight plan segments must describe an open-leaf rectangle.
    if len(plan_segments) != 4 or any(len(segment) != 2 for segment in plan_segments):
        raise ValueError("Door lacks a supported authored open-leaf plan rectangle")
    points = [p for segment in plan_segments for p in segment]
    target_lo = [min(p[k] for p in points) for k in range(2)]
    target_hi = [max(p[k] for p in points) for k in range(2)]
    if abs(target_hi[0] - target_lo[0] - (hi[1] - lo[1])) > tolerance or abs(target_hi[1] - target_lo[1] - width) > tolerance:
        raise ValueError("Door plan does not describe a perpendicular open leaf")
    expected = list(product(*zip(target_lo, target_hi, strict=True)))
    if any(sum(math.dist(p[:2], q) < tolerance for p in points) != 2 for q in expected):
        raise ValueError("Door plan rectangle corners are ambiguous")
    perimeter = sum(math.dist(a[:2], b[:2]) for a, b in plan_segments)
    if abs(perimeter - 2 * (width + hi[1] - lo[1])) > 4 * tolerance:
        raise ValueError("Door plan segments do not form a rectangle")
    swings = []
    for hx, hy, sign in product((lo[0], hi[0]), (lo[1], hi[1]), (-1, 1)):
        moved = [(hx - sign * (vertices[i][1] - hy), hy + sign * (vertices[i][0] - hx), vertices[i][2]) for i in island]
        bounds = [min(p[k] for p in moved) for k in range(2)] + [max(p[k] for p in moved) for k in range(2)]
        if max(abs(a - b) for a, b in zip(bounds, target_lo + target_hi, strict=True)) < tolerance:
            swings.append((moved, hx, hy, sign))
    if len(swings) != 1:
        raise ValueError("Door hinge/swing is not uniquely supported by IFC plan geometry")
    moved, hx, hy, sign = swings[0]
    return dict(indices=island, vertices=moved, hinge=[hx, hy], angle_degrees=90 * sign,
                method="detached_cuboid_matching_IFC_dimensions_and_authored_plan")


def _ego_door_leaf_spec(ifc, entity, obj, scale):
    """Cross-check untouched Blender Body geometry against native IFC geometry."""
    import ifcopenshell.geom
    import numpy as np
    from ifcopenshell.util.placement import get_mappeditem_transformation
    from ifcopenshell.util.unit import calculate_unit_scale

    if (obj.type != "MESH" or obj.mode != "OBJECT" or obj.modifiers or obj.constraints or
            obj.animation_data or obj.data.shape_keys or obj.data.animation_data or obj.parent or
            obj.library or obj.data.library):
        raise ValueError(f"Door {entity.id()} has unsupported dynamic, linked, parented or modified geometry")
    matrix = np.array(obj.matrix_world)
    if (not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or
            not np.allclose(matrix[:3, 2], [0, 0, 1], atol=1e-6)):
        raise ValueError(f"Door {entity.id()} is not a rigid vertical object")
    unit = calculate_unit_scale(ifc)
    settings = ifcopenshell.geom.settings()
    shape = ifcopenshell.geom.create_shape(settings, entity)
    native = np.array(shape.geometry.verts).reshape((-1, 3))
    vertices = np.array([tuple(v.co) for v in obj.data.vertices]) * scale
    if len(native) != len(vertices) or any(np.min(np.linalg.norm(native - v, axis=1)) > 0.0002 for v in vertices):
        raise ValueError(f"Door {entity.id()} Blender mesh does not match native IFC Body")
    # Compare vertex multiplicities and coplanar surface patch signatures too:
    # equal point clouds alone do not rule out edited leaf/frame faces.
    from collections import Counter

    def point_key(point):
        return tuple(round(float(v), 5) for v in point)

    if Counter(map(point_key, native)) != Counter(map(point_key, vertices)):
        raise ValueError(f"Door {entity.id()} native vertex multiplicities differ")
    native_faces = np.array(shape.geometry.faces).reshape((-1, 3))
    obj.data.calc_loop_triangles()
    def surface_signature(points, faces):
        # Bonsai can dissolve native triangle diagonals into quads. Compare
        # coplanar patch boundaries and areas, independent of those diagonals.
        patches = {}
        for face in faces:
            triangle = points[list(face)]
            cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            area2 = float(np.linalg.norm(cross))
            if area2 < 1e-10:
                raise ValueError("Degenerate door Body face")
            normal = cross / area2
            if normal[int(np.argmax(np.abs(normal)))] < 0:
                normal = -normal
            key = point_key(normal) + (round(float(np.dot(normal, triangle[0])), 5),)
            boundary, areas = patches.setdefault(key, (Counter(), []))
            for a, b in ((0, 1), (1, 2), (2, 0)):
                boundary[tuple(sorted((point_key(triangle[a]), point_key(triangle[b]))))] += 1
            areas.append(area2 / 2)
        result = {}
        for key, (boundary, areas) in patches.items():
            if any(count > 2 for count in boundary.values()):
                raise ValueError("Nonmanifold coplanar door Body faces")
            result[key] = (tuple(sorted(edge for edge, count in boundary.items() if count == 1)), round(sum(areas), 5))
        return result

    if surface_signature(native, native_faces) != surface_signature(vertices, [tuple(face.vertices) for face in obj.data.loop_triangles]):
        raise ValueError(f"Door {entity.id()} Blender topology does not match native IFC Body")
    segments = []

    def visit(item, transform):
        if item.is_a("IfcMappedItem"):
            mapped = transform @ get_mappeditem_transformation(item)
            for child in item.MappingSource.MappedRepresentation.Items:
                visit(child, mapped)
        elif item.is_a("IfcGeometricSet"):
            for child in item.Elements:
                visit(child, transform)
        elif item.is_a("IfcPolyline"):
            points = []
            for point in item.Points:
                xyz = list(point.Coordinates) + [0.] * (3 - len(point.Coordinates))
                points.append(tuple((transform @ np.array(xyz + [1.]))[:3] * unit))
            segments.append(points)
        elif not item.is_a("IfcTrimmedCurve"):
            raise ValueError(f"Unsupported door Plan item: {item.is_a()}")

    for representation in entity.Representation.Representations:
        if representation.RepresentationIdentifier == "Plan":
            for item in representation.Items:
                visit(item, np.eye(4))
    spec = _ego_leaf_geometry(vertices.tolist(), [tuple(e.vertices) for e in obj.data.edges],
                              float(entity.OverallWidth) * unit, float(entity.OverallHeight) * unit, segments)
    spec["vertices"] = [[v / scale for v in point] for point in spec["vertices"]]
    spec["door_id"] = entity.id()
    return spec


def _ego_door_arc_bounds(vertices, hinge, angle_degrees, matrix, scale):
    """Exact world SI AABB of leaf vertex arcs over the authored quarter turn."""
    import math

    points = []
    low, high = sorted((0.0, math.radians(angle_degrees)))
    for vertex in vertices:
        dx, dy = vertex[0] - hinge[0], vertex[1] - hinge[1]
        center = [matrix[k][0] * hinge[0] + matrix[k][1] * hinge[1] + matrix[k][2] * vertex[2] + matrix[k][3] for k in range(3)]
        a = [matrix[k][0] * dx + matrix[k][1] * dy for k in range(3)]
        b = [-matrix[k][0] * dy + matrix[k][1] * dx for k in range(3)]
        angles = [low, high]
        for axis in range(3):
            critical = math.atan2(b[axis], a[axis])
            angles.extend(critical + n * math.pi for n in range(-2, 3) if low <= critical + n * math.pi <= high)
        points.extend([(center[k] + a[k] * math.cos(t) + b[k] * math.sin(t)) * scale for k in range(3)] for t in angles)
    return [[min(p[k] for p in points) for k in range(3)], [max(p[k] for p in points) for k in range(3)]]


def _ego_door_motion_record(obj, original, spec, scale):
    """Capture original geometry before the mesh pointer is changed."""
    original.calc_loop_triangles()
    vertices = [tuple(v.co) for v in original.vertices]
    world = [tuple(float(v) * scale for v in (obj.matrix_world @ vertex.co)) for vertex in original.vertices]
    leaf = [vertices[i] for i in spec["indices"]]
    hinge = [v / scale for v in spec["hinge"]]
    record = {key: value for key, value in spec.items() if key != "vertices"}
    leaf_indices = set(spec["indices"])
    hinge_z = min(vertex[2] for vertex in leaf)
    world_hinge = [(obj.matrix_world[k][0] * hinge[0] + obj.matrix_world[k][1] * hinge[1] +
                    obj.matrix_world[k][2] * hinge_z + obj.matrix_world[k][3]) * scale for k in range(3)]
    determinant = obj.matrix_world.to_3x3().determinant()
    record.update(_object=obj, _original_leaf_vertices=leaf, _hinge_local=hinge,
                  _closed_triangles=[tuple(world[i] for i in tri.vertices) for tri in original.loop_triangles],
                  _leaf_closed_triangles=[tuple(world[i] for i in tri.vertices) for tri in original.loop_triangles if set(tri.vertices) <= leaf_indices],
                  _frame_triangles=[tuple(world[i] for i in tri.vertices) for tri in original.loop_triangles if not set(tri.vertices) <= leaf_indices],
                  _world_hinge=world_hinge, _world_angle_sign=1 if determinant > 0 else -1,
                  door_sweep_bounds_m=_ego_door_arc_bounds(leaf, hinge, spec["angle_degrees"], obj.matrix_world, scale))
    return record


def _ego_door_mesh_variants(meshes, records):
    """Attach original closed geometry; triangles remain the actual open mesh."""
    for record in records:
        matches = [mesh for mesh in meshes if mesh.get("ifc_id") == record["door_id"]]
        if len(matches) != 1:
            raise ValueError(f"Staged door {record['door_id']} must have one evaluated mesh, without instances")
        matches[0].update(closed_triangles=record["_closed_triangles"],
                          door_open_angle_degrees=record["angle_degrees"],
                          door_sweep_bounds_m=record["door_sweep_bounds_m"],
                          door_hinge_world_m=record["_world_hinge"],
                          door_world_angle_sign=record["_world_angle_sign"],
                          door_leaf_closed_triangles=record["_leaf_closed_triangles"],
                          door_frame_triangles=record["_frame_triangles"],
                          door_angle_tolerance_degrees=1.0,
                          door_motion_validation={"method": "continuous_interval_support_prisms",
                                                  "linear_tolerance_m": 1e-6,
                                                  "unresolved_policy": "reject",
                                                  "baseline_penetration_policy": "reject"})
    return meshes


def _ego_door_pose_triangles(mesh, angle_degrees):
    """Exact rigid world-SI leaf triangles plus the untouched original frame."""
    import math

    limit = mesh["door_open_angle_degrees"]
    if not math.isfinite(angle_degrees) or not 0 <= angle_degrees / limit <= 1:
        raise ValueError("Door pose exceeds the modeled authored swing")
    angle = math.radians(angle_degrees * mesh["door_world_angle_sign"])
    c, s = math.cos(angle), math.sin(angle)
    hx, hy, _hz = mesh["door_hinge_world_m"]
    leaf = [tuple((hx + c * (p[0] - hx) - s * (p[1] - hy),
                   hy + s * (p[0] - hx) + c * (p[1] - hy), p[2]) for p in triangle)
            for triangle in mesh["door_leaf_closed_triangles"]]
    return list(mesh["door_frame_triangles"]) + leaf


def _ego_door_motion_validation(mesh, from_angle_degrees, angle_degrees, physical_meshes):
    """Strict continuous leaf motion check against the complete current scene.

    Existing support-material penetration is NOT waived. An invalid report can
    also mean an unresolved interval; it is never permission to force a pose.
    Frames and other leaves stay present. Semantic volumes alone are excluded.
    """
    target = _ego_door_pose_triangles(mesh, angle_degrees)
    initial = _ego_door_pose_triangles(mesh, from_angle_degrees)
    leaf_count = len(mesh["door_leaf_closed_triangles"])
    if not leaf_count:
        return {"valid": False, "blocking_door_ids": [], "reason": "missing_leaf_geometry", "blockers": []}
    leaf = mesh["door_leaf_closed_triangles"]
    hinge = mesh["door_hinge_world_m"]
    sign = mesh["door_world_angle_sign"]
    sweep_low, sweep_high = mesh["door_sweep_bounds_m"]

    def box(triangles):
        return ([min(point[k] for tri in triangles for point in tri) for k in range(3)],
                [max(point[k] for tri in triangles for point in tri) for k in range(3)])

    def overlaps(low, high):
        return all(low[k] <= sweep_high[k] + 1e-6 and high[k] >= sweep_low[k] - 1e-6 for k in range(3))

    def centroid(triangles):
        points = {tuple(point) for triangle in triangles for point in triangle}
        return tuple(sum(point[k] for point in points) / len(points) for k in range(3))

    centers = (centroid(initial[-leaf_count:]), centroid(target[-leaf_count:]))
    failures = []
    obstacles = [dict(triangles=mesh["door_frame_triangles"], ifc_id=mesh.get("ifc_id"), ifc_class="IfcDoorFrame")]
    obstacles.extend(other for other in physical_meshes
                     if other.get("ifc_id") != mesh.get("ifc_id")
                     and other.get("ifc_class") not in {"IfcSpace", "IfcOpeningElement", "IfcVirtualElement"})
    for obstacle in obstacles:
        triangles = obstacle.get("triangles", [])
        if not triangles:
            continue
        low, high = box(triangles)
        if not overlaps(low, high):
            continue
        # Crossings alone miss a leaf wholly enclosed by a closed solid. An
        # ambiguous/open mesh is treated conservatively by this parity check.
        contained = any(all(low[k] < center[k] < high[k] for k in range(3)) and
                        _ego_point_inside_triangles(center, triangles) for center in centers)
        nearby = [triangle for triangle in triangles if overlaps(*box([triangle]))]
        if contained or (nearby and not _ego_leaf_motion_clear(leaf, hinge, from_angle_degrees * sign,
                                                                angle_degrees * sign, nearby)):
            failures.append({"ifc_id": obstacle.get("ifc_id"), "ifc_class": obstacle.get("ifc_class"),
                             "reason": "solid_containment" if contained else "penetration_or_unresolved_continuous_motion"})
    blockers = sorted({failure["ifc_id"] for failure in failures
                       if failure["ifc_class"] == "IfcDoor" and failure["ifc_id"] is not None})
    structural = any(failure["ifc_class"] != "IfcDoor" for failure in failures)
    return {"valid": not failures, "blocking_door_ids": blockers,
            "reason": "validated_continuous_motion" if not failures else
                      ("retained_structure_or_unresolved_motion" if structural else "other_door_motion_blocker"),
            "blockers": failures, "method": "continuous_interval_support_prisms", "linear_tolerance_m": 1e-6,
            "baseline_penetration_policy": "reject", "unresolved_policy": "reject"}


def _ego_door_pose_is_valid(mesh, angle_degrees, physical_meshes, from_angle_degrees=0.0):
    return _ego_door_motion_validation(mesh, from_angle_degrees, angle_degrees, physical_meshes)["valid"]


def _ego_door_angles_at_frame(records, actions, frame):
    angles = {record["door_id"]: 0.0 for record in records}
    for action in actions:
        if frame < action["start_frame"]:
            break
        t = min(1.0, (frame - action["start_frame"]) / (action["end_frame"] - action["start_frame"]))
        angles[action["door_id"]] = action["from_angle_degrees"] + t * (action["to_angle_degrees"] - action["from_angle_degrees"])
    return angles


def _ego_validate_door_schedule(records, navigation, frame_count, fps):
    """Reject ambiguous action endpoints before touching frame handlers."""
    import math

    expected = {record["door_id"]: record["angle_degrees"] for record in records}
    states = {identifier: 0.0 for identifier in expected}
    actions = navigation.get("door_actions", [])
    previous_end = -1
    for action in actions:
        identifier, start, end = action["door_id"], action["start_frame"], action["end_frame"]
        if identifier not in expected or type(start) is not int or type(end) is not int:
            raise ValueError("Invalid staged door action identifier/frame")
        if not 0 <= start < end < frame_count or end - start < fps or start < previous_end:
            raise ValueError("Door actions must be sequential stationary holds lasting at least one second")
        before, after = action["from_angle_degrees"], action["to_angle_degrees"]
        if not math.isfinite(before) or not math.isfinite(after) or before != states[identifier]:
            raise ValueError("Door action does not continue the preceding exact pose")
        limit = expected[identifier]
        if before == after or not (0 <= before / limit <= 1 and 0 <= after / limit <= 1):
            raise ValueError("Door action exceeds the modeled authored swing interval")
        states[identifier], previous_end = after, end
    if {action["door_id"] for action in actions} != set(expected):
        raise ValueError("Door actions must cover exactly the selected traversed doors")
    frames = navigation.get("frame_door_angles", [])
    if expected and len(frames) != frame_count:
        raise ValueError("Door frame states must cover every camera frame")
    for index, angles in enumerate(frames):
        calculated = _ego_door_angles_at_frame(records, actions, index)
        if set(angles) != {str(key) for key in calculated} or any(
                not math.isfinite(angles[str(key)]) or abs(angles[str(key)] - value) > 1e-9
                for key, value in calculated.items()):
            raise ValueError("Door actions disagree with per-frame rigid angles")
    return actions


def _ego_apply_door_angles(records, angles):
    """Rigid leaf pose from immutable original coordinates, never accumulation."""
    import math

    for record in records:
        angle = math.radians(angles[record["door_id"]])
        c, s = math.cos(angle), math.sin(angle)
        hx, hy = record["_hinge_local"]
        mesh = record["_object"].data
        for index, vertex in zip(record["indices"], record["_original_leaf_vertices"], strict=True):
            dx, dy = vertex[0] - hx, vertex[1] - hy
            mesh.vertices[index].co = (hx + c * dx - s * dy, hy + s * dx + c * dy, vertex[2])
        mesh.update()


def _ego_door_frame_handler(records, actions, frame_count, errors):
    def update(scene, _depsgraph=None):
        try:
            frame = scene.frame_current - 1 + scene.frame_subframe
            if not 0 <= frame <= frame_count - 1:
                raise ValueError("Door animation frame lies outside the validated schedule")
            _ego_apply_door_angles(records, _ego_door_angles_at_frame(records, actions, frame))
        except Exception as exc:
            errors.append(exc)
            raise
    return update


@contextlib.contextmanager
def _ego_temporary_open_doors(scene, ifc, selected, edges, scale):
    """Open only door edges actually used by the selected semantic route.

    Original datablocks, transforms, parents, constraints and visibility are never
    mutated. The outer finally covers partial setup, planning and render failures.
    IFC entities are read-only. Frames and opened leaves remain collision meshes.
    """
    route = selected["route_space_ids"]
    pairs = {frozenset((a, b)) for a, b in zip(route, route[1:], strict=False) if a != b}
    for pair in pairs:
        alternatives = [edge for edge in edges if frozenset((edge["source"], edge["target"])) == pair]
        if len(alternatives) != 1 and any(edge.get("kind") == "door" for edge in alternatives):
            raise ValueError("Ambiguous door alternatives require an explicit selected connector")
    ids = sorted({edge["connector_id"] for edge in edges if edge.get("kind") == "door"
                  and frozenset((edge["source"], edge["target"])) in pairs})
    changes = []
    records = []
    try:
        for identifier in ids:
            entity = ifc.by_id(identifier)
            if not entity.is_a("IfcDoor"):
                raise ValueError(f"Selected door edge {identifier} is not IfcDoor")
            objects = [o for o in scene.objects if (_element_for_object(o) is not None and _element_for_object(o).id() == identifier)]
            if len(objects) != 1:
                raise ValueError(f"Door {identifier} requires exactly one Blender mesh object")
            obj = objects[0]
            spec = _ego_door_leaf_spec(ifc, entity, obj, scale)
            original = obj.data
            motion = _ego_door_motion_record(obj, original, spec, scale)
            temporary = original.copy()
            changes.append((obj, original, temporary))
            obj.data = temporary
            for index, point in zip(spec["indices"], spec["vertices"], strict=True):
                temporary.vertices[index].co = point
            temporary.update()
            records.append(motion)
        bpy.context.view_layer.update()
        yield records
    finally:
        errors = []
        for obj, original, temporary in reversed(changes):
            try:
                obj.data = original
            except Exception as exc:
                errors.append(exc)
            try:
                if temporary.users == 0:
                    bpy.data.meshes.remove(temporary)
            except Exception as exc:
                errors.append(exc)
        bpy.context.view_layer.update()
        if errors:
            raise RuntimeError("Could not fully restore temporary door mesh state") from errors[0]


