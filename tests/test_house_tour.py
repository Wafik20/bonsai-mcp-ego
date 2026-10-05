"""House-tour topology tests without Blender or an IfcOpenShell dependency."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def planner():
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    tree = ast.parse(source.read_text())
    names = {
        "_house_tour",
        "_house_tour_classifications",
        "_house_tour_door_opening_evidence",
        "_h_plan_house_tour",
        "_house_tour_virtual_portals",
        "_house_tour_stair_endpoints",
        "_house_tour_triangle_overlap_area_xy",
    }
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and (n.name in names or n.name.startswith("_house_tour_"))
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class Entity:
    def __init__(self, eid, kind, **attrs):
        self.eid, self.kind = eid, kind
        self.__dict__.update(attrs)

    def id(self):
        return self.eid

    def is_a(self, kind=None):
        return self.kind if kind is None else self.kind == kind


class IFC:
    def __init__(self, *entities):
        self.entities = list(entities)

    def by_type(self, kind):
        return [e for e in self.entities if e.is_a(kind)]


def boundary(eid, space, element):
    return Entity(eid, "IfcRelSpaceBoundary", RelatingSpace=space, RelatedBuildingElement=element)


def rooms(count):
    return [
        Entity(i, "IfcSpace", Name=f"Bedroom {i}", LongName=f"Space {i}")
        for i in range(1, count + 1)
    ]


def door(ifc, eid, *spaces):
    d = Entity(eid, "IfcDoor")
    ifc.entities.extend([d] + [boundary(eid * 10 + i, s, d) for i, s in enumerate(spaces)])
    return d


def test_every_space_components_no_teleport(planner):
    a, b, c, d, e, f = rooms(6)
    model = IFC(a, b, c, d, e, f)
    door(model, 20, a, b)
    door(model, 21, b, c)
    door(model, 22, d, e)
    result = planner["_house_tour"](model)
    assert [n["id"] for n in result["nodes"]] == [1, 2, 3, 4, 5, 6]
    assert result["components"] == [[1, 2, 3], [4, 5], [6]]
    assert result["unreachable_space_ids"] == []
    assert result["isolated_space_ids"] == [6]
    assert result["coverage"] == 1
    assert result["route_space_ids"] == []
    assert [t["route_space_ids"] for t in result["component_tours"]] == [[1, 2, 3], [4, 5], [6]]
    assert result["coverage_complete"] is False
    assert "#6 Bedroom 6 / Space 6" in result["graph_text"]
    assert result["whole_model_coverage"] == 1
    assert result["all_spaces_visited"] is True
    explicit = planner["_house_tour"](model, 4)
    assert explicit["route_space_ids"] == []
    assert explicit["reachable_from_requested_start_space_ids"] == [4, 5]
    assert explicit["unreachable_from_requested_start_space_ids"] == [1, 2, 3, 6]
    assert explicit["component_tours"][1]["route_space_ids"] == [4, 5]


def test_seeded_shortest_path_with_revisits(planner):
    a, b, c, d = rooms(4)
    model = IFC(a, b, c, d)
    for eid, end in enumerate([b, c, d], 20):
        door(model, eid, a, end)
    result = planner["_house_tour"](model, seed=41)
    assert result == planner["_house_tour"](model, seed=41)
    route = result["route_space_ids"]
    assert len(route) == 6 and route.count(1) == 3
    edges = {frozenset((e["source"], e["target"])) for e in result["edges"]}
    assert all(frozenset(pair) in edges for pair in zip(route, route[1:], strict=False))
    assert result["visited_space_ids"] == [1, 2, 3, 4]
    # Input ordering must not affect the seed.
    model.entities.reverse()
    assert result == planner["_house_tour"](model, seed=41)


def test_wall_not_a_portal_ambiguous_door_not_clique(planner):
    a, b, c = rooms(3)
    wall = Entity(9, "IfcWall")
    model = IFC(a, b, c, wall, boundary(10, a, wall), boundary(11, b, wall))
    door(model, 20, a, b, c)
    result = planner["_house_tour"](model)
    assert result["edges"] == []
    assert result["unresolved_connectors"][0]["reason"] == "ambiguous_multiple_spaces"


@pytest.mark.parametrize("filling_kind,expected", [("IfcDoor", 1), ("IfcWindow", 0), (None, 1)])
def test_opening_fills_and_voids(planner, filling_kind, expected):
    a, b = rooms(2)
    opening, wall = Entity(9, "IfcOpeningElement"), Entity(8, "IfcWall")
    model = IFC(
        a,
        b,
        opening,
        wall,
        boundary(10, a, opening),
        boundary(11, b, opening),
        Entity(
            12, "IfcRelVoidsElement", RelatedOpeningElement=opening, RelatingBuildingElement=wall
        ),
    )
    if filling_kind:
        filling = Entity(20, filling_kind)
        model.entities.extend(
            [
                filling,
                Entity(
                    13,
                    "IfcRelFillsElement",
                    RelatingOpeningElement=opening,
                    RelatedBuildingElement=filling,
                ),
            ]
        )
    result = planner["_house_tour"](model)
    assert len(result["edges"]) == expected
    if expected:
        assert 12 in result["edges"][0]["evidence"]["relationship_ids"]
        assert result["edges"][0]["passability"] == "not_verified"


def test_mixed_door_opening_boundaries(planner):
    a, b = rooms(2)
    opening, d = Entity(9, "IfcOpeningElement"), Entity(20, "IfcDoor")
    model = IFC(
        a,
        b,
        opening,
        d,
        boundary(10, a, opening),
        boundary(11, b, d),
        Entity(13, "IfcRelFillsElement", RelatingOpeningElement=opening, RelatedBuildingElement=d),
    )
    result = planner["_house_tour"](model)
    assert len(result["edges"]) == 1
    assert result["edges"][0]["evidence"]["connector_ids"] == [9, 20]


def test_stair_flight_and_storey_hierarchy(planner):
    a, b, c = rooms(3)
    low, high = (
        Entity(100, "IfcBuildingStorey", Name="Ground"),
        Entity(101, "IfcBuildingStorey", Name="Upper"),
    )
    stair, flight = Entity(30, "IfcStair"), Entity(31, "IfcStairFlight")
    model = IFC(
        a,
        b,
        c,
        low,
        high,
        stair,
        flight,
        Entity(110, "IfcRelAggregates", RelatingObject=low, RelatedObjects=[a]),
        Entity(111, "IfcRelAggregates", RelatingObject=high, RelatedObjects=[b, c]),
        Entity(112, "IfcRelAggregates", RelatingObject=stair, RelatedObjects=[flight]),
        boundary(120, a, flight),
        boundary(121, b, flight),
    )
    result = planner["_house_tour"](model)
    assert result["nodes"][0]["storey_name"] == "Ground"
    assert result["nodes"][1]["storey_id"] == 101
    assert result["unreachable_space_ids"] == []
    assert all(e["kind"] == "stair" for e in result["edges"])
    assert any(112 in e["evidence"]["relationship_ids"] for e in result["edges"])


def test_stair_on_storey_does_not_link_all_rooms(planner):
    a, b = rooms(2)
    stair, storey = Entity(30, "IfcStair"), Entity(100, "IfcBuildingStorey")
    model = IFC(
        a,
        b,
        stair,
        storey,
        Entity(110, "IfcRelAggregates", RelatingObject=storey, RelatedObjects=[a, b]),
        Entity(
            111,
            "IfcRelContainedInSpatialStructure",
            RelatingStructure=storey,
            RelatedElements=[stair],
        ),
    )
    result = planner["_house_tour"](model)
    assert result["edges"] == []
    assert result["unresolved_connectors"][0]["connector_id"] == 30


@pytest.mark.parametrize(
    "start,seed",
    [
        (True, 0),
        (0, 0),
        (1.2, 0),
        (None, True),
        (None, "2"),
        (None, -1),
        (None, 2147483648),
        (999, 0),
    ],
)
def test_invalid_inputs(planner, start, seed):
    with pytest.raises(ValueError):
        planner["_house_tour"](IFC(*rooms(1)), start, seed)


def test_empty_missing_and_handler_readonly(planner):
    with pytest.raises(RuntimeError, match="No IFC"):
        planner["_house_tour"](None)
    with pytest.raises(ValueError, match="no IfcSpace"):
        planner["_house_tour"](IFC())
    model = IFC(*rooms(1))
    planner["_get_loaded_ifc"] = lambda: model
    assert planner["_h_plan_house_tour"]({})["coverage"] == 1
    source = Path(__file__).resolve().parents[1] / "blender_addon" / "bonsai_bridge.py"
    text = source.read_text()
    assert '"plan_house_tour": _h_plan_house_tour' in text
    edits = text[text.index("_EDIT_COMMANDS") : text.index("_ACTIVITY_LOG_SIZE")]
    assert "plan_house_tour" not in edits


def test_virtual_portal_evidence_integration_and_ambiguity(planner):
    a, b, c = rooms(3)
    boundaries = [boundary(100, a, None), boundary(101, b, None), boundary(102, c, None)]
    for item in boundaries:
        item.PhysicalOrVirtualBoundary = "VIRTUAL"
    storey = Entity(200, "IfcBuildingStorey")
    model = IFC(
        a,
        b,
        c,
        *boundaries,
        storey,
        Entity(201, "IfcRelAggregates", RelatingObject=storey, RelatedObjects=[a, b, c]),
    )
    first = {
        "space_ids": [1, 2],
        "boundary_ids": [100, 101],
        "kind": "paired_virtual_boundary_polygon_overlap",
        "width_m": 1.0,
        "height_m": 2.0,
    }
    planner["_house_tour_virtual_portals"] = lambda ifc: [first]
    result = planner["_house_tour"](model)
    assert result["edges"][0]["kind"] == "opening"
    assert result["reachable_space_ids"] == [1, 2, 3]
    assert result["unresolved_connectors"][0]["connector_id"] == 102
    second = dict(first, space_ids=[1, 3], boundary_ids=[100, 102])
    planner["_house_tour_virtual_portals"] = lambda ifc: [first, second]
    result = planner["_house_tour"](model)
    assert result["edges"] == []
    assert len(result["unresolved_connectors"]) == 3


def test_virtual_geometry_unavailable_is_disclosed(planner):
    a = rooms(1)[0]
    rel = boundary(100, a, None)
    rel.PhysicalOrVirtualBoundary = "VIRTUAL"

    def unavailable(ifc):
        raise ImportError("geometry dependency unavailable")

    planner["_house_tour_virtual_portals"] = unavailable
    result = planner["_house_tour"](IFC(a, rel))
    assert result["unresolved_connectors"][0]["detail"] == "geometry dependency unavailable"
    assert result["tour"][0]["space_id"] == 1
    assert result["tour"][0]["name"] == "Bedroom 1"


def test_stair_geometry_evidence_and_failure_diagnostics(planner):
    a, b = rooms(2)
    flight = Entity(30, "IfcStairFlight")
    model = IFC(a, b, flight)
    portal = {
        "flight_id": 30,
        "space_ids": [1, 2],
        "kind": "stair_tread_space_floor_polygon_overlap",
        "endpoints": [],
    }
    planner["_house_tour_stair_endpoints"] = lambda ifc, **kwargs: ([portal], [])
    result = planner["_house_tour"](model)
    assert result["reachable_space_ids"] == [1, 2]
    assert result["unresolved_connectors"] == []
    assert result["edges"][0]["evidence"]["max_endpoint_step_m"] == 0.25
    planner["_house_tour_stair_endpoints"] = lambda ifc, **kwargs: (
        [],
        [{"flight_id": 30, "reason": "unsupported_stair_geometry"}],
    )
    result = planner["_house_tour"](model)
    assert result["edges"] == []
    assert result["geometry_diagnostics"][0]["flight_id"] == 30
    assert result["unresolved_connectors"][0]["connector_id"] == 30


def test_virtual_duplicate_patches_merge_and_different_storeys_rejected(planner):
    a, b = rooms(2)
    low, high = Entity(200, "IfcBuildingStorey"), Entity(201, "IfcBuildingStorey")
    aggregation = Entity(202, "IfcRelAggregates", RelatingObject=low, RelatedObjects=[a, b])
    model = IFC(a, b, low, high, aggregation)
    first = {
        "space_ids": [1, 2],
        "boundary_ids": [100, 101],
        "kind": "paired_virtual_boundary_polygon_overlap",
        "width_m": 1.0,
        "height_m": 2.0,
    }
    second = dict(first, boundary_ids=[102, 103])
    planner["_house_tour_virtual_portals"] = lambda ifc: [first, second]
    result = planner["_house_tour"](model)
    assert len(result["edges"]) == 1
    assert len(result["edges"][0]["evidence"]["patches"]) == 2
    assert result["edges"][0]["evidence"]["relationship_ids"] == [100, 101, 102, 103]
    aggregation.RelatedObjects = [a]
    model.entities.append(Entity(203, "IfcRelAggregates", RelatingObject=high, RelatedObjects=[b]))
    assert planner["_house_tour"](model)["edges"] == []


def properties(model, space, **values):
    props = [
        Entity(9000 + i, "IfcPropertySingleValue", Name=name, NominalValue=value)
        for i, (name, value) in enumerate(values.items())
    ]
    pset = Entity(9100, "IfcPropertySet", Name="Pset_SpaceCommon", HasProperties=props)
    model.entities.append(
        Entity(
            9200 + space.id(),
            "IfcRelDefinesByProperties",
            RelatedObjects=[space],
            RelatingPropertyDefinition=pset,
        )
    )


def test_metadata_precedence_and_conflicts(planner):
    a, b, c, d = rooms(4)
    a.LongName = "Roof"
    a.InteriorOrExteriorSpace = "INTERNAL"
    b.LongName = "Room"
    c.InteriorOrExteriorSpace = "EXTERNAL"
    model = IFC(a, b, c, d)
    properties(model, a, OccupancyType="Bedroom")
    properties(model, b, **{"Category Description": "Stairway"})
    properties(model, d, IsExternal=True, OccupancyType="Bedroom")
    result = planner["_house_tour"](model)
    assert [s["id"] for s in result["tourable_spaces"]] == [1]
    assert [s["id"] for s in result["circulation_spaces"]] == [2]
    assert [s["id"] for s in result["excluded_spaces"]] == [3]
    assert result["unresolved_spaces"][0]["reason"] == "conflicting_ifc_metadata"
    assert result["classification_complete"] is False
    assert result["classification_completeness"] == 0.75
    assert all(s["evidence"] for s in result["spaces"])


def test_unknown_is_not_a_shortcut_or_silently_removed(planner):
    a, middle, b = rooms(3)
    middle.Name, middle.LongName = "X002", "Unspecified"
    model = IFC(a, middle, b)
    door(model, 20, a, middle)
    door(model, 21, middle, b)
    result = planner["_house_tour"](model)
    assert len(result["raw_edges"]) == 2
    assert result["edges"] == []
    assert result["raw_components"] == [[1, 2, 3]]
    assert result["components"] == [[1], [3]]
    assert result["target_space_ids"] == [1, 3]
    assert result["covered_target_space_ids"] == [1, 3]
    assert result["coverage"] == result["tourable_room_coverage"] == 1
    assert result["coverage_complete"] is False
    assert result["graph_validation_status"] == "unresolved"
    assert result["unresolved_spaces"][0]["reason"]
    assert result["route_space_ids"] == result["tour"] == []
    with pytest.raises(ValueError, match="classified interior"):
        planner["_house_tour"](model, 2)


def test_circulation_coverage_and_dead_end_pruning(planner):
    a, hall, b, unused = rooms(4)
    hall.Name = hall.LongName = "Corridor"
    unused.Name = unused.LongName = "Stairway"
    model = IFC(a, hall, b, unused)
    door(model, 20, a, hall)
    door(model, 21, hall, b)
    door(model, 22, hall, unused)
    result = planner["_house_tour"](model)
    assert result["target_space_ids"] == [1, 2, 3]
    assert result["tourable_room_coverage"] == 1
    assert result["covered_target_space_ids"] == [1, 2, 3]
    assert 4 not in result["route_space_ids"]
    assert planner["_house_tour"](model, 4)["target_space_ids"] == [1, 2, 3, 4]


def test_multiple_component_start_changes_only_local_start(planner):
    a, b, c, d, e = rooms(5)
    model = IFC(a, b, c, d, e)
    door(model, 20, a, b)
    door(model, 21, c, d)
    result = planner["_house_tour"](model, 4)
    assert [t["route_space_ids"] for t in result["tours"]] == [[1, 2], [4, 3], [5]]
    assert result["target_space_ids"] == result["covered_target_space_ids"] == [1, 2, 3, 4, 5]
    assert result["tours"][-1]["graph_validation_status"] == "unresolved_missing_connection"
    assert result["legacy_flat_tour_available"] is False
    edges = {frozenset((edge["source"], edge["target"])) for edge in result["edges"]}
    assert all(
        frozenset(pair) in edges
        for tour in result["tours"]
        for pair in zip(tour["route_space_ids"], tour["route_space_ids"][1:], strict=False)
    )


def test_roof_name_alone_does_not_exclude(planner):
    roof = Entity(1, "IfcSpace", Name="R301", LongName="Roof")
    result = planner["_house_tour"](IFC(roof))
    assert result["excluded_spaces"] == []
    assert result["unresolved_spaces"][0]["id"] == 1
    assert result["coverage"] is None
    assert result["coverage_complete"] is False


def service_roof():
    roof = Entity(1, "IfcSpace", Name="R301", LongName="Roof", InteriorOrExteriorSpace="INTERNAL")
    storey = Entity(10, "IfcBuildingStorey", Name="Roof")
    wall = Entity(20, "IfcWall")
    rel = boundary(30, roof, wall)
    rel.InternalOrExternalBoundary = "EXTERNAL"
    model = IFC(
        roof,
        storey,
        wall,
        rel,
        Entity(40, "IfcRelAggregates", RelatingObject=storey, RelatedObjects=[roof]),
    )
    properties(model, roof, **{"Category Description": "Other General Facility Service Spaces"})
    return model, roof


def test_service_roof_policy_and_occupied_override(planner):
    model, roof = service_roof()
    result = planner["_house_tour"](model)
    excluded = result["excluded_spaces"][0]
    assert excluded["policy_exclusion"] is True
    assert excluded["confidence"] == "policy_exclusion_not_physical_accessibility"
    assert "not_proven_inaccessible" in excluded["reason"]
    properties(model, roof, IsAccessible=True)
    result = planner["_house_tour"](model)
    assert result["excluded_spaces"] == []
    assert result["tourable_spaces"][0]["id"] == 1


def test_service_roof_portal_conflict_stays_unknown(planner):
    model, roof = service_roof()
    door(model, 50, roof)
    result = planner["_house_tour"](model)
    assert result["excluded_spaces"] == []
    assert result["unresolved_spaces"][0]["reason"] == "service_space_usage_requires_review"


def test_ambiguous_door_geometry_resolution_retains_original_candidates(planner):
    a, b, c = rooms(3)
    model = IFC(a, b, c)
    door(model, 20, a, b, c)
    planner["_house_tour_door_opening_evidence"] = lambda *args: (
        {"space_ids": [1, 2], "kind": "door_opening_host_wall_space_volume_sides"},
        [],
    )
    result = planner["_house_tour"](model)
    assert len(result["edges"]) == 1
    assert result["edges"][0]["evidence"]["original_candidate_space_ids"] == [1, 2, 3]
    planner["_house_tour_door_opening_evidence"] = lambda *args: (None, [{"reason": "ambiguous"}])
    result = planner["_house_tour"](model)
    assert result["edges"] == []
    assert (
        result["unresolved_connectors"][0]["evidence"]["geometry_diagnostics"][0]["reason"]
        == "ambiguous"
    )


def test_distinct_buildings_separation_needs_explicit_evidence(planner):
    a, b, c, d = rooms(4)
    one, two = Entity(100, "IfcBuilding"), Entity(101, "IfcBuilding")
    model = IFC(
        a,
        b,
        c,
        d,
        one,
        two,
        Entity(110, "IfcRelAggregates", RelatingObject=one, RelatedObjects=[a, b]),
        Entity(111, "IfcRelAggregates", RelatingObject=two, RelatedObjects=[c, d]),
    )
    door(model, 20, a, b)
    door(model, 21, c, d)
    result = planner["_house_tour"](model)
    assert all(t["graph_validation_status"] == "physically_separate" for t in result["tours"])
    assert result["graph_validation_status"] == "validated_topology"
    model.entities.remove(one)
    model.entities = [e for e in model.entities if e.id() != 110]
    result = planner["_house_tour"](model)
    assert all(
        t["graph_validation_status"] == "unresolved_missing_connection" for t in result["tours"]
    )


@pytest.mark.parametrize(
    "failure", [None, "same_zones", "hosted_opening", "missing_partition", "same_wall_face"]
)
def test_physical_unit_separation_requires_all_evidence(planner, monkeypatch, failure):
    import sys
    import types

    a, b, c, d = rooms(4)
    wall = Entity(20, "IfcWall", Name="Party Wall", ObjectType=None, HasOpenings=[])
    rows = [boundary(100 + s.id(), s, wall) for s in (a, b, c, d)]
    for row in rows:
        row.PhysicalOrVirtualBoundary = "PHYSICAL"
    if failure == "hosted_opening":
        opening = Entity(60, "IfcOpeningElement", HasFillings=[])
        wall.HasOpenings = [Entity(61, "IfcRelVoidsElement", RelatedOpeningElement=opening)]
    if failure == "missing_partition":
        wall.Name = "Facade"
    entities = {entity.id(): entity for entity in (a, b, c, d, wall)}
    model = IFC(*entities.values(), *rows)
    model.by_id = entities.__getitem__
    model.get_inverse = lambda entity: rows if entity == wall else []
    for s in (a, b, c, d, wall):
        s.Representation = types.SimpleNamespace(
            Representations=[types.SimpleNamespace(RepresentationIdentifier="Body")]
        )
    for s in (a, b, c, d):
        external = boundary(200 + s.id(), s, Entity(30 if s.id() < 3 else 31, "IfcDoor"))
        external.InternalOrExternalBoundary = "EXTERNAL"
        s.BoundedBy = [external]

    def get_psets(s):
        zone = "A" if s.id() < 3 or failure == "same_zones" else "B"
        return {"Pset": {"OccupancyZoneName": zone}}

    def shape(settings, entity, rep):
        bounds = (
            (0, 1)
            if entity == wall
            else (1, 2)
            if entity.id() < 3 or failure == "same_wall_face"
            else (-1, 0)
        )
        vertices = [
            coordinate for x in bounds for y in (0, 10) for z in (0, 3) for coordinate in (x, y, z)
        ]
        return types.SimpleNamespace(
            geometry=types.SimpleNamespace(verts=vertices, faces=[0, 1, 2, 4, 6, 7])
        )

    geom = types.ModuleType("ifcopenshell.geom")
    geom.settings = lambda: types.SimpleNamespace(USE_WORLD_COORDS=1, set=lambda *args: None)
    geom.create_shape = shape
    element = types.ModuleType("ifcopenshell.util.element")
    element.get_psets = get_psets
    util = types.ModuleType("ifcopenshell.util")
    util.element = element
    root = types.ModuleType("ifcopenshell")
    root.geom, root.util = geom, util
    for name, module in [
        ("ifcopenshell", root),
        ("ifcopenshell.geom", geom),
        ("ifcopenshell.util", util),
        ("ifcopenshell.util.element", element),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    result = planner["_house_tour_component_separation_evidence"](model, [[1, 2], [3, 4]])
    assert result["status"] == ("unresolved" if failure else "physically_separate")
    if failure is None:
        assert result["pairs"][0]["dividing_walls"][0]["wall_id"] == 20


@pytest.mark.parametrize("name,long_name", [(None, None), ("R001", "Room"), ("Room", "Space")])
def test_internal_alone_or_generic_room_does_not_prove_tourable_usage(planner, name, long_name):
    space = Entity(1, "IfcSpace", Name=name, LongName=long_name, InteriorOrExteriorSpace="INTERNAL")
    result = planner["_house_tour"](IFC(space))
    assert result["unresolved_spaces"][0]["id"] == 1
    assert result["tourable_spaces"] == []
    space.LongName = "Kitchen"
    assert planner["_house_tour"](IFC(space))["tourable_spaces"][0]["id"] == 1


def test_nonportal_and_redundant_virtual_audits_do_not_hide_coverage(planner):
    a, b = rooms(2)
    model = IFC(a, b)
    door(model, 20, a, b)
    virtual = boundary(100, a, None)
    virtual.PhysicalOrVirtualBoundary = "VIRTUAL"
    model.entities.append(virtual)
    result = planner["_house_tour"](model)
    assert result["coverage_complete"] is True
    assert result["blocking_unresolved_connectors"] == []
    assert result["connector_audit_notes"][0]["connector_id"] == 100
    assert result["unresolved_connectors"][0]["connector_id"] == 100
    assert "evidence" not in result["tour"][0]


def test_irrelevant_properties_not_repeated_as_classification_evidence(planner):
    a = rooms(1)[0]
    model = IFC(a)
    properties(model, a, Manufacturer="placeholder", Comments="ignored", OccupancyType="Bedroom")
    evidence = planner["_house_tour"](model)["spaces"][0]["evidence"]
    assert [item["field"] for item in evidence] == ["Pset_SpaceCommon.OccupancyType"]
