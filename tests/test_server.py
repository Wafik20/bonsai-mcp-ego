"""MCP server tests."""

from __future__ import annotations

import json

import pytest

from bonsai_mcp.schemas import (
    GetSceneInfoInput,
    ViewportScreenshotInput,
    input_schema_for,
)
from bonsai_mcp.server import (
    _PROMPTS,
    _dispatch_tool,
    _prompt_text,
    _read_resource_payload,
    _screenshot_to_mcp_content,
    _tool_definitions,
    build_server,
)
from bonsai_mcp.tools import ALL_TOOL_NAMES, EDIT_TOOL_NAMES, QUERY_TOOL_NAMES
from tests.conftest import FakeBlenderBridgeClient


def test_all_tools_have_definitions():
    defined = {t.name for t in _tool_definitions()}
    assert defined == set(ALL_TOOL_NAMES)


def test_definitions_have_required_fields():
    for tool in _tool_definitions():
        assert tool.name
        assert tool.title, f"{tool.name} is missing a display title"
        assert tool.description, f"{tool.name} is missing a description"
        assert isinstance(tool.inputSchema, dict)
        assert isinstance(tool.outputSchema, dict), f"{tool.name} is missing outputSchema"


def test_descriptions_have_category_prefix():
    for tool in _tool_definitions():
        if tool.name in QUERY_TOOL_NAMES:
            assert tool.description.startswith("[QUERY] "), tool.name
        else:
            assert tool.description.startswith("[EDIT] "), tool.name


def test_annotations_match_category():
    by_name = {t.name: t for t in _tool_definitions()}
    for name in QUERY_TOOL_NAMES:
        ann = by_name[name].annotations
        assert ann is not None and ann.readOnlyHint is True, name
        assert ann.idempotentHint is True, name
    for name in EDIT_TOOL_NAMES:
        ann = by_name[name].annotations
        assert ann is not None and ann.readOnlyHint is False, name
    for name, tool in by_name.items():
        assert tool.annotations.openWorldHint is False, name
    # code execution is not idempotent; saving the same file twice is
    assert by_name["execute_ifc_code"].annotations.idempotentHint is False
    assert by_name["execute_blender_code"].annotations.idempotentHint is False
    assert by_name["save_ifc_file"].annotations.idempotentHint is True


class TestSchemaSingleSourcing:
    def test_input_schemas_come_from_the_pydantic_models(self):
        by_name = {t.name: t for t in _tool_definitions()}
        assert by_name["get_scene_info"].inputSchema == input_schema_for(GetSceneInfoInput)
        assert by_name["get_viewport_screenshot"].inputSchema == input_schema_for(
            ViewportScreenshotInput
        )

    def test_generated_schema_shape(self):
        schema = input_schema_for(GetSceneInfoInput)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        serialized = json.dumps(schema)
        assert "$defs" not in serialized
        assert "$ref" not in serialized
        assert '"title"' not in serialized

    def test_every_schema_is_clean_json_schema(self):
        for tool in _tool_definitions():
            serialized = json.dumps(tool.inputSchema)
            assert "$ref" not in serialized, tool.name
            assert tool.inputSchema.get("type") == "object", tool.name

    def test_execute_code_description_matches_injected_namespace(self):
        # regression for the drift the roadmap found: the schema description
        # must not promise pre-bound modules the handler does not inject
        by_name = {t.name: t for t in _tool_definitions()}
        code_desc = by_name["execute_blender_code"].inputSchema["properties"]["code"][
            "description"
        ]
        assert "bpy" in code_desc
        assert "not pre-bound" in code_desc


def test_build_server_does_not_touch_bridge():
    client = FakeBlenderBridgeClient()
    server = build_server(client)
    assert server is not None
    assert client.calls == []


_DISPATCH_CASES = {
    "generate_ego_video": ({"output_path": "/tmp/walk.mp4"}, "generate_ego_video"),
    "get_scene_info": ({}, "get_scene_info"),
    "get_selected_objects": ({}, "get_selected_objects"),
    "list_elements": ({}, "list_elements"),
    "get_psets": ({"global_ids": ["AAAA"]}, "get_psets"),
    "get_viewport_screenshot": ({}, "get_viewport_screenshot"),
    "get_ifc_project_info": ({}, "get_ifc_project_info"),
    "get_spatial_structure": ({}, "get_spatial_structure"),
    "get_quantities": ({}, "get_quantities"),
    "plan_house_tour": ({}, "plan_house_tour"),
    "execute_ifc_code": ({"code": "print(1)"}, "execute_ifc_code"),
    "execute_blender_code": ({"code": "print(1)"}, "execute_code"),
    "save_ifc_file": ({"output_path": "/tmp/out.ifc"}, "save_ifc_file"),
    "refresh_view": ({"global_ids": ["AAAA"]}, "refresh_view"),
    "refresh_geometry": ({"global_ids": ["AAAA"]}, "refresh_geometry"),
    "reload_project": ({}, "reload_project"),
}


def test_dispatch_cases_cover_every_tool():
    assert set(_DISPATCH_CASES) == set(ALL_TOOL_NAMES)


@pytest.mark.parametrize("tool_name", list(_DISPATCH_CASES))
def test_dispatch_routes_tool_to_its_bridge_command(tool_name):
    def on_send(cmd, _params):
        if cmd == "get_viewport_screenshot":
            return {"path": "/tmp/v.png", "image_base64": "AAAA", "format": "png"}
        if cmd == "get_selected_objects":
            return {"objects": [], "total": 0, "truncated": False}
        return {"ok": True}

    arguments, expected_cmd = _DISPATCH_CASES[tool_name]
    client = FakeBlenderBridgeClient(on_send=on_send)

    content, structured = _dispatch_tool(client, tool_name, arguments)
    assert content, tool_name
    assert isinstance(structured, dict), tool_name
    assert client.calls[-1][0] == expected_cmd


def test_dispatch_returns_payload_as_structured_content():
    client = FakeBlenderBridgeClient(
        responses={"get_ifc_project_info": {"schema": "IFC4", "entity_counts": {}}}
    )
    content, structured = _dispatch_tool(client, "get_ifc_project_info", {})
    assert structured == {"schema": "IFC4", "entity_counts": {}}
    assert content[0].type == "text"
    assert json.loads(content[0].text) == structured


def test_dispatch_screenshot_structured_content_omits_pixels_and_path():
    payload = {
        "path": "/tmp/v.png",
        "image_base64": "AAAA",
        "format": "png",
        "width": 8,
        "height": 6,
        "viewport": {"view_perspective": "ORTHO"},
    }
    client = FakeBlenderBridgeClient(responses={"get_viewport_screenshot": payload})
    content, structured = _dispatch_tool(client, "get_viewport_screenshot", {})
    assert "image_base64" not in structured
    assert "path" not in structured
    assert structured["viewport"] == {"view_perspective": "ORTHO"}
    assert any(c.type == "image" for c in content)


def test_dispatch_unknown_tool_raises():
    client = FakeBlenderBridgeClient()
    with pytest.raises(ValueError, match="Unknown tool:"):
        _dispatch_tool(client, "does_not_exist", {})
    assert client.calls == []


def test_screenshot_content_success_returns_image_and_note():
    import base64

    b64 = base64.b64encode(b"\x89PNG not-a-real-image").decode("ascii")
    out = _screenshot_to_mcp_content(
        {
            "path": "/tmp/v.png",
            "image_base64": b64,
            "format": "png",
            "width": 800,
            "height": 600,
        }
    )
    image = next(c for c in out if c.type == "image")
    text = next(c for c in out if c.type == "text")
    assert image.mimeType == "image/png"
    assert image.data == b64
    assert out[0].type == "image", "image block must come first for client compatibility"
    assert f"{len(b64)} base64 chars" in text.text
    assert "800x600 px" in text.text
    # the bridge-side temp path is dead tokens for the client and is omitted
    assert "/tmp/v.png" not in text.text


def test_screenshot_content_includes_viewport_state_as_text():
    import base64

    b64 = base64.b64encode(b"fake").decode("ascii")
    viewport = {
        "view_rotation": [0.7071, 0.7071, 0.0, 0.0],
        "view_perspective": "ORTHO",
        "objects_in_view": [
            {
                "name": "Wall",
                "ifc_class": "IfcWall",
                "global_id": "XYZ",
                "box": [0.1, 0.2, 0.5, 0.9],
                "depth": 12.5,
            }
        ],
    }
    out = _screenshot_to_mcp_content(
        {"path": "/tmp/v.jpg", "image_base64": b64, "format": "jpeg", "viewport": viewport}
    )
    text = next(c for c in out if c.type == "text")
    assert "Viewport state:" in text.text
    assert '"view_perspective": "ORTHO"' in text.text
    assert '"global_id": "XYZ"' in text.text
    assert '"depth": 12.5' in text.text


def test_screenshot_content_surfaces_downgrade_note_and_legend():
    import base64

    b64 = base64.b64encode(b"fake").decode("ascii")
    out = _screenshot_to_mcp_content(
        {
            "image_base64": b64,
            "format": "jpeg",
            "note": "png was auto-downgraded to jpeg",
            "class_legend": {"IfcWall": [0.12, 0.47, 0.71]},
        }
    )
    text = next(c for c in out if c.type == "text")
    assert "auto-downgraded" in text.text
    assert "Class colors" in text.text
    assert "IfcWall" in text.text


def test_screenshot_content_error_raises_tool_error():
    from bonsai_mcp.tools import ToolError

    with pytest.raises(ToolError, match="no viewport"):
        _screenshot_to_mcp_content({"error": "no viewport"})


def test_screenshot_content_invalid_base64_raises_tool_error():
    from bonsai_mcp.tools import ToolError

    with pytest.raises(ToolError, match="decode failed"):
        _screenshot_to_mcp_content({"image_base64": "!!! not base64 !!!"})


def test_screenshot_content_empty_payload_raises_tool_error():
    from bonsai_mcp.tools import ToolError

    with pytest.raises(ToolError, match="no payload"):
        _screenshot_to_mcp_content({})


class TestResources:
    def test_project_resource_routes_to_project_info(self):
        client = FakeBlenderBridgeClient(
            responses={"get_ifc_project_info": {"schema": "IFC4"}}
        )
        payload = _read_resource_payload(client, "bonsai://project")
        assert payload == {"schema": "IFC4"}
        assert client.calls[-1][0] == "get_ifc_project_info"

    def test_trailing_slash_is_tolerated(self):
        # pydantic URL normalization may append a slash
        client = FakeBlenderBridgeClient(
            responses={"get_ifc_project_info": {"schema": "IFC4"}}
        )
        payload = _read_resource_payload(client, "bonsai://project/")
        assert payload == {"schema": "IFC4"}

    def test_scene_resource_routes_to_scene_info(self):
        client = FakeBlenderBridgeClient(
            responses={"get_scene_info": {"scene_name": "S"}}
        )
        payload = _read_resource_payload(client, "bonsai://scene")
        assert payload["scene_name"] == "S"
        assert client.calls[-1] == ("get_scene_info", {})

    def test_element_psets_template(self):
        client = FakeBlenderBridgeClient(responses={"get_psets": {"results": []}})
        _read_resource_payload(client, "bonsai://element/2O2Fr$t4X7Zf8NOew3FLKr/psets")
        cmd, params = client.calls[-1]
        assert cmd == "get_psets"
        assert params["global_ids"] == ["2O2Fr$t4X7Zf8NOew3FLKr"]

    def test_unknown_uri_raises(self):
        client = FakeBlenderBridgeClient()
        with pytest.raises(ValueError, match="Unknown resource URI"):
            _read_resource_payload(client, "bonsai://nonsense")
        assert client.calls == []


class TestPrompts:
    def test_prompt_names(self):
        assert {p.name for p in _PROMPTS} == {"model-audit", "visual-verify"}

    def test_model_audit_mentions_the_query_workflow(self):
        text = _prompt_text("model-audit", None)
        assert "get_spatial_structure" in text
        assert "get_quantities" in text
        assert "get_psets" in text

    def test_model_audit_focus_is_injected(self):
        text = _prompt_text("model-audit", {"focus": "fire safety"})
        assert "fire safety" in text

    def test_visual_verify_mentions_reload_and_screenshots(self):
        text = _prompt_text("visual-verify", {"what_changed": "moved door D1"})
        assert "moved door D1" in text
        assert "save_ifc_file" in text
        assert "get_viewport_screenshot" in text

    def test_unknown_prompt_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            _prompt_text("nope", None)


def test_server_instructions_describe_categories():
    from bonsai_mcp.server import _SERVER_INSTRUCTIONS

    assert "[QUERY]" in _SERVER_INSTRUCTIONS
    assert "[EDIT]" in _SERVER_INSTRUCTIONS
    assert "execute_ifc_code" in _SERVER_INSTRUCTIONS
    assert "get_spatial_structure" in _SERVER_INSTRUCTIONS
    assert "get_quantities" in _SERVER_INSTRUCTIONS
    assert "list_elements" in _SERVER_INSTRUCTIONS


def test_generate_ego_video_structured_output_and_edit_registration():
    from bonsai_mcp.schemas import GenerateEgoVideoInput

    payload = {"success": True, "video_path": "walk.mp4", "frame_count": 300, "navigation": {"seed": 0}}
    client = FakeBlenderBridgeClient(responses={"generate_ego_video": payload})
    content, structured = _dispatch_tool(client, "generate_ego_video", {"output_path": "walk.mp4"})
    assert structured == payload
    assert json.loads(content[0].text) == payload
    tool = next(t for t in _tool_definitions() if t.name == "generate_ego_video")
    assert tool.inputSchema == input_schema_for(GenerateEgoVideoInput)
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.idempotentHint is False
    assert "generate_ego_video" in EDIT_TOOL_NAMES


def test_generate_ego_video_tour_contract():
    definitions = {tool.name: tool for tool in _tool_definitions()}
    video = definitions["generate_ego_video"]
    component = video.inputSchema["properties"]["component_id"]
    assert component["default"] is None
    assert {"minimum": 0, "type": "integer"} in component["anyOf"]
    assert "component_id" not in video.inputSchema["required"]
    assert "minimum" in video.inputSchema["properties"]["duration_seconds"]["description"].lower()
    assert "not a route truncation limit" in video.description
    properties = video.outputSchema["properties"]
    for field in ("tour", "visited_spaces"):
        assert properties[field]["type"] == "array"
        assert properties[field]["items"] == {"type": "integer"}
    for field in ("tourable_rooms_visited", "tourable_rooms_total", "component_id"):
        assert properties[field] == {"type": "integer", "minimum": 0}
    assert properties["coverage"] == {"type": "number", "minimum": 0, "maximum": 1}
    assert properties["route_length_m"] == {"type": "number", "minimum": 0}
    assert "deterministically" in video.description
    payload = {
        "component_id": 0, "requested_duration_seconds": 1.0,
        "duration_seconds": 20.0, "frame_count": 200,
        "tour": [42, 84, 42], "visited_spaces": [42, 84],
        "tourable_rooms_visited": 2, "tourable_rooms_total": 2,
        "coverage": 1.0, "route_length_m": 24.5,
        "navigation": {"route_space_ids": [42, 84, 42]},
    }
    client = FakeBlenderBridgeClient(responses={"generate_ego_video": payload})
    content, structured = _dispatch_tool(client, "generate_ego_video", {
        "output_path": "walk.mp4", "component_id": 0, "duration_seconds": 1.0,
    })
    assert structured == payload
    assert json.loads(content[0].text) == payload
    assert client.calls[0][1]["component_id"] == 0
    assert client.calls[0][1]["duration_seconds"] == 1.0


def test_plan_house_tour_read_only_schema_and_structured_output():
    from bonsai_mcp.schemas import PlanHouseTourInput

    definition = next(t for t in _tool_definitions() if t.name == "plan_house_tour")
    assert definition.inputSchema == input_schema_for(PlanHouseTourInput)
    assert definition.annotations.readOnlyHint is True
    assert definition.annotations.destructiveHint is False
    assert definition.annotations.idempotentHint is True
    assert "plan_house_tour" in QUERY_TOOL_NAMES
    assert "plan_house_tour" not in EDIT_TOOL_NAMES
    payload = {
        "spaces": [{"id": 42}, {"id": 84}], "connections": [], "tour": [],
        "tourable_spaces": [{"id": 42}, {"id": 84}], "circulation_spaces": [],
        "excluded_spaces": [], "unresolved_spaces": [],
        "tours": [
            {"component_id": "component_1", "tour": [{"space_id": 42}], "coverage": 1.0},
            {"component_id": "component_2", "tour": [{"space_id": 84}], "coverage": 1.0},
        ],
        "visited_space_ids": [42, 84], "reachable_space_ids": [42, 84],
        "unreachable_space_ids": [], "coverage": 1.0,
        "classification_complete": True, "graph_validation_status": "unresolved",
        "coverage_complete": False,
        "debug": {"adjacency_text": "42: isolated; 84: isolated"},
    }
    client = FakeBlenderBridgeClient(responses={"plan_house_tour": payload})
    content, structured = _dispatch_tool(client, "plan_house_tour", {"start_space_id": 42})
    assert structured == payload
    assert json.loads(content[0].text) == payload
    assert client.calls == [("plan_house_tour", {"start_space_id": 42, "seed": 0})]
