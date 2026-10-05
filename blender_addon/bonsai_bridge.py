"""Local Blender bridge for Bonsai MCP."""

from __future__ import annotations

import base64
import collections
import contextlib
import errno
import hmac
import io
import json
import os
import queue
import re as _re
import select
import shutil
import socket
import socketserver
import struct
import threading
import time
import traceback

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel

bl_info = {
    "name": "Bonsai MCP Bridge",
    "author": "Show2Instruct",
    "version": (1, 2, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Bonsai MCP",
    "description": "Localhost TCP bridge for the bonsai-mcp server.",
    "category": "Development",
}

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9878
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
# How long a connection handler waits for the main thread before cancelling
# the queued request and reporting a timeout to the client.
MAIN_THREAD_WAIT_SECONDS = 120.0
# Commands that modify state; rejected when the "Allow edits" preference is off.
_EDIT_COMMANDS = frozenset(
    {
        "execute_code",
        "execute_ifc_code",
        "save_ifc_file",
        "refresh_view",
        "refresh_geometry",
        "reload_project",
        "generate_ego_video",
    }
)
_ACTIVITY_LOG_SIZE = 5
_STATE: dict[str, object] = {
    "server": None,
    "thread": None,
    "request_queue": queue.Queue(),
    "timer_registered": False,
    "bound": None,
    "last_error": None,
    "requests_served": 0,
    "last_command": None,
    "activity": collections.deque(maxlen=_ACTIVITY_LOG_SIZE),
    # snapshot of the token preference, taken on the main thread when the
    # bridge starts so handler threads never touch bpy preferences
    "token": "",
}


class _ProtocolError(ValueError):
    """A malformed inbound frame; the peer gets an error reply, then we close."""

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _read_message(sock: socket.socket) -> dict | None:
    """Read one framed request. None means the peer closed the connection.

    Raises _ProtocolError for malformed input (oversized frame, non-JSON
    body) so the handler can send an error reply instead of silently
    dropping the connection.
    """
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length > MAX_MESSAGE_BYTES:
        raise _ProtocolError(
            f"frame of {length} bytes exceeds the {MAX_MESSAGE_BYTES} byte cap"
        )
    body = _recv_exact(sock, length)
    if body is None:
        return None
    try:
        message = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ProtocolError(f"frame body is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise _ProtocolError(
            f"frame body must be a JSON object, got {type(message).__name__}"
        )
    return message


def _send_message(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body)

def _try_import_ifcopenshell():
    try:
        import ifcopenshell  # type: ignore
        return ifcopenshell
    except ImportError:
        return None


def _get_bonsai_tool():
    """Return bonsai.tool (or the legacy blenderbim.tool), or None."""
    with contextlib.suppress(ImportError):
        import bonsai.tool as tool_mod  # type: ignore

        return tool_mod
    with contextlib.suppress(ImportError):
        import blenderbim.tool as tool_mod  # type: ignore

        return tool_mod
    return None


# Memoizes _get_loaded_ifc for the duration of one bridge request. Without
# this, per-object lookups while filtering can re-resolve (and on the
# fallback path re-open from disk!) the IFC file thousands of times.
_IFC_CACHE: dict[str, object] = {"valid": False, "ifc": None}


def _invalidate_ifc_cache() -> None:
    _IFC_CACHE["valid"] = False
    _IFC_CACHE["ifc"] = None


def _get_loaded_ifc():
    """Return the loaded IFC file, if available. Memoized per bridge request."""
    if _IFC_CACHE["valid"]:
        return _IFC_CACHE["ifc"]
    ifc = _resolve_loaded_ifc()
    _IFC_CACHE["valid"] = True
    _IFC_CACHE["ifc"] = ifc
    return ifc


def _resolve_loaded_ifc():
    """Resolve the loaded IFC file from Bonsai, or best-effort from disk."""
    tool_mod = _get_bonsai_tool()
    if tool_mod is not None:
        try:
            ifc = tool_mod.Ifc.get()
            if ifc is not None:
                return ifc
        except Exception:
            pass

    try:
        ifc_path = getattr(bpy.context.scene.BIMProperties, "ifc_file", "")  # type: ignore[attr-defined]
        if ifc_path and os.path.isfile(ifc_path):
            ifcopenshell = _try_import_ifcopenshell()
            if ifcopenshell is not None:
                return ifcopenshell.open(ifc_path)
    except Exception:
        pass

    return None


def _bonsai_project_path() -> str | None:
    """Return the file path of the currently loaded IFC project, if known."""
    import importlib

    for mod_name in ("bonsai.bim.ifc", "blenderbim.bim.ifc"):
        with contextlib.suppress(Exception):
            path = importlib.import_module(mod_name).IfcStore.path
            if path:
                return path
    return None


def _save_ifc_project(path: str) -> str:
    """Write the in-memory IFC model to `path`; return the method used.

    Prefers Bonsai's IfcExporter, which first syncs pending Blender-side
    edits into the IFC model. Falls back to a plain ifcopenshell write.
    """
    import importlib
    import logging

    for pkg in ("bonsai", "blenderbim"):
        try:
            export_ifc = importlib.import_module(f"{pkg}.bim.export_ifc")
        except ImportError:
            continue
        settings = export_ifc.IfcExportSettings.factory(
            bpy.context, path, logging.getLogger("BonsaiExport")
        )
        export_ifc.IfcExporter(settings).export()
        return f"{pkg}.bim.export_ifc.IfcExporter"

    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError("No IFC project is loaded; cannot save.")
    ifc.write(path)
    return "ifcopenshell.write"


def _reload_ifc_project(path: str) -> None:
    """Clear the scene and reload the project so the viewport matches the IFC."""
    tool_mod = _get_bonsai_tool()
    if tool_mod is None:
        raise RuntimeError("Bonsai is not available; cannot reload the project.")
    # Bonsai's loader purges IfcStore and can lazily re-resolve the project
    # from the scene's ifc_file property; unless that points at the target
    # first, the reload silently re-imports the previous file.
    with contextlib.suppress(Exception):
        bpy.context.scene.BIMProperties.ifc_file = path  # type: ignore[attr-defined]
    tool_mod.IfcGit.load_project(path)
    _invalidate_ifc_cache()


def _bridge_get_ifc_file():
    """Return the loaded IFC file or raise. Injected as get_ifc_file()."""
    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError("No IFC file open. Load a project in Bonsai first.")
    return ifc


def _bridge_get_default_container():
    """Return the active spatial container. Injected as get_default_container()."""
    tool_mod = _get_bonsai_tool()
    if tool_mod is None:
        raise RuntimeError("Bonsai (bonsai.tool) is not available.")
    container = tool_mod.Root.get_default_container()
    if not container:
        raise RuntimeError("No active spatial container.")
    return container


def _bridge_save_and_load_ifc(path: str | None = None) -> str:
    """Save the project (to its own file by default) and reload it.

    Reloading is what makes IFC-level edits visible in the Blender viewport.
    Returns the saved path. Injected as save_and_load_ifc().
    """
    target = path or _bonsai_project_path()
    if not target:
        raise RuntimeError(
            "The project has no file path yet. Pass an explicit path, e.g. "
            "save_and_load_ifc(r'C:/path/model.ifc')."
        )
    _save_ifc_project(target)
    _reload_ifc_project(target)
    return target


def _element_for_object(obj):
    """Return the IFC entity behind a Blender object, or None."""
    ifc = _get_loaded_ifc()
    if ifc is None:
        return None
    try:
        ifc_def_id = obj.BIMObjectProperties.ifc_definition_id  # type: ignore[attr-defined]
    except Exception:
        return None
    if not ifc_def_id:
        return None
    try:
        return ifc.by_id(ifc_def_id)
    except Exception:
        return None


def _name_prefix_class(obj) -> str | None:
    """Fallback class guess from the 'IfcClass/Name' object naming convention."""
    name = getattr(obj, "name", "") or ""
    if "/" in name:
        prefix = name.split("/", 1)[0]
        if prefix.startswith("Ifc"):
            return prefix
    return None


def _ifc_class_for_object(obj) -> str | None:
    """Best-effort IFC class lookup for a Blender object."""
    element = _element_for_object(obj)
    if element is not None:
        try:
            return element.is_a()
        except Exception:
            pass
    return _name_prefix_class(obj)


def _object_matches_class(obj, target: str) -> bool:
    """Inheritance-aware class test: IfcWallStandardCase matches IfcWall.

    Uses the boolean form element.is_a(target), which walks the schema
    inheritance chain (string equality on is_a() misses subtypes such as
    IfcWallStandardCase in IFC2X3 models). Falls back to an exact name-prefix
    match when no IFC entity is resolvable.
    """
    element = _element_for_object(obj)
    if element is not None:
        try:
            return bool(element.is_a(target))
        except Exception:
            return False
    return _name_prefix_class(obj) == target


def _global_id_for_object(obj) -> str | None:
    element = _element_for_object(obj)
    if element is None:
        return None
    try:
        return getattr(element, "GlobalId", None)
    except Exception:
        return None


def _object_summary(obj) -> dict:
    loc = getattr(obj, "location", None)
    dims = getattr(obj, "dimensions", None)
    return {
        "name": obj.name,
        "type": getattr(obj, "type", None),
        "location": [loc.x, loc.y, loc.z] if loc is not None else None,
        "dimensions": [dims.x, dims.y, dims.z] if dims is not None else None,
        "ifc_class": _ifc_class_for_object(obj),
        "global_id": _global_id_for_object(obj),
    }

def _h_ping(_params):
    return {
        "status": "ok",
        "service": "bonsai-mcp-bridge",
        "blender_version": bpy.app.version_string,
        "addon_version": ".".join(str(x) for x in bl_info["version"]),
        "ifcopenshell_available": _try_import_ifcopenshell() is not None,
        "ifc_loaded": _get_loaded_ifc() is not None,
        "edits_allowed": _edits_allowed(),
        "token_required": bool(_STATE.get("token")),
        "requests_served": _STATE.get("requests_served", 0),
    }


def _paginate(items: list, params: dict, default_limit: int = 200, max_limit: int = 1000):
    """Slice `items` by the request's limit/offset; return page plus metadata."""
    try:
        limit = int(params.get("limit") or default_limit)
    except (TypeError, ValueError):
        limit = default_limit
    limit = max(1, min(limit, max_limit))
    try:
        offset = max(0, int(params.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    page = items[offset : offset + limit]
    truncated = offset + len(page) < len(items)
    return page, len(items), truncated, limit, offset


_QUERY_KEYWORD_TO_CLASS = {
    "walls": "IfcWall",
    "doors": "IfcDoor",
    "windows": "IfcWindow",
    "spaces": "IfcSpace",
    "slabs": "IfcSlab",
    "columns": "IfcColumn",
    "beams": "IfcBeam",
    "roofs": "IfcRoof",
    "stairs": "IfcStair",
}


def _matching_objects(objects, params) -> list:
    """Apply the optional `query` filter and return the matching objects.

    Summaries are built later, for the requested page only, so a paged
    query on a huge scene does not pay per-object IFC lookups for every
    match.
    """
    query = (params.get("query") or "").strip().lower()
    if not query:
        return []

    if query == "all":
        return list(objects)

    if query == "selected":
        return list(bpy.context.selected_objects)

    if query in _QUERY_KEYWORD_TO_CLASS:
        target = _QUERY_KEYWORD_TO_CLASS[query]
        return [o for o in objects if _object_matches_class(o, target)]

    if query == "by_class":
        target = params.get("ifc_class")
        if not target:
            raise ValueError("query='by_class' requires 'ifc_class'")
        return [o for o in objects if _object_matches_class(o, target)]

    if query == "by_name":
        target = params.get("name")
        if not target:
            raise ValueError("query='by_name' requires 'name'")
        return [o for o in objects if o.name == target]

    if query == "by_global_id":
        target = params.get("global_id")
        if not target:
            raise ValueError("query='by_global_id' requires 'global_id'")
        return [o for o in objects if _global_id_for_object(o) == target]

    raise ValueError(f"Unknown query: {query!r}")


# Cap on the selected-object name list in the scene summary; a box-select can
# grab thousands of objects and the full list would bloat every summary frame.
_SCENE_SELECTED_NAMES_CAP = 50


def _h_get_scene_info(params):
    scene = bpy.context.scene
    objects = list(scene.objects)
    selected = [o.name for o in bpy.context.selected_objects]
    collections = [c.name for c in scene.collection.children_recursive] if hasattr(
        scene.collection, "children_recursive"
    ) else [c.name for c in scene.collection.children]
    object_type_counts: dict[str, int] = {}
    for obj in objects:
        object_type_counts[obj.type] = object_type_counts.get(obj.type, 0) + 1

    payload: dict = {
        "scene_name": scene.name,
        "object_count": len(objects),
        "selected_count": len(selected),
        "selected_objects": selected[:_SCENE_SELECTED_NAMES_CAP],
        "selected_objects_truncated": len(selected) > _SCENE_SELECTED_NAMES_CAP,
        "collections": collections,
        "object_type_counts": object_type_counts,
        "ifc_available": _get_loaded_ifc() is not None,
        "blender_version": bpy.app.version_string,
    }

    if params and (params.get("query") or "").strip():
        matches = _matching_objects(objects, params)
        page, total, truncated, limit, offset = _paginate(matches, params)
        payload["objects"] = [_object_summary(o) for o in page]
        payload["objects_total"] = total
        payload["objects_truncated"] = truncated
        payload["objects_offset"] = offset
        payload["objects_limit"] = limit

    return payload


def _h_get_selected_objects(params):
    selected = list(bpy.context.selected_objects)
    page, total, truncated, _limit, _offset = _paginate(selected, params or {})
    return {
        "objects": [_object_summary(o) for o in page],
        "total": total,
        "truncated": truncated,
    }


_EXEC_OUTPUT_CAP_BYTES = 256 * 1024


def _trim_output(text: str) -> tuple[str, bool, int]:
    """Trim captured execution output."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _EXEC_OUTPUT_CAP_BYTES:
        return text, False, len(encoded)
    cut = encoded[:_EXEC_OUTPUT_CAP_BYTES].decode("utf-8", errors="replace")
    return cut, True, len(encoded)


def _h_execute_code(params):
    code = params.get("code", "")
    if not code:
        raise ValueError("'code' is required")

    stdout = io.StringIO()
    stderr = io.StringIO()
    namespace = {
        "bpy": bpy,
        "__name__": "__bonsai_mcp_exec__",
        "get_ifc_file": _bridge_get_ifc_file,
        "get_default_container": _bridge_get_default_container,
        "save_and_load_ifc": _bridge_save_and_load_ifc,
    }
    success = True
    error: str | None = None
    tb: str | None = None
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "<bonsai-mcp>", "exec"), namespace)  # noqa: S102
    except (Exception, SystemExit) as exc:
        # SystemExit: generated code calling sys.exit() must not kill Blender
        success = False
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()

    stdout_text, stdout_truncated, stdout_bytes = _trim_output(stdout.getvalue())
    stderr_text, stderr_truncated, stderr_bytes = _trim_output(stderr.getvalue())

    return {
        "success": success,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "error": error,
        "traceback": tb,
    }


_BPY_PATTERN = _re.compile(
    r"(^|\s)(import\s+bpy|from\s+bpy\s|bpy\.)", _re.MULTILINE
)


def _h_execute_ifc_code(params):
    """Execute IFC code without direct bpy access."""
    code = params.get("code", "")
    if not code:
        raise ValueError("'code' is required")

    if _BPY_PATTERN.search(code):
        raise ValueError(
            "execute_ifc_code does not allow bpy access. "
            "Use execute_blender_code instead for Blender-specific operations. "
            "This tool is restricted to IfcOpenShell and Bonsai API operations."
        )

    ifcopenshell = _try_import_ifcopenshell()
    if ifcopenshell is None:
        raise RuntimeError(
            "IfcOpenShell is not available in this Blender environment. "
            "Install it via Bonsai or manually to use execute_ifc_code."
        )

    ifc = _get_loaded_ifc()

    ifc_util_element = None
    ifc_api = None
    with contextlib.suppress(ImportError):
        import ifcopenshell.util.element as ifc_util_element  # type: ignore
    with contextlib.suppress(ImportError):
        import ifcopenshell.api as ifc_api  # type: ignore

    bonsai_tool = None
    with contextlib.suppress(ImportError):
        import bonsai.tool as bonsai_tool  # type: ignore
    if bonsai_tool is None:
        with contextlib.suppress(ImportError):
            import blenderbim.tool as bonsai_tool  # type: ignore

    namespace = {
        "__name__": "__bonsai_mcp_ifc_exec__",
        "ifcopenshell": ifcopenshell,
        "ifc": ifc,
        "ifc_api": ifc_api,
        "element_util": ifc_util_element,
        "tool": bonsai_tool,
        "get_ifc_file": _bridge_get_ifc_file,
        "get_default_container": _bridge_get_default_container,
        "save_and_load_ifc": _bridge_save_and_load_ifc,
    }

    stdout = io.StringIO()
    stderr = io.StringIO()
    success = True
    error: str | None = None
    tb: str | None = None
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "<bonsai-mcp-ifc>", "exec"), namespace)  # noqa: S102
    except (Exception, SystemExit) as exc:
        # SystemExit: generated code calling sys.exit() must not kill Blender
        success = False
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()

    stdout_text, stdout_truncated, stdout_bytes = _trim_output(stdout.getvalue())
    stderr_text, stderr_truncated, stderr_bytes = _trim_output(stderr.getvalue())

    return {
        "success": success,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "error": error,
        "traceback": tb,
        "ifc_available": ifc is not None,
        "namespace_keys": [
            k for k in namespace if not k.startswith("_") and namespace[k] is not None
        ],
    }


_SCREENSHOT_MAX_B64_CHARS = 700 * 1024  # keep the MCP payload well under the 1 MB cap
# Rough png bytes-per-pixel estimate used to downgrade to jpeg BEFORE
# rendering; calibrated against measured viewport output, with headroom.
_PNG_BYTES_PER_PIXEL_ESTIMATE = 1.0
_SCREENSHOT_MIN_SIZE = 64
_SCREENSHOT_MAX_SIZE = 2048
_SCREENSHOT_DEFAULT_SIZE = 800

_VIEW_AXIS_MAP = {
    "top": "TOP",
    "bottom": "BOTTOM",
    "front": "FRONT",
    "back": "BACK",
    "left": "LEFT",
    "right": "RIGHT",
}
_VALID_VIEWS = (*_VIEW_AXIS_MAP, "iso", "camera")
_VALID_FITS = ("all", "selected")
_SHADING_MAP = {
    "wireframe": "WIREFRAME",
    "solid": "SOLID",
    "material": "MATERIAL",
    "rendered": "RENDERED",
    "class_colors": "SOLID",
}

# distinct flat colors for shading='class_colors' (one per IFC class)
_CLASS_COLOR_PALETTE = (
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
    (0.70, 0.87, 0.54),
    (1.00, 0.73, 0.47),
)


def _find_view3d():
    """Return (window, area, region) for the largest open 3D viewport, or Nones."""
    best = (None, None, None)
    best_size = -1
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            size = area.width * area.height
            if size > best_size:
                best = (window, area, region)
                best_size = size
    return best


def _visible_mesh_objects(selected_only: bool = False):
    objects = bpy.context.selected_objects if selected_only else bpy.context.view_layer.objects
    return [o for o in objects if o.type == "MESH" and o.visible_get()]


def _tighten_framing(area, region, selected_only: bool) -> None:
    """Direction-aware zoom: fit the content's projected 2D extent.

    view_all/view_selected frame the bounding sphere; on flat-ish buildings
    an elevation view then fills only ~30% of the frame. Ortho zoom scales
    linearly with view_distance, so one multiplicative correction against
    the measured screen coverage is exact there and a safe approximation in
    perspective (clamped to avoid near-clipping surprises).
    """
    rv3d = area.spaces.active.region_3d
    if rv3d.view_perspective == "CAMERA":
        return
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from mathutils import Vector

    xs: list[float] = []
    ys: list[float] = []
    for obj in _visible_mesh_objects(selected_only):
        matrix = obj.matrix_world
        for corner in obj.bound_box:
            point = location_3d_to_region_2d(region, rv3d, matrix @ Vector(corner))
            if point is not None:
                xs.append(point.x)
                ys.append(point.y)
    if not xs or region.width <= 0 or region.height <= 0:
        return
    # The capture renders at the scene render aspect, not the region's;
    # measure against the letterboxed intersection of both frames so
    # tightening can never crop content out of the render.
    render = bpy.context.scene.render
    render_x = max(1, int(render.resolution_x * render.resolution_percentage / 100))
    render_y = max(1, int(render.resolution_y * render.resolution_percentage / 100))
    render_aspect = render_x / render_y
    frame_w = min(float(region.width), region.height * render_aspect)
    frame_h = min(float(region.height), region.width / render_aspect)
    center_x = region.width / 2.0
    center_y = region.height / 2.0
    coverage_x = 2.0 * max(abs(x - center_x) for x in xs) / frame_w
    coverage_y = 2.0 * max(abs(y - center_y) for y in ys) / frame_h
    coverage = max(coverage_x, coverage_y)
    if coverage <= 0.0:
        return
    scale = coverage / 0.9  # leave a 10% margin
    if scale >= 1.0:
        return  # never zoom out: the fit operator already shows everything
    if rv3d.view_perspective != "ORTHO":
        scale = max(scale, 0.2)
    rv3d.view_distance *= max(scale, 0.05)


def _orient_viewport(
    view: str | None,
    fit: str | None,
    azimuth: float | None = None,
    elevation: float | None = None,
) -> None:
    """Aim the largest 3D viewport before capturing (persists after the call)."""
    window, area, region = _find_view3d()
    if window is None:
        raise RuntimeError(
            "No 3D viewport is open; 'view'/'fit' need a visible VIEW_3D area."
        )
    r3d = area.spaces.active.region_3d

    prefs_view = bpy.context.preferences.view
    original_smooth = prefs_view.smooth_view
    prefs_view.smooth_view = 0  # animated transitions would smear the capture
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            if view in _VIEW_AXIS_MAP:
                bpy.ops.view3d.view_axis(type=_VIEW_AXIS_MAP[view])
            elif view == "iso":
                import math

                from mathutils import Euler

                r3d.view_perspective = "PERSP"
                r3d.view_rotation = Euler(
                    (math.radians(60.0), 0.0, math.radians(45.0)), "XYZ"
                ).to_quaternion()
            elif view == "camera":
                if bpy.context.scene.camera is None:
                    raise RuntimeError("view='camera' requires a scene camera.")
                r3d.view_perspective = "CAMERA"
            elif view:
                raise ValueError(f"'view' must be one of {_VALID_VIEWS}, got {view!r}")
            elif azimuth is not None or elevation is not None:
                import math

                from mathutils import Euler

                # same convention as 'iso' (which is azimuth=45, elevation=30):
                # azimuth 0 = front, counter-clockwise seen from above;
                # elevation 0 = horizontal, 90 = bird's eye
                az = float(azimuth) if azimuth is not None else 0.0
                el = float(elevation) if elevation is not None else 30.0
                el = max(-90.0, min(90.0, el))
                r3d.view_perspective = "PERSP"
                r3d.view_rotation = Euler(
                    (math.radians(90.0 - el), 0.0, math.radians(az)), "XYZ"
                ).to_quaternion()

            if fit == "all":
                bpy.ops.view3d.view_all()
            elif fit == "selected":
                if not bpy.context.selected_objects:
                    raise ValueError("fit='selected' requires at least one selected object.")
                bpy.ops.view3d.view_selected()
            elif fit:
                raise ValueError(f"'fit' must be one of {_VALID_FITS}, got {fit!r}")
            if fit:
                with contextlib.suppress(Exception):
                    _tighten_framing(area, region, fit == "selected")
    finally:
        prefs_view.smooth_view = original_smooth


def _viewport_snapshot(include_objects: bool, max_objects: int) -> dict | None:
    """Structured viewport state, optionally with screen-space object boxes.

    Box format: [x_min, y_min, x_max, y_max], normalized 0-1, origin at the
    image's top-left. `depth` is the view-space distance to the object's
    bounding-box centre (model units), so near/far ordering is available
    without pixels. Boxes are approximate (full object extent, ignoring
    occlusion) but preserve relative spatial layout.
    """
    _window, area, region = _find_view3d()
    if area is None:
        return None
    rv3d = area.spaces.active.region_3d
    state: dict = {
        "view_rotation": [round(v, 4) for v in rv3d.view_rotation],
        "view_perspective": rv3d.view_perspective,
        "is_orthographic_side_view": rv3d.is_orthographic_side_view,
        "view_distance": round(rv3d.view_distance, 3),
        "view_location": [round(v, 3) for v in rv3d.view_location],
    }
    if not include_objects:
        return state

    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from mathutils import Vector

    view_matrix = rv3d.view_matrix
    # (box area, ifc class, entry) per candidate object
    candidates: list[tuple[float, str, dict]] = []
    for obj in bpy.context.view_layer.objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue
        matrix = obj.matrix_world
        corners = [matrix @ Vector(c) for c in obj.bound_box]
        pts = []
        for corner in corners:
            point = location_3d_to_region_2d(region, rv3d, corner)
            if point is not None:
                pts.append(point)
        if not pts:
            continue
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        if max(xs) < 0 or min(xs) > region.width or max(ys) < 0 or min(ys) > region.height:
            continue
        x0 = max(0.0, min(xs)) / region.width
        x1 = min(float(region.width), max(xs)) / region.width
        y0 = 1.0 - min(float(region.height), max(ys)) / region.height
        y1 = 1.0 - max(0.0, min(ys)) / region.height
        centre = sum(corners, Vector((0.0, 0.0, 0.0))) / 8.0
        depth = -(view_matrix @ centre).z  # positive = in front of the viewpoint
        ifc_class = _ifc_class_for_object(obj)
        candidates.append(
            (
                (x1 - x0) * (y1 - y0),
                ifc_class or "(unclassified)",
                {
                    "name": obj.name,
                    "ifc_class": ifc_class,
                    "global_id": _global_id_for_object(obj),
                    "box": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                    "depth": round(depth, 2),
                },
            )
        )

    # round-robin across IFC classes (largest box first within each class)
    # so a few huge slabs cannot crowd out doors and windows
    by_class: dict[str, list[tuple[float, str, dict]]] = {}
    for item in candidates:
        by_class.setdefault(item[1], []).append(item)
    for items in by_class.values():
        items.sort(key=lambda item: item[0], reverse=True)
    class_order = sorted(by_class, key=lambda cls: by_class[cls][0][0], reverse=True)
    picked: list[tuple[float, str, dict]] = []
    rank = 0
    while len(picked) < max_objects:
        advanced = False
        for cls in class_order:
            items = by_class[cls]
            if rank < len(items):
                picked.append(items[rank])
                advanced = True
                if len(picked) >= max_objects:
                    break
        if not advanced:
            break
        rank += 1
    picked.sort(key=lambda item: item[0], reverse=True)
    state["objects_in_view"] = [entry for _, _, entry in picked]
    state["objects_in_view_total"] = len(candidates)
    state["objects_truncated"] = len(candidates) > len(picked)
    return state


def _screenshot_output_path(ext: str) -> str:
    """Return the viewport screenshot path for the given extension."""
    tmp_dir = _STATE.get("screenshot_dir")
    if not isinstance(tmp_dir, str):
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="bonsai_mcp_")
        _STATE["screenshot_dir"] = tmp_dir
    return os.path.join(tmp_dir, f"viewport.{ext}")


def _apply_class_colors():
    """Assign one flat color per IFC class via object color.

    Returns (legend, restore) where legend maps class -> [r, g, b] and
    restore is a list of (object, original_color) pairs. Colors are
    assigned to classes in sorted order, so the same model always gets the
    same legend.
    """
    pairs = [
        (obj, _ifc_class_for_object(obj) or "(unclassified)")
        for obj in _visible_mesh_objects()
    ]
    classes = sorted({cls for _, cls in pairs})
    class_to_color = {
        cls: _CLASS_COLOR_PALETTE[i % len(_CLASS_COLOR_PALETTE)]
        for i, cls in enumerate(classes)
    }
    restore = []
    for obj, cls in pairs:
        color = class_to_color[cls]
        restore.append((obj, tuple(obj.color)))
        obj.color = (color[0], color[1], color[2], 1.0)
    legend = {
        cls: [round(c, 3) for c in color] for cls, color in class_to_color.items()
    }
    return legend, restore


def _isolate_storey(storey_key: str) -> list:
    """Hide everything not contained in the storey; return objects to unhide."""
    ids = _element_ids_in_storey(storey_key)
    hidden = []
    for obj in bpy.context.view_layer.objects:
        if obj.hide_get():
            continue
        element = _element_for_object(obj)
        keep = False
        if element is not None:
            with contextlib.suppress(Exception):
                keep = element.id() in ids
        if not keep:
            hidden.append(obj)
            obj.hide_set(True)
    return hidden


def _h_get_viewport_screenshot(params):
    params = params or {}
    try:
        max_size = int(params.get("max_size") or _SCREENSHOT_DEFAULT_SIZE)
        quality = int(params.get("quality") or 85)
    except (TypeError, ValueError):
        raise ValueError("'max_size' and 'quality' must be integers") from None
    max_size = max(_SCREENSHOT_MIN_SIZE, min(max_size, _SCREENSHOT_MAX_SIZE))
    quality = max(1, min(quality, 100))
    fmt = str(params.get("format") or "jpeg").lower()
    if fmt not in ("jpeg", "jpg", "png"):
        raise ValueError("'format' must be 'jpeg' or 'png'")
    is_jpeg = fmt != "png"

    include_objects = bool(params.get("include_objects", False))
    try:
        max_objects = int(params.get("max_objects") or 50)
    except (TypeError, ValueError):
        raise ValueError("'max_objects' must be an integer") from None
    max_objects = max(1, min(max_objects, 200))

    view = params.get("view") or None
    fit = params.get("fit") or None
    azimuth = params.get("azimuth")
    elevation = params.get("elevation")
    if azimuth is not None or elevation is not None:
        if view:
            raise ValueError("Pass either 'view' or 'azimuth'/'elevation', not both.")
        try:
            azimuth = float(azimuth) if azimuth is not None else None
            elevation = float(elevation) if elevation is not None else None
        except (TypeError, ValueError):
            raise ValueError("'azimuth' and 'elevation' must be numbers") from None
    storey_key = params.get("storey") or None
    shading = params.get("shading") or None
    if shading is not None and shading not in _SHADING_MAP:
        raise ValueError(
            f"'shading' must be one of {tuple(_SHADING_MAP)}, got {shading!r}"
        )
    show_overlays = bool(params.get("show_overlays", False))

    window, area, region = _find_view3d()
    if window is None:
        raise RuntimeError(
            "Screenshot requires an open 3D viewport, and this Blender "
            "session has no VIEW_3D area (background/headless mode, or all "
            "3D viewports are closed)."
        )
    space = area.spaces.active

    note_parts: list[str] = []
    class_legend = None
    color_restore: list = []
    hidden_for_storey: list = []
    original_shading_type = space.shading.type
    original_color_type = space.shading.color_type
    original_overlays = space.overlay.show_overlays

    try:
        if storey_key:
            hidden_for_storey = _isolate_storey(storey_key)

        if view or fit or azimuth is not None or elevation is not None:
            _orient_viewport(view, fit, azimuth, elevation)

        if shading is not None:
            space.shading.type = _SHADING_MAP[shading]
            if shading == "class_colors":
                space.shading.color_type = "OBJECT"
                class_legend, color_restore = _apply_class_colors()
        space.overlay.show_overlays = show_overlays

        scene = bpy.context.scene
        render = scene.render
        settings = render.image_settings
        original_filepath = render.filepath
        original_format = settings.file_format
        original_color_mode = settings.color_mode
        original_quality = settings.quality
        original_res = (
            render.resolution_x,
            render.resolution_y,
            render.resolution_percentage,
        )

        # render.opengl uses the scene render resolution; downscale only, never upscale
        effective_x = max(1, int(render.resolution_x * render.resolution_percentage / 100))
        effective_y = max(1, int(render.resolution_y * render.resolution_percentage / 100))
        scale = min(1.0, max_size / max(effective_x, effective_y))
        width = max(1, int(effective_x * scale))
        height = max(1, int(effective_y * scale))

        # pre-render size estimate: degrade before paying for the render
        # round trip instead of erroring after it
        if not is_jpeg:
            estimated_b64 = width * height * _PNG_BYTES_PER_PIXEL_ESTIMATE * 4 / 3
            if estimated_b64 > _SCREENSHOT_MAX_B64_CHARS:
                is_jpeg = True
                note_parts.append(
                    f"png at {width}x{height} was estimated to exceed the "
                    f"response size cap; auto-downgraded to jpeg (quality "
                    f"{quality}). Request a smaller max_size to get png."
                )

        out_path = _screenshot_output_path("jpg" if is_jpeg else "png")
        try:
            render.filepath = out_path
            render.resolution_x = width
            render.resolution_y = height
            render.resolution_percentage = 100
            if is_jpeg:
                settings.file_format = "JPEG"
                settings.color_mode = "RGB"
                settings.quality = quality
            else:
                settings.file_format = "PNG"
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.render.opengl(write_still=True)
            with open(out_path, "rb") as fh:
                image_bytes = fh.read()
        finally:
            render.filepath = original_filepath
            settings.file_format = original_format
            settings.color_mode = original_color_mode
            settings.quality = original_quality
            (
                render.resolution_x,
                render.resolution_y,
                render.resolution_percentage,
            ) = original_res

        encoded = base64.b64encode(image_bytes).decode("ascii")
        if len(encoded) > _SCREENSHOT_MAX_B64_CHARS:
            raise RuntimeError(
                f"Screenshot is still {len(image_bytes)} bytes at {width}x{height}, "
                "too large for an MCP response. Retry with a smaller max_size "
                "and/or format='jpeg'."
            )

        viewport = _viewport_snapshot(include_objects, max_objects)
        if viewport is not None:
            if azimuth is not None:
                viewport["azimuth"] = azimuth
            if elevation is not None:
                viewport["elevation"] = elevation
            if storey_key:
                viewport["storey"] = storey_key

        payload = {
            "path": out_path,
            "image_base64": encoded,
            "base64_chars": len(encoded),
            "format": "jpeg" if is_jpeg else "png",
            "width": width,
            "height": height,
            "bytes": len(image_bytes),
            "view": view,
            "fit": fit,
            "viewport": viewport,
        }
        if class_legend:
            payload["class_legend"] = class_legend
        if note_parts:
            payload["note"] = " ".join(note_parts)
        return payload
    finally:
        with contextlib.suppress(Exception):
            space.shading.type = original_shading_type
        with contextlib.suppress(Exception):
            space.shading.color_type = original_color_type
        with contextlib.suppress(Exception):
            space.overlay.show_overlays = original_overlays
        for obj, color in color_restore:
            with contextlib.suppress(Exception):
                obj.color = color
        for obj in hidden_for_storey:
            with contextlib.suppress(Exception):
                obj.hide_set(False)

_IFC_SUMMARY_NAME_LIMIT = 100


def _summarise_materials(ifc) -> dict:
    """Return a small summary of IfcMaterial entities in the file."""
    try:
        materials = list(ifc.by_type("IfcMaterial"))
    except Exception:
        return {"count": 0, "names": [], "truncated": False}

    names: list[str] = []
    for mat in materials:
        name = getattr(mat, "Name", None)
        if name:
            names.append(name)
    unique_sorted = sorted(set(names))
    truncated = len(unique_sorted) > _IFC_SUMMARY_NAME_LIMIT
    return {
        "count": len(materials),
        "names": unique_sorted[:_IFC_SUMMARY_NAME_LIMIT],
        "truncated": truncated,
    }


def _summarise_classifications(ifc) -> dict:
    """Return a small summary of IfcClassification systems in the file."""
    try:
        classifications = list(ifc.by_type("IfcClassification"))
    except Exception:
        return {"count": 0, "systems": [], "truncated": False}

    systems: list[dict] = []
    for cls in classifications:
        systems.append(
            {
                "name": getattr(cls, "Name", None),
                "source": getattr(cls, "Source", None),
                "edition": getattr(cls, "Edition", None),
            }
        )
    truncated = len(systems) > _IFC_SUMMARY_NAME_LIMIT
    return {
        "count": len(classifications),
        "systems": systems[:_IFC_SUMMARY_NAME_LIMIT],
        "truncated": truncated,
    }


def _h_get_ifc_project_info(_params):
    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )

    project = next(iter(ifc.by_type("IfcProject")), None)
    entity_counts: dict[str, int] = {}
    for ifc_class in (
        "IfcSite",
        "IfcBuilding",
        "IfcBuildingStorey",
        "IfcWall",
        "IfcDoor",
        "IfcWindow",
        "IfcSlab",
        "IfcSpace",
        "IfcColumn",
        "IfcBeam",
        "IfcRoof",
        "IfcStair",
    ):
        try:
            entity_counts[ifc_class] = len(ifc.by_type(ifc_class))
        except Exception:
            entity_counts[ifc_class] = 0

    return {
        "schema": getattr(ifc, "schema", None),
        "project_name": getattr(project, "Name", None) if project else None,
        "project_global_id": getattr(project, "GlobalId", None) if project else None,
        "entity_counts": entity_counts,
        "materials": _summarise_materials(ifc),
        "classifications": _summarise_classifications(ifc),
    }


def _object_names_by_definition_id() -> dict[int, str]:
    """Map ifc_definition_id -> Blender object name, built in one scene pass."""
    mapping: dict[int, str] = {}
    for obj in bpy.context.scene.objects:
        try:
            def_id = obj.BIMObjectProperties.ifc_definition_id  # type: ignore[attr-defined]
        except Exception:
            continue
        if def_id and def_id not in mapping:
            mapping[def_id] = obj.name
    return mapping


def _find_ifc_element(ifc, name: str | None, global_id: str | None, id_to_name=None):
    """Resolve an IFC element and its Blender object name."""
    if global_id:
        try:
            element = ifc.by_guid(global_id)
        except Exception:
            element = None
        if element is not None:
            if id_to_name is None:
                id_to_name = _object_names_by_definition_id()
            try:
                blender_name = id_to_name.get(element.id())
            except Exception:
                blender_name = None
            return element, blender_name

    if name:
        obj = bpy.context.scene.objects.get(name)
        if obj is None:
            return None, None
        ifc_def_id = None
        try:
            ifc_def_id = obj.BIMObjectProperties.ifc_definition_id  # type: ignore[attr-defined]
        except Exception:
            ifc_def_id = None
        if not ifc_def_id:
            return None, obj.name
        try:
            return ifc.by_id(ifc_def_id), obj.name
        except Exception:
            return None, obj.name

    return None, None


def _h_get_psets(params):
    """Return property and quantity sets for IFC objects."""
    global_ids = list(params.get("global_ids") or [])
    names = list(params.get("names") or [])
    if not global_ids and not names:
        raise ValueError("Provide at least one entry in 'global_ids' or 'names'.")

    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )

    try:
        from ifcopenshell.util.element import (  # type: ignore
            get_psets as _get_psets,
        )
    except Exception as exc:
        raise RuntimeError(
            f"IfcOpenShell util.element is unavailable: {exc}"
        ) from exc

    requests: list[tuple[dict, str | None, str | None]] = []
    for gid in global_ids:
        requests.append(({"global_id": gid}, None, gid))
    for name in names:
        requests.append(({"name": name}, name, None))

    # one scene pass shared by every GlobalId lookup in this batch
    id_to_name = _object_names_by_definition_id() if global_ids else None

    results: list[dict] = []
    for request, lookup_name, lookup_gid in requests:
        element, blender_name = _find_ifc_element(ifc, lookup_name, lookup_gid, id_to_name)
        if element is None:
            results.append({"request": request, "error": "not found"})
            continue
        try:
            psets = _get_psets(element, psets_only=True) or {}
            qtos = _get_psets(element, qtos_only=True) or {}
            results.append(
                {
                    "request": request,
                    "object": {
                        "name": blender_name,
                        "global_id": getattr(element, "GlobalId", None),
                        "ifc_class": element.is_a(),
                    },
                    "property_sets": psets,
                    "quantity_sets": qtos,
                }
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                {"request": request, "error": f"{type(exc).__name__}: {exc}"}
            )

    return {"results": results}


def _find_storey(ifc, key: str):
    """Find an IfcBuildingStorey by GlobalId or (exact) Name."""
    try:
        storeys = list(ifc.by_type("IfcBuildingStorey"))
    except Exception:
        return None
    for storey in storeys:
        if getattr(storey, "GlobalId", None) == key:
            return storey
    for storey in storeys:
        if (getattr(storey, "Name", None) or "") == key:
            return storey
    return None


def _contained_element_ids(spatial) -> set:
    """Ids of elements contained in a spatial element, recursing into
    aggregated children (so elements inside a storey's spaces count too)."""
    ids: set = set()
    seen: set = set()
    stack = [spatial]
    while stack:
        node = stack.pop()
        try:
            node_id = node.id()
        except Exception:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        for rel in getattr(node, "ContainsElements", None) or []:
            for element in getattr(rel, "RelatedElements", None) or []:
                with contextlib.suppress(Exception):
                    ids.add(element.id())
        for rel in getattr(node, "IsDecomposedBy", None) or []:
            for child in getattr(rel, "RelatedObjects", None) or []:
                stack.append(child)
    return ids


def _element_ids_in_storey(key: str) -> set:
    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )
    storey = _find_storey(ifc, key)
    if storey is None:
        raise ValueError(
            f"No IfcBuildingStorey with Name or GlobalId {key!r}. "
            "Use get_spatial_structure to list the storeys."
        )
    return _contained_element_ids(storey)


_SELECTOR_CHEAT_SHEET = (
    "Selector examples: 'IfcWall' (one class), 'IfcWall, IfcSlab' (union), "
    "'IfcWall, material=concrete', 'IfcWall, Pset_WallCommon.FireRating=F30' "
    "(property value), 'IfcElement, Name=/W.*1/' (regex on an attribute). "
    "This is ifcopenshell.util.selector syntax."
)


def _selector_element_ids(selector: str) -> set:
    """Resolve an IfcOpenShell selector query to a set of element ids."""
    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )
    try:
        import ifcopenshell.util.selector as selector_util  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ifcopenshell.util.selector is not available in this Blender "
            "environment; update Bonsai/IfcOpenShell to use the 'selector' "
            "filter, or filter by ifc_class/name_contains instead."
        ) from exc
    try:
        elements = selector_util.filter_elements(ifc, selector)
    except Exception as exc:
        raise ValueError(
            f"Invalid selector {selector!r}: {exc}. {_SELECTOR_CHEAT_SHEET}"
        ) from exc
    ids: set = set()
    for element in elements:
        with contextlib.suppress(Exception):
            ids.add(element.id())
    return ids


def _h_list_elements(params):
    """List IFC-backed objects with class/name/storey/selector filters and paging."""
    params = params or {}
    ifc_class = params.get("ifc_class") or None
    name_contains = (params.get("name_contains") or "").strip().lower() or None
    storey_key = params.get("storey") or None
    selector = (params.get("selector") or "").strip() or None

    storey_ids = _element_ids_in_storey(storey_key) if storey_key else None
    selector_ids = _selector_element_ids(selector) if selector else None

    matches = []
    for obj in bpy.context.scene.objects:
        element = _element_for_object(obj)
        if element is None:
            continue
        if ifc_class:
            try:
                if not element.is_a(ifc_class):
                    continue
            except Exception:
                continue
        if name_contains and name_contains not in (obj.name or "").lower():
            continue
        if storey_ids is not None:
            try:
                if element.id() not in storey_ids:
                    continue
            except Exception:
                continue
        if selector_ids is not None:
            try:
                if element.id() not in selector_ids:
                    continue
            except Exception:
                continue
        matches.append(obj)

    page, total, truncated, limit, offset = _paginate(matches, params)
    return {
        "elements": [_object_summary(o) for o in page],
        "total": total,
        "truncated": truncated,
        "offset": offset,
        "limit": limit,
    }


_SPATIAL_TREE_MAX_DEPTH = 10


def _spatial_node(entity, include_counts: bool, depth: int = 0) -> dict:
    node: dict = {"name": getattr(entity, "Name", None)}
    try:
        node["ifc_class"] = entity.is_a()
    except Exception:
        node["ifc_class"] = None
    node["global_id"] = getattr(entity, "GlobalId", None)
    with contextlib.suppress(Exception):
        if entity.is_a("IfcBuildingStorey"):
            node["elevation"] = getattr(entity, "Elevation", None)

    if include_counts:
        counts: dict[str, int] = {}
        for rel in getattr(entity, "ContainsElements", None) or []:
            for element in getattr(rel, "RelatedElements", None) or []:
                with contextlib.suppress(Exception):
                    cls = element.is_a()
                    counts[cls] = counts.get(cls, 0) + 1
        if counts:
            node["element_counts"] = dict(sorted(counts.items()))
            node["element_total"] = sum(counts.values())

    children: list[dict] = []
    if depth < _SPATIAL_TREE_MAX_DEPTH:
        for rel in getattr(entity, "IsDecomposedBy", None) or []:
            for child in getattr(rel, "RelatedObjects", None) or []:
                with contextlib.suppress(Exception):
                    children.append(_spatial_node(child, include_counts, depth + 1))
    if children:
        node["children"] = children
    return node


def _h_get_spatial_structure(params):
    """Site -> building -> storey -> space tree with optional element counts."""
    params = params or {}
    include_counts = bool(params.get("include_element_counts", True))
    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )
    project = next(iter(ifc.by_type("IfcProject")), None)
    if project is None:
        raise RuntimeError("The IFC file contains no IfcProject entity.")
    return {
        "schema": getattr(ifc, "schema", None),
        "tree": _spatial_node(project, include_counts),
    }


def _storey_of(element):
    """Climb containment/aggregation up to the IfcBuildingStorey, or None."""
    seen: set = set()
    current = element
    while current is not None:
        try:
            current_id = current.id()
        except Exception:
            return None
        if current_id in seen:
            return None
        seen.add(current_id)
        with contextlib.suppress(Exception):
            if current.is_a("IfcBuildingStorey"):
                return current
        parent = None
        for rel in getattr(current, "ContainedInStructure", None) or []:
            parent = getattr(rel, "RelatingStructure", None)
            if parent is not None:
                break
        if parent is None:
            for rel in getattr(current, "Decomposes", None) or []:
                parent = getattr(rel, "RelatingObject", None)
                if parent is not None:
                    break
        current = parent
    return None


def _add_quantities(bucket: dict, qtos: dict) -> bool:
    """Accumulate numeric quantity values into bucket; True if any were found."""
    found = False
    for qset in (qtos or {}).values():
        if not isinstance(qset, dict):
            continue
        for qname, qval in qset.items():
            if qname == "id" or isinstance(qval, bool):
                continue
            if isinstance(qval, (int, float)):
                entry = bucket.setdefault(qname, {"sum": 0.0, "elements": 0})
                entry["sum"] += float(qval)
                entry["elements"] += 1
                found = True
    return found


def _round_quantity_sums(bucket: dict) -> dict:
    return {
        qname: {"sum": round(entry["sum"], 4), "elements": entry["elements"]}
        for qname, entry in sorted(bucket.items())
    }


def _project_units(ifc) -> dict:
    """Best-effort map of unit type -> unit name (e.g. LENGTHUNIT -> millimetre)."""
    units: dict[str, str] = {}
    try:
        project = next(iter(ifc.by_type("IfcProject")), None)
        assignment = getattr(project, "UnitsInContext", None)
        for unit in getattr(assignment, "Units", None) or []:
            unit_type = getattr(unit, "UnitType", None)
            name = getattr(unit, "Name", None)
            if not unit_type or not name:
                continue
            prefix = getattr(unit, "Prefix", None)
            label = f"{prefix}{name}".lower() if prefix else str(name).lower()
            units[str(unit_type)] = label
    except Exception:
        pass
    return units


_DEFAULT_QUANTITY_CLASSES = (
    "IfcWall",
    "IfcSlab",
    "IfcColumn",
    "IfcBeam",
    "IfcDoor",
    "IfcWindow",
    "IfcRoof",
    "IfcStair",
    "IfcCovering",
    "IfcSpace",
)


def _h_get_quantities(params):
    """Aggregate IFC base quantities by class, optionally per storey."""
    params = params or {}
    classes = params.get("ifc_classes") or list(_DEFAULT_QUANTITY_CLASSES)
    if not isinstance(classes, (list, tuple)) or not all(
        isinstance(c, str) and c for c in classes
    ):
        raise ValueError("'ifc_classes' must be a list of IFC class names")
    by_storey = bool(params.get("by_storey", False))

    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )
    try:
        from ifcopenshell.util.element import get_psets as _get_psets_util  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"IfcOpenShell util.element is unavailable: {exc}") from exc

    classes_out: dict[str, dict] = {}
    storeys_out: dict[str, dict] = {}
    for ifc_class in classes:
        try:
            elements = list(ifc.by_type(ifc_class))
        except Exception:
            classes_out[ifc_class] = {
                "count": 0,
                "quantities": {},
                "note": "class not found in this IFC schema",
            }
            continue

        bucket: dict = {}
        without_quantities = 0
        for element in elements:
            qtos: dict = {}
            with contextlib.suppress(Exception):
                qtos = _get_psets_util(element, qtos_only=True) or {}
            if not _add_quantities(bucket, qtos):
                without_quantities += 1

            if by_storey:
                storey = _storey_of(element)
                storey_key = (
                    getattr(storey, "Name", None)
                    or getattr(storey, "GlobalId", None)
                    or "(no storey)"
                ) if storey is not None else "(no storey)"
                storey_entry = storeys_out.setdefault(storey_key, {})
                class_entry = storey_entry.setdefault(
                    ifc_class, {"count": 0, "quantities": {}}
                )
                class_entry["count"] += 1
                _add_quantities(class_entry["quantities"], qtos)

        classes_out[ifc_class] = {
            "count": len(elements),
            "elements_without_quantities": without_quantities,
            "quantities": _round_quantity_sums(bucket),
        }

    out: dict = {"classes": classes_out, "units": _project_units(ifc)}
    if by_storey:
        for storey_entry in storeys_out.values():
            for class_entry in storey_entry.values():
                class_entry["quantities"] = _round_quantity_sums(class_entry["quantities"])
        out["by_storey"] = storeys_out
    return out


def _h_save_ifc_file(params):
    output_path = params.get("output_path")
    overwrite = bool(params.get("overwrite", False))
    reload_after = bool(params.get("reload", False))

    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError("No IFC project is loaded; cannot save.")

    in_place = not output_path
    if in_place:
        output_path = _bonsai_project_path()
        if not output_path:
            raise RuntimeError(
                "The project has no file path yet (it was never saved). "
                "Pass 'output_path' to choose where to save it."
            )
    else:
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file at {output_path}. "
                "Pass overwrite=true to force, or omit output_path entirely to "
                "save the project back to its own file."
            )
        parent = os.path.dirname(output_path)
        if parent and not os.path.isdir(parent):
            raise FileNotFoundError(f"Output directory does not exist: {parent}")

    method = _save_ifc_project(output_path)

    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"Save reported no error but no file was written at {output_path}."
        )

    reloaded = False
    if reload_after:
        _reload_ifc_project(output_path)
        reloaded = True

    return {
        "saved": True,
        "output_path": output_path,
        "in_place": in_place,
        "method": method,
        "reloaded": reloaded,
    }


_REFRESH_MAX_TARGETS = 200


def _blender_name_for_element(tool_mod, element) -> str:
    """Bonsai's object-name convention for an element ('IfcClass/Name')."""
    with contextlib.suppress(Exception):
        return tool_mod.Loader.get_name(element)
    name = getattr(element, "Name", None) or "Unnamed"
    return f"{element.is_a()}/{name}"


def _refresh_targets(params) -> list[str]:
    global_ids = list((params or {}).get("global_ids") or [])
    if not global_ids:
        raise ValueError("'global_ids' is required: pass the GlobalIds of the edited elements.")
    if len(global_ids) > _REFRESH_MAX_TARGETS:
        raise ValueError(
            f"Too many targets ({len(global_ids)} > {_REFRESH_MAX_TARGETS}). "
            "Refresh in batches, or use reload_project if most of the model changed."
        )
    return global_ids


def _require_bonsai_and_ifc():
    """Return (tool module, ifc file); refresh needs both."""
    tool_mod = _get_bonsai_tool()
    if tool_mod is None:
        raise RuntimeError("Bonsai (bonsai.tool) is not available; cannot refresh the scene.")
    ifc = _get_loaded_ifc()
    if ifc is None:
        raise RuntimeError(
            "No IFC project appears to be loaded. Open an IFC file in Bonsai first."
        )
    return tool_mod, ifc


def _sync_object_placement(tool_mod, ifc, element, obj) -> bool:
    """Set obj.matrix_world from the element's IFC placement. True on success."""
    try:
        import ifcopenshell.util.placement as placement_util  # type: ignore
        import ifcopenshell.util.unit as unit_util  # type: ignore

        placement = getattr(element, "ObjectPlacement", None)
        if placement is None:
            return False
        matrix = placement_util.get_local_placement(placement).copy()
        matrix[:3, 3] *= unit_util.calculate_unit_scale(ifc)
        obj.matrix_world = tool_mod.Loader.apply_blender_offset_to_matrix_world(obj, matrix)
        return True
    except Exception:
        return False


def _h_refresh_view(params):
    """Tier 1: sync names after data-only IFC edits. No geometry, no disk I/O."""
    global_ids = _refresh_targets(params)
    tool_mod, ifc = _require_bonsai_and_ifc()

    results: list[dict] = []
    for gid in global_ids:
        entry: dict = {"global_id": gid}
        try:
            element = ifc.by_guid(gid)
        except Exception:
            entry["error"] = (
                "No element with this GlobalId in the loaded IFC. If you deleted "
                "it, use reload_project to remove its object from the scene."
            )
            results.append(entry)
            continue
        obj = None
        with contextlib.suppress(Exception):
            obj = tool_mod.Ifc.get_object(element)
        if obj is None:
            entry["error"] = (
                "The element has no Blender object yet (newly created?). "
                "reload_project materializes new elements."
            )
            results.append(entry)
            continue
        new_name = _blender_name_for_element(tool_mod, element)
        entry["object"] = obj.name
        if obj.name != new_name:
            obj.name = new_name
            entry["object"] = new_name
            entry["renamed"] = True
        results.append(entry)

    return {
        "refreshed": sum(1 for r in results if "error" not in r),
        "results": results,
        "note": (
            "Data-only sync (names). Nothing was written to disk; edits stay "
            "in memory until the project is saved."
        ),
    }


def _h_refresh_geometry(params):
    """Tier 2: rebuild geometry and placement for specific elements. No disk I/O."""
    global_ids = _refresh_targets(params)
    tool_mod, ifc = _require_bonsai_and_ifc()

    results: list[dict] = []
    to_reload = []
    for gid in global_ids:
        entry: dict = {"global_id": gid}
        try:
            element = ifc.by_guid(gid)
        except Exception:
            entry["error"] = (
                "No element with this GlobalId in the loaded IFC. If you deleted "
                "it, use reload_project to remove its object from the scene."
            )
            results.append(entry)
            continue
        obj = None
        with contextlib.suppress(Exception):
            obj = tool_mod.Ifc.get_object(element)
        if obj is None:
            entry["error"] = (
                "The element has no Blender object yet (newly created?). "
                "reload_project materializes new elements."
            )
            results.append(entry)
            continue
        entry["object"] = obj.name
        entry["placement_synced"] = _sync_object_placement(tool_mod, ifc, element, obj)
        with contextlib.suppress(Exception):
            new_name = _blender_name_for_element(tool_mod, element)
            if obj.name != new_name:
                obj.name = new_name
                entry["object"] = new_name
        to_reload.append(obj)
        results.append(entry)

    representations_reloaded = False
    if to_reload:
        try:
            tool_mod.Geometry.reload_representation(to_reload)
            representations_reloaded = True
        except Exception as exc:
            for entry in results:
                if "error" not in entry:
                    entry["representation_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "refreshed": sum(1 for r in results if "error" not in r),
        "representations_reloaded": representations_reloaded,
        "results": results,
        "note": (
            "Targeted geometry rebuild. Nothing was written to disk; edits stay "
            "in memory until the project is saved."
        ),
    }


def _restore_project_path(path: str) -> bool:
    """Point Bonsai's project path back at `path` (after a temp-file reload)."""
    import importlib

    restored = False
    for mod_name in ("bonsai.bim.ifc", "blenderbim.bim.ifc"):
        with contextlib.suppress(Exception):
            importlib.import_module(mod_name).IfcStore.path = path
            restored = True
            break
    with contextlib.suppress(Exception):
        bpy.context.scene.BIMProperties.ifc_file = path  # type: ignore[attr-defined]
    return restored


def _reload_temp_path(original: str) -> str:
    """A temp path for reload_project, reused across calls, cleaned on stop."""
    tmp_dir = _STATE.get("reload_dir")
    if not isinstance(tmp_dir, str):
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="bonsai_mcp_reload_")
        _STATE["reload_dir"] = tmp_dir
    return os.path.join(tmp_dir, os.path.basename(original) or "project.ifc")


def _h_reload_project(_params):
    """Tier 3: rebuild the whole scene from the in-memory model.

    Saves to a temp file and reloads from it, then restores the project path,
    so the user's file on disk is never touched and Ctrl+S still saves to it.
    """
    tool_mod = _get_bonsai_tool()
    if tool_mod is None:
        raise RuntimeError("Bonsai is not available; cannot reload the project.")
    original = _bonsai_project_path()
    if not original:
        raise RuntimeError(
            "The project has no file path yet (it was never saved). Save it "
            "once in Blender (or via save_ifc_file with output_path) first."
        )
    tmp_path = _reload_temp_path(original)
    _save_ifc_project(tmp_path)
    _reload_ifc_project(tmp_path)
    path_restored = _restore_project_path(original)
    return {
        "reloaded": True,
        "project_path": original,
        "path_restored": path_restored,
        "note": (
            "Scene rebuilt from the in-memory model via a temporary file. The "
            "project file on disk was not modified; saving still targets "
            f"{original}."
        ),
    }


def _cleanup_reload_dir() -> None:
    tmp = _STATE.pop("reload_dir", None)
    if isinstance(tmp, str):
        shutil.rmtree(tmp, ignore_errors=True)



# Pure-Python deterministic geometry planner. Coordinates are SI metres.
def _ego_plan(meshes, camera_height, seed, frame_count, fps):
    import math
    import random
    from collections import Counter

    if not (0 <= float(camera_height) <= 10):
        raise ValueError("camera_height must be between 0 and 10 metres")
    if not (1 <= frame_count <= 216000 and 1 <= fps <= 120):
        raise ValueError("Invalid planner frame count or fps")
    radius, step = 0.30, 0.06
    clearance = radius + step / 2 + 0.005
    body_height = max(1.8, camera_height + 0.2)

    def sub(a, b):
        return tuple(a[i] - b[i] for i in range(3))

    def add(a, b):
        return tuple(a[i] + b[i] for i in range(3))

    def mul(a, s):
        return tuple(v * s for v in a)

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def norm2(a):
        return dot(a, a)

    def ray(origin, direction, tri):
        a, b, c = tri
        e1, e2 = sub(b, a), sub(c, a)
        h = cross(direction, e2)
        det = dot(e1, h)
        if abs(det) < 1e-10:
            return None
        s = sub(origin, a)
        u = dot(s, h) / det
        if u < -1e-8 or u > 1 + 1e-8:
            return None
        q = cross(s, e1)
        v = dot(direction, q) / det
        if v < -1e-8 or u + v > 1 + 1e-8:
            return None
        t = dot(e2, q) / det
        return t if t >= -1e-8 else None

    def point_triangle(p, tri):
        a, b, c = tri
        ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0 and d2 <= 0:
            return norm2(ap)
        bp = sub(p, b)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0 and d4 <= d3:
            return norm2(bp)
        vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            return norm2(sub(p, add(a, mul(ab, d1 / (d1 - d3)))))
        cp = sub(p, c)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0 and d5 <= d6:
            return norm2(cp)
        vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            return norm2(sub(p, add(a, mul(ac, d2 / (d2 - d6)))))
        va = d3 * d6 - d5 * d4
        if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
            return norm2(sub(p, add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))))))
        n = cross(ab, ac)
        return dot(ap, n) ** 2 / norm2(n)

    def segments(a, b, c, d):
        # Closest points on two finite segments, including parallel edges.
        u, v, w = sub(b, a), sub(d, c), sub(a, c)
        aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
        den = aa * cc - bb * bb
        s = max(0.0, min(1.0, (bb * ee - cc * dd) / den)) if den > 1e-15 else 0.0
        t = (bb * s + ee) / cc if cc > 1e-15 else 0.0
        if t < 0:
            t, s = 0.0, max(0.0, min(1.0, -dd / aa))
        elif t > 1:
            t, s = 1.0, max(0.0, min(1.0, (bb - dd) / aa))
        return norm2(sub(add(a, mul(u, s)), add(c, mul(v, t))))

    def capsule_distance(a, b, tri):
        hit = ray(a, sub(b, a), tri)
        if hit is not None and hit <= 1:
            return 0.0
        return min(
            point_triangle(a, tri),
            point_triangle(b, tri),
            *(segments(a, b, tri[i], tri[(i + 1) % 3]) for i in range(3)),
        )

    def bounds(tri):
        return tuple(min(p[i] for p in tri) for i in range(3)), tuple(
            max(p[i] for p in tri) for i in range(3)
        )

    def record(tri):
        return (tri, *bounds(tri))

    spaces, obstacles, floors, walls, solids = [], [], [], [], []
    space_names = []
    total = 0
    for mesh in meshes:
        cls = str(mesh.get("ifc_class", ""))
        if cls == "IfcDoor":
            continue
        triangles = []
        for raw in mesh.get("triangles", []):
            total += 1
            if total > 250000:
                raise ValueError("Interior planner triangle limit exceeded")
            tri = tuple(tuple(float(x) for x in p) for p in raw)
            if len(tri) != 3 or any(
                len(p) != 3 or not all(math.isfinite(x) and abs(x) < 1e7 for x in p) for p in tri
            ):
                raise ValueError("Invalid world-space triangle")
            if norm2(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))) < 1e-16:
                continue
            triangles.append(tri)
        if cls == "IfcSpace":
            # A space is trusted only when its triangle shell is watertight.
            edges = Counter()
            for tri in triangles:
                keys = [tuple(round(x, 5) for x in p) for p in tri]
                for i in range(3):
                    edges[tuple(sorted((keys[i], keys[(i + 1) % 3])))] += 1
            if triangles and edges and all(n == 2 for n in edges.values()):
                spaces.append(triangles)
                space_names.append(str(mesh.get("name", "IfcSpace")))
        elif cls not in {
            "IfcOpeningElement",
            "IfcAnnotation",
            "IfcSite",
            "IfcBuilding",
            "IfcBuildingStorey",
        }:
            obstacles.extend(record(t) for t in triangles)
            counts = Counter()
            for tri in triangles:
                keys = [tuple(round(x, 5) for x in p) for p in tri]
                for i in range(3):
                    counts[tuple(sorted((keys[i], keys[(i + 1) % 3])))] += 1
            if triangles and all(n == 2 for n in counts.values()):
                lo = tuple(min(p[i] for tri in triangles for p in tri) for i in range(3))
                hi = tuple(max(p[i] for tri in triangles for p in tri) for i in range(3))
                solids.append((triangles, lo, hi))
            if cls in {"IfcSlab", "IfcCovering"}:
                floors.extend(triangles)
            if cls in {"IfcWall", "IfcWallStandardCase", "IfcCurtainWall"}:
                walls.extend(triangles)
    if not obstacles:
        raise ValueError("No physical building geometry for safe interior planning")
    if any(str(m.get("ifc_class", "")) == "IfcSpace" for m in meshes) and not spaces:
        raise ValueError("IFC spaces are not watertight; refusing uncertain interior")

    def inside(point, triangles):
        hits = [ray(point, (0.912871, 0.365148, 0.182574), t) for t in triangles]
        return len({round(t, 6) for t in hits if t is not None and t > 1e-7}) % 2 == 1

    space_bounds = [
        (
            tuple(min(p[i] for t in s for p in t) for i in range(3)),
            tuple(max(p[i] for t in s for p in t) for i in range(3)),
        )
        for s in spaces
    ]

    def rooms_at(point):
        return {
            i
            for i, s in enumerate(spaces)
            if all(
                space_bounds[i][0][j] - 1e-7 <= point[j] <= space_bounds[i][1][j] + 1e-7
                for j in range(3)
            )
            and inside(point, s)
        }

    # Remove only proven coincident coplanar patches between adjacent spaces.
    # Patch boundary comparison tolerates different diagonals of the same face.
    # Noncoincident door gaps are deliberately not guessed to be navigable.
    shell = []
    if spaces:
        patches = {}
        for space_index, tris in enumerate(spaces):
            planes = {}
            for tri in tris:
                normal = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))
                normal = mul(normal, 1 / math.sqrt(norm2(normal)))
                if next(v for v in normal if abs(v) > 1e-8) < 0:
                    normal = mul(normal, -1)
                plane = tuple(round(v, 5) for v in normal) + (round(dot(normal, tri[0]), 5),)
                planes.setdefault(plane, []).append(tri)
            for plane, tris in planes.items():
                counts = Counter()
                for tri in tris:
                    keys = [tuple(round(v, 5) for v in p) for p in tri]
                    for i in range(3):
                        counts[tuple(sorted((keys[i], keys[(i + 1) % 3])))] += 1
                boundary = tuple(sorted(edge for edge, n in counts.items() if n != 2))
                patches.setdefault((plane, boundary), []).append((space_index, tris))
        for groups in patches.values():
            if len(groups) == 1:
                shell.extend(groups[0][1])
            elif len(groups) == 2:
                # Coincident duplicates, unlike adjacent volumes, are not a portal.
                tri = groups[0][1][0]
                center = tuple(sum(p[i] for p in tri) / 3 for i in range(3))
                normal = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))
                normal = mul(normal, 0.005 / math.sqrt(norm2(normal)))
                left = add(center, normal)
                if inside(left, spaces[groups[0][0]]) == inside(left, spaces[groups[1][0]]):
                    shell.extend(groups[0][1])
            else:
                raise ValueError("Overlapping IFC spaces have ambiguous boundaries")
    regions = ([shell, floors] if floors and len(spaces) > 1 else [shell]) if spaces else [floors]
    if not regions or not regions[0]:
        raise ValueError("No IFC space or bounded floor found")
    rng = random.Random(int(seed))
    checks = 0
    candidates = []
    # Horizontal support surfaces only: do not invent stairs or jump storeys.
    for region_index, region in enumerate(regions):
        horizontal = [t for t in region if max(p[2] for p in t) - min(p[2] for p in t) < 0.015]
        for tri in horizontal:
            lo, hi = bounds(tri)
            z = sum(p[2] for p in tri) / 3
            # Triangle centroids plus a bounded grid; ray test keeps points off holes.
            xy = [(sum(p[0] for p in tri) / 3, sum(p[1] for p in tri) / 3)]
            nx, ny = int((hi[0] - lo[0]) / 0.65) + 1, int((hi[1] - lo[1]) / 0.65) + 1
            if nx * ny > 10000:
                raise ValueError("Interior support extent exceeds planner limit")
            xy.extend((lo[0] + i * 0.65, lo[1] + j * 0.65) for i in range(nx) for j in range(ny))
            for x, y in xy:
                if ray((x, y, z + 0.1), (0, 0, -1), tri) is not None:
                    candidates.append((region_index, round(x, 5), round(y, 5), round(z, 5)))
            if len(candidates) > 20000:
                raise ValueError("Interior candidate limit exceeded")
    candidates = sorted(set(candidates))
    rng.shuffle(candidates)
    candidates.sort(key=lambda c: -c[0])
    duration = (frame_count - 1) / fps
    lengths = sorted(set([min(24.0, max(1.2, duration * 0.7)), 1.2]), reverse=True)
    directions = [2 * math.pi * i / 16 for i in range(16)]
    rng.shuffle(directions)
    # XY buckets avoid scanning every building triangle for each capsule sample.
    barrier_grid, broad_barriers = {}, []
    index_entries = 0
    physical_ids = {id(item[0]) for item in obstacles}
    for item in obstacles + ([record(t) for t in shell] if spaces else []):
        tri, lo, hi = item
        x0, x1 = math.floor(lo[0] - clearance), math.floor(hi[0] + clearance)
        y0, y1 = math.floor(lo[1] - clearance), math.floor(hi[1] + clearance)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 500:
            broad_barriers.append(item)
            continue
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                index_entries += 1
                if index_entries > 1500000:
                    raise ValueError("Interior spatial index limit exceeded")
                barrier_grid.setdefault((ix, iy), []).append(item)

    def graph_route(safe, support, floor_z):
        import heapq

        # A small regular graph, with each edge capsule-checked independently.
        # No spline is allowed to cut a corner outside the checked edge corridor.
        spacing = 0.45
        lo = tuple(min(p[i] for t in support for p in t) for i in range(2))
        hi = tuple(max(p[i] for t in support for p in t) for i in range(2))
        nx, ny = int((hi[0] - lo[0]) / spacing), int((hi[1] - lo[1]) / spacing)
        if nx * ny > 6000:
            return None
        points, labels = {}, {}
        # Same-level space envelopes prune remote slabs and external paving.
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                px, py = lo[0] + (ix + 0.5) * spacing, lo[1] + (iy + 0.5) * spacing
                mid = (px, py, floor_z + body_height / 2)
                if not any(
                    bounds_lo[0] - 0.75 < px < bounds_hi[0] + 0.75
                    and bounds_lo[1] - 0.75 < py < bounds_hi[1] + 0.75
                    and bounds_lo[2] < mid[2] < bounds_hi[2]
                    for bounds_lo, bounds_hi in space_bounds
                ):
                    continue
                if safe(px, py):
                    points[(ix, iy)] = (px, py)
                    labels[(ix, iy)] = rooms_at(mid)
        if len(points) < 3:
            return None
        edges = {}

        def edge_ok(a, b):
            key = tuple(sorted((a, b)))
            if key not in edges:
                p, q = points[a], points[b]
                dist = math.hypot(q[0] - p[0], q[1] - p[1])
                count = max(1, int(math.ceil(dist / step)))
                edges[key] = all(
                    safe(p[0] + (q[0] - p[0]) * i / count, p[1] + (q[1] - p[1]) * i / count)
                    for i in range(1, count)
                )
            return edges[key]

        def shortest(start):
            distances, prev = {start: 0.0}, {}
            queue = [(0.0, start)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance > distances[node] + 1e-9:
                    continue
                for dx, dy in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ]:
                    nxt = (node[0] + dx, node[1] + dy)
                    if nxt not in points or not edge_ok(node, nxt):
                        continue
                    cost = distance + spacing * math.hypot(dx, dy)
                    if cost < distances.get(nxt, float("inf")) - 1e-9:
                        distances[nxt], prev[nxt] = cost, node
                        heapq.heappush(queue, (cost, nxt))
            return distances, prev

        seeds = sorted(node for node in points if labels[node])
        rng.shuffle(seeds)
        covered = set()
        selected = None
        # Compare connected components instead of taking the first short room.
        for start in seeds:
            if start in covered:
                continue
            distances, _ = shortest(start)
            covered.update(distances)
            room_set = set().union(*(labels[node] for node in distances))
            if len(room_set) < 2:
                continue
            far = max(distances, key=lambda node: (distances[node], node))
            distances, prev = shortest(far)
            end = max(
                (node for node in distances if labels[node]),
                key=lambda node: (distances[node], node),
            )
            route = [end]
            while route[-1] != far:
                route.append(prev[route[-1]])
            route.reverse()
            rank = (len(room_set), distances[end])
            if selected is None or rank > selected[0]:
                selected = (rank, route)
        if selected is None:
            return None
        route = selected[1]
        # Greedy line-of-sight reduction uses the exact same conservative checker.
        reduced = [route[0]]
        index = 0
        while index < len(route) - 1:
            target = min(len(route) - 1, index + 50)
            while target > index + 1 and not edge_ok(route[index], route[target]):
                target -= 1
            reduced.append(route[target])
            index = target
        waypoints = [points[node] for node in reduced]
        lens = [math.dist(a, b) for a, b in zip(waypoints, waypoints[1:], strict=False)]
        total_length = sum(lens)
        if total_length < 1.2:
            return None
        # Stop at corners. Reserve time to rotate at <=60 degrees/second.
        headings = [
            math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(waypoints, waypoints[1:], strict=False)
        ]
        turns = [
            abs((b - a + math.pi) % (2 * math.pi) - math.pi)
            for a, b in zip(headings, headings[1:], strict=False)
        ]
        turn_times = [angle / (math.pi / 3) + 0.2 for angle in turns]
        available = duration - sum(turn_times)
        if available <= 0 or total_length / available > 1.4:
            return None
        # Constant edge speed with cubic easing over 0.25 s at each end.
        # Edge durations are length-proportional; corner holds avoid snap turns.
        edge_times = [available * length / total_length for length in lens]
        timeline = []
        clock = 0.0
        for i, seconds in enumerate(edge_times):
            timeline.append((clock, clock + seconds, i, False))
            clock += seconds
            if i < len(turn_times):
                timeline.append((clock, clock + turn_times[i], i, True))
                clock += turn_times[i]

        def eased_distance(t, seconds):
            ramp = min(0.25, seconds / 3)
            scale = 1 / (seconds - ramp)
            if t < ramp:
                return scale * t * t / (2 * ramp)
            if t > seconds - ramp:
                return 1 - scale * (seconds - t) ** 2 / (2 * ramp)
            return scale * (t - ramp / 2)

        positions = []
        segment = 0
        for i in range(frame_count):
            time = i / fps
            while segment < len(timeline) - 1 and time > timeline[segment][1]:
                segment += 1
            begin, end, index, hold = timeline[segment]
            a, b = waypoints[index], waypoints[index + 1]
            u = 1.0 if hold else eased_distance(max(0, min(end - begin, time - begin)), end - begin)
            positions.append(
                (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, floor_z + camera_height)
            )
        visited = sorted(
            {j for px, py, _ in positions for j in rooms_at((px, py, floor_z + body_height / 2))}
        )
        if len(visited) < 2:
            return None
        return positions, {
            "method": "enclosed_floor_graph",
            "seed": int(seed),
            "floor_z_m": floor_z,
            "planned_path_length_m": total_length,
            "path_length_m": sum(
                math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False)
            ),
            "human_radius_m": radius,
            "body_height_m": body_height,
            "heading": (math.cos(headings[0]), math.sin(headings[0]), 0.0),
            "clearance_checks": checks,
            "space_index": visited[0],
            "visited_space_indices": visited,
            "visited_space_names": [space_names[i] for i in visited],
            "multi_room_attempted": True,
            "route_waypoints_m": [(px, py, floor_z + camera_height) for px, py in waypoints],
            "corner_stop_seconds": turn_times,
            "mean_moving_speed_mps": total_length / available,
        }

    best = None
    multi_attempts = 0
    graph_tried = set()
    physical_levels = sorted(
        {
            round(sum(p[2] for p in t) / 3, 5)
            for t in floors
            if max(p[2] for p in t) - min(p[2] for p in t) < 0.015
        }
    )
    if not physical_levels:
        raise ValueError("No physical horizontal floor support found")
    for region_index, x, y, z in candidates[:700]:
        level = min(physical_levels, key=lambda level: abs(level - z))
        if abs(level - z) > 0.06:
            continue
        z = level
        bridge_mode = bool(spaces and region_index == 1)
        region = regions[region_index]
        support = [t for t in floors if all(abs(p[2] - z) < 0.015 for p in t)]
        if not support:
            continue
        edge_counts = Counter()
        for tri in support:
            keys = [tuple(round(v, 5) for v in p[:2]) for p in tri]
            for i in range(3):
                edge_counts[tuple(sorted((keys[i], keys[(i + 1) % 3])))] += 1
        support_edges = [edge for edge, count in edge_counts.items() if count != 2]
        safe_cache = {}

        def safe(px, py, safe_cache=safe_cache):
            key = (round(px, 5), round(py, 5))
            if key not in safe_cache:
                safe_cache[key] = safe_uncached(px, py)
            return safe_cache[key]

        def safe_uncached(
            px,
            py,
            z=z,
            support=support,
            support_edges=support_edges,
            bridge_mode=bridge_mode,
            region=region,
        ):
            nonlocal checks
            checks += 1
            if checks > 120000:
                raise ValueError("Interior planning search limit exceeded")
            # Exact distance to planar support boundary protects small holes as well.
            if not any(ray((px, py, z + 0.04), (0, 0, -1), t) is not None for t in support):
                return False
            for start, end in support_edges:
                ex, ey = end[0] - start[0], end[1] - start[1]
                denom = ex * ex + ey * ey
                t = (
                    max(0.0, min(1.0, ((px - start[0]) * ex + (py - start[1]) * ey) / denom))
                    if denom
                    else 0.0
                )
                if (px - start[0] - t * ex) ** 2 + (py - start[1] - t * ey) ** 2 <= clearance**2:
                    return False
            for angle in range(8):
                sx, sy = (
                    px + clearance * math.cos(angle * math.pi / 4),
                    py + clearance * math.sin(angle * math.pi / 4),
                )
                if not any(ray((sx, sy, z + 0.04), (0, 0, -1), t) is not None for t in support):
                    return False
            mid = (px, py, z + body_height / 2)
            for solid, lo, hi in solids:
                if all(lo[i] < mid[i] < hi[i] for i in range(3)) and inside(mid, solid):
                    return False
            if spaces and not bridge_mode:
                if not inside(mid, region):
                    return False
            elif bridge_mode and all(
                rooms_at(probe)
                for probe in (
                    (px, py, z + 0.05),
                    (px, py, z + body_height),
                    (px - clearance, py, mid[2]),
                    (px + clearance, py, mid[2]),
                    (px, py - clearance, mid[2]),
                    (px, py + clearance, mid[2]),
                )
            ):
                pass
            else:
                if bridge_mode and not any(point_triangle(mid, t) < 0.75**2 for t in shell):
                    return False
                # No space semantics: insist on nearby walls on every bearing and a roof.
                for k in range(16):
                    direction = (math.cos(k * math.pi / 8), math.sin(k * math.pi / 8), 0)
                    if not any(
                        (lambda h: h is not None and clearance < h < 15)(ray(mid, direction, t))
                        for t in walls
                    ):
                        return False
                if not any(
                    (lambda h: h is not None and body_height / 2 < h < 8)(ray(mid, (0, 0, 1), t))
                    for t, _, _ in obstacles
                ):
                    return False
            a, b = (px, py, z + clearance + 0.015), (px, py, z + body_height - clearance)
            for tri, lo, hi in (
                barrier_grid.get((math.floor(px), math.floor(py)), []) + broad_barriers
            ):
                if id(tri) not in physical_ids and (
                    bridge_mode or all(abs(p[2] - z) < 0.08 for p in tri)
                ):
                    continue
                if (
                    hi[0] < px - clearance
                    or lo[0] > px + clearance
                    or hi[1] < py - clearance
                    or lo[1] > py + clearance
                    or hi[2] < a[2] - clearance
                    or lo[2] > b[2] + clearance
                ):
                    continue
                if capsule_distance(a, b, tri) <= clearance**2:
                    return False
            return True

        if not safe(x, y):
            continue
        if bridge_mode and round(z, 3) not in graph_tried:
            graph_tried.add(round(z, 3))
            graph_result = graph_route(safe, support, z)
            if graph_result is not None:
                return graph_result
        for length in lengths:
            for angle in directions:
                if best is not None:
                    multi_attempts += 1
                    if multi_attempts > 10000:
                        return best
                dx, dy = math.cos(angle) * length, math.sin(angle) * length
                n = int(math.ceil(length / step))
                start_rooms = rooms_at((x, y, z + camera_height))
                end_rooms = rooms_at((x + dx, y + dy, z + camera_height))
                if bridge_mode and (not start_rooms or not end_rooms or start_rooms == end_rooms):
                    continue
                if (
                    best is not None
                    and length <= best[1]["path_length_m"]
                    and (not start_rooms or not end_rooms or start_rooms == end_rooms)
                ):
                    continue
                if not safe(x + dx, y + dy):
                    continue
                if not all(safe(x + dx * i / n, y + dy * i / n) for i in range(1, n)):
                    continue
                # Smoothstep gives a forward-only path with zero endpoint velocity.
                positions = []
                for i in range(frame_count):
                    t = i / (frame_count - 1) if frame_count > 1 else 0.0
                    u = t * t * (3 - 2 * t)
                    positions.append((x + dx * u, y + dy * u, z + camera_height))
                visited = sorted(
                    {j for px, py, _ in positions for j in rooms_at((px, py, z + body_height / 2))}
                )
                result = (
                    positions,
                    {
                        "method": "enclosed_floor_bridge"
                        if bridge_mode
                        else ("ifc_space" if spaces else "enclosed_floor"),
                        "seed": int(seed),
                        "floor_z_m": z,
                        "planned_path_length_m": length,
                        "path_length_m": sum(
                            math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False)
                        ),
                        "human_radius_m": radius,
                        "body_height_m": body_height,
                        "heading": (math.cos(angle), math.sin(angle), 0.0),
                        "clearance_checks": checks,
                        "space_index": visited[0] if visited else None,
                        "visited_space_indices": visited,
                        "visited_space_names": [space_names[i] for i in visited],
                        "multi_room_attempted": len(spaces) > 1,
                    },
                )
                if len(spaces) < 2 or len(visited) > 1:
                    return result
                if best is None or length > best[1]["path_length_m"]:
                    best = result
    if best is not None:
        return best
    raise ValueError("No conservatively safe single-storey interior route found")


def _ego_portal_geometry(ifc, edges, meshes):
    """Resolve only the graph's existing connectors, without adding any edges."""
    import numpy as np
    from ifcopenshell.util.placement import get_axis2placement, get_local_placement
    from ifcopenshell.util.unit import calculate_unit_scale

    scale = calculate_unit_scale(ifc)
    available = {m.get("ifc_id") for m in meshes}
    resolved = []
    for edge in edges:
        item = dict(edge)
        patches = edge.get("evidence", {}).get("patches", [])
        triangles = []
        variants = []
        for patch in patches:
            rectangles = []
            for bid in patch["boundary_ids"]:
                boundary = ifc.by_id(bid)
                try:
                    surface = boundary.ConnectionGeometry.SurfaceOnRelatingElement
                    curve = surface.SweptCurve.Curve
                    if not surface.is_a("IfcSurfaceOfLinearExtrusion") or not curve.is_a("IfcPolyline") or len(curve.Points) != 2:
                        raise ValueError("Unsupported virtual portal surface")
                    mat = get_local_placement(boundary.RelatingSpace.ObjectPlacement) @ get_axis2placement(surface.Position)
                    points = [np.array(tuple(p.Coordinates) + (0.0,) * (3 - len(p.Coordinates))) for p in curve.Points]
                    direction = np.array(surface.ExtrudedDirection.DirectionRatios, dtype=float)
                    direction /= np.linalg.norm(direction)
                    rectangles.append(np.array([(mat @ np.append(v, 1))[:3] * scale for v in points + [v + direction * surface.Depth for v in points[::-1]]]))
                except (AttributeError, TypeError, RuntimeError) as exc:
                    raise ValueError(f"Cannot resolve portal boundary {bid}") from exc
            if len(rectangles) != 2:
                raise ValueError("Virtual portal needs exactly two boundary polygons")
            a, b = rectangles
            u = a[1] - a[0]
            u /= np.linalg.norm(u)
            normal = np.cross(u, [0, 0, 1])
            if abs(u[2]) > 1e-5 or np.max(np.abs((b - a[0]) @ normal)) > 1e-5:
                raise ValueError("Virtual portal polygons are not coplanar vertical rectangles")
            low = max(min((a - a[0]) @ u), min((b - a[0]) @ u))
            high = min(max((a - a[0]) @ u), max((b - a[0]) @ u))
            bottom, top = max(min(a[:, 2]), min(b[:, 2])), min(max(a[:, 2]), max(b[:, 2]))
            if high <= low or top <= bottom:
                raise ValueError("Virtual portal polygons do not overlap")
            vertices = []
            for along, z in ((low, bottom), (high, bottom), (high, top), (low, top)):
                q = a[0] + along * u
                q[2] = z
                vertices.append(tuple(float(v) for v in q))
            patch_triangles = [(vertices[0], vertices[1], vertices[2]), (vertices[0], vertices[2], vertices[3])]
            triangles.extend(patch_triangles)
            variants.append(patch_triangles)
        # IFC opening bodies are portal volumes, never collision exemptions.
        # They may have no Blender object because Bonsai hides opening objects.
        if not patches:
            ids = edge.get("evidence", {}).get("connector_ids", [edge["connector_id"]])
            for identifier in ids:
                entity = ifc.by_id(identifier)
                if not entity.is_a("IfcOpeningElement"):
                    continue
                if identifier in available:
                    for mesh in meshes:
                        if mesh.get("ifc_id") == identifier:
                            triangles.extend(mesh["triangles"])
                    continue
                import ifcopenshell.geom
                settings = ifcopenshell.geom.settings()
                settings.set(settings.USE_WORLD_COORDS, True)
                try:
                    shape = ifcopenshell.geom.create_shape(settings, entity)
                    verts, faces = shape.geometry.verts, shape.geometry.faces
                    points = [tuple(verts[i:i + 3]) for i in range(0, len(verts), 3)]
                    triangles.extend(tuple(points[faces[i + j]] for j in range(3)) for i in range(0, len(faces), 3))
                except (RuntimeError, ValueError) as exc:
                    raise ValueError(f"Cannot resolve opening geometry {identifier}") from exc
        if triangles:
            item["portal_triangles"] = triangles
        if variants:
            item["portal_variants"] = variants
        resolved.append(item)
    return resolved


def _ego_tour_plan(meshes, tour_plan, camera_height, frame_count, fps, component_id=None, progress=None):
    """Follow the exact semantic tour through room volumes and real connectors.

    Furniture, door leaves and human-body clearance do not constrain this
    plausible camera path. Wall crossings must use the selected portal.
    Stair heights and continuous timing reuse the existing tread traversal.
    """
    import heapq
    import math
    from collections import Counter
    from functools import lru_cache

    report = progress or (lambda *args, **kwargs: None)
    report("navigation_geometry", "Preparing room, connector and stair geometry", 0, len(meshes))

    def sub(a, b):
        return tuple(a[i] - b[i] for i in range(3))

    def add(a, b):
        return tuple(a[i] + b[i] for i in range(3))

    def mul(a, s):
        return tuple(v * s for v in a)

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def norm2(a):
        return dot(a, a)

    def ray(origin, direction, tri):
        a, b, c = tri
        e1, e2 = sub(b, a), sub(c, a)
        h = cross(direction, e2)
        det = dot(e1, h)
        if abs(det) < 1e-10:
            return None
        s = sub(origin, a)
        u = dot(s, h) / det
        if u < -1e-8 or u > 1 + 1e-8:
            return None
        q = cross(s, e1)
        v = dot(direction, q) / det
        if v < -1e-8 or u + v > 1 + 1e-8:
            return None
        t = dot(e2, q) / det
        return t if t >= -1e-8 else None

    def point_triangle(p, tri):
        a, b, c = tri
        ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0 and d2 <= 0:
            return norm2(ap)
        bp = sub(p, b)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0 and d4 <= d3:
            return norm2(bp)
        vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            return norm2(sub(p, add(a, mul(ab, d1 / (d1 - d3)))))
        cp = sub(p, c)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0 and d5 <= d6:
            return norm2(cp)
        vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            return norm2(sub(p, add(a, mul(ac, d2 / (d2 - d6)))))
        va = d3 * d6 - d5 * d4
        if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
            return norm2(sub(p, add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))))))
        n = cross(ab, ac)
        return dot(ap, n) ** 2 / norm2(n)

    def bounds(tri):
        return tuple(min(p[i] for p in tri) for i in range(3)), tuple(
            max(p[i] for p in tri) for i in range(3)
        )

    def record(tri):
        return (tri, *bounds(tri))


    if not math.isfinite(camera_height) or not 0 < camera_height <= 10:
        raise ValueError("Invalid camera height")
    if not 1 <= frame_count <= 216000 or not 1 <= fps <= 60:
        raise ValueError("Invalid frame count or fps")
    if tour_plan.get("graph_validation_status") != "validated_topology":
        raise ValueError("House tour graph is unresolved; refusing partial coverage")
    components = tour_plan.get("component_tours", [])
    if not components:
        raise ValueError("House tour has no eligible component")
    component = next((c for c in components if c["component_id"] == component_id), None) if component_id is not None else min(components, key=lambda c: c["component_id"])
    if component is None:
        raise ValueError(f"Unknown component_id {component_id}")
    route = component["route_space_ids"]
    if not route or component.get("coverage") != 1:
        raise ValueError("Selected component has no complete semantic tour")
    nodes = {n["id"]: n for n in tour_plan["nodes"]}
    wanted = set(route)
    spaces, walls, supports, connectors = {}, [], [], {}
    grid_step, sample_step = 0.15, 0.04
    total = 0

    def parse_triangles(raw_triangles):
        nonlocal total
        triangles = []
        for raw in raw_triangles:
            tri = tuple(tuple(float(v) for v in point) for point in raw)
            total += 1
            if total > 250000:
                raise ValueError("Tour planner triangle limit exceeded")
            if len(tri) != 3 or any(len(q) != 3 or not all(math.isfinite(v) and abs(v) < 1e7 for v in q) for q in tri):
                raise ValueError("Invalid world triangle")
            if norm2(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))) > 1e-16:
                triangles.append(tri)
        return triangles

    def is_closed(triangles):
        counts = Counter()
        for tri in triangles:
            keys = [tuple(round(v, 5) for v in q) for q in tri]
            for i in range(3):
                counts[tuple(sorted((keys[i], keys[(i + 1) % 3])))] += 1
        return bool(counts) and all(n == 2 for n in counts.values())

    for mesh_index, mesh in enumerate(meshes):
        report("navigation_geometry", "Reading room and connector geometry", mesh_index, len(meshes),
               ifc_id=mesh.get("ifc_id"), triangles_processed=total)
        cls, sid = mesh.get("ifc_class"), mesh.get("ifc_id")
        # Door entities still connect rooms; their physical meshes do not
        # constrain ego tours. Walls and authored opening portals stay intact.
        if cls == "IfcDoor":
            continue
        triangles = parse_triangles(mesh.get("triangles", []))
        if sid is not None:
            connectors.setdefault(sid, []).extend(triangles)
        closed = is_closed(triangles)
        if cls == "IfcSpace":
            if sid in wanted:
                if not closed or sid in spaces:
                    raise ValueError(f"Space {sid} has missing, duplicate or non-watertight geometry")
                spaces[sid] = triangles
        elif cls not in {"IfcOpeningElement", "IfcVirtualElement", "IfcAnnotation", "IfcSite", "IfcBuilding", "IfcBuildingStorey"}:
            if cls in {"IfcWall", "IfcWallStandardCase", "IfcCurtainWall"}:
                walls.extend(triangles)
            if cls in {"IfcSlab", "IfcCovering", "IfcStair", "IfcStairFlight", "IfcRamp", "IfcRampFlight"}:
                for tri in triangles:
                    normal = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))
                    if abs(normal[2]) / math.sqrt(norm2(normal)) > 0.85:
                        supports.append(tri)
    if set(spaces) != wanted or not supports:
        raise ValueError(f"Missing physical support or evaluated spaces {sorted(wanted - set(spaces))}")

    def bbox(tris):
        return tuple(min(q[i] for t in tris for q in t) for i in range(3)), tuple(max(q[i] for t in tris for q in t) for i in range(3))

    space_bounds = {sid: bbox(tris) for sid, tris in spaces.items()}
    def in_bounds(q, box, pad=0):
        return all(box[0][i] - pad <= q[i] <= box[1][i] + pad for i in range(3))

    def inside(q, tris):
        hits = [ray(q, (0.912871, 0.365148, 0.182574), tri) for tri in tris]
        return len({round(t, 6) for t in hits if t is not None and t > 1e-7}) % 2 == 1

    @lru_cache(maxsize=200000)
    def rooms(q):
        probe = (q[0], q[1], q[2] + 0.10)
        return {sid for sid, tris in spaces.items() if in_bounds(probe, space_bounds[sid]) and inside(probe, tris)}

    def camera_rooms(q):
        probe = (q[0], q[1], q[2] + camera_height)
        return {sid for sid, tris in spaces.items() if in_bounds(probe, space_bounds[sid]) and inside(probe, tris)}

    def index_triangles(tris, pad):
        index = {}
        cells = 0
        for tri in tris:
            lo, hi = bounds(tri)
            cells += (math.floor(hi[0] + pad) - math.floor(lo[0] - pad) + 1) * (math.floor(hi[1] + pad) - math.floor(lo[1] - pad) + 1)
            if cells > 2000000:
                raise ValueError("Tour geometry spatial index limit exceeded")
            for x in range(math.floor(lo[0] - pad), math.floor(hi[0] + pad) + 1):
                for y in range(math.floor(lo[1] - pad), math.floor(hi[1] + pad) + 1):
                    index.setdefault((x, y), []).append((tri, lo, hi))
        return index

    report("spatial_index", "Indexing walls and stair/floor heights",
           wall_triangles=len(walls), support_triangles=len(supports))
    wall_index = index_triangles(walls, 0)
    support_index = index_triangles(supports, 0)
    @lru_cache(maxsize=200000)
    def heights(x, y):
        result = set()
        for tri, lo, hi in support_index.get((math.floor(x), math.floor(y)), []):
            if lo[0] - 1e-7 <= x <= hi[0] + 1e-7 and lo[1] - 1e-7 <= y <= hi[1] + 1e-7:
                h = ray((x, y, hi[2] + 1), (0, 0, -1), tri)
                if h is not None:
                    result.add(round(hi[2] + 1 - h, 5))
        return sorted(result)

    def supported(q, stair=False):
        # One camera footpoint, not a five-contact human support footprint.
        if not stair:
            return True
        return any(-0.255 <= q[2] - z <= 0.255 for z in heights(q[0], q[1]))

    # The same physical step envelope applies on a flight's entry tread, even
    # when that tread is reached through the preceding opening edge. It never
    # applies to arbitrary low obstacles in a room.
    flight_triangles = []
    for edge in tour_plan["edges"]:
        if edge["kind"] == "stair" and {edge["source"], edge["target"]} <= wanted:
            for identifier in edge.get("evidence", {}).get("connector_ids", [edge["connector_id"]]):
                flight_triangles.extend(connectors.get(identifier, []))
    flight_records = [record(tri) for tri in flight_triangles]

    @lru_cache(maxsize=200000)
    def on_flight(q):
        # Preserve the existing entry-riser envelope and stair height routing.
        reach = 0.31
        return any(lo[2] - 0.26 <= q[2] <= hi[2] + 0.26
                   and in_bounds(q, (lo, hi), reach)
                   and point_triangle(q, tri) <= reach ** 2
                   for tri, lo, hi in flight_records)

    # Build alternate supported observations. A safe centroid-near point may
    # sit in a disconnected under-stair pocket, so reachability decides later.
    observation_candidates = {}
    observations = {}
    for space_index, sid in enumerate(sorted(wanted)):
        report("observations", f"Finding room observation points in space {sid}", space_index, len(wanted), space_id=sid)
        lo, hi = space_bounds[sid]
        center = tuple((lo[i] + hi[i]) / 2 for i in range(3))
        candidates = []
        nx, ny = math.ceil((hi[0] - lo[0]) / grid_step), math.ceil((hi[1] - lo[1]) / grid_step)
        if nx * ny > 200000:
            raise ValueError(f"Space {sid} exceeds navigation grid limit")
        for ix in range(math.ceil(lo[0] / grid_step), math.floor(hi[0] / grid_step) + 1):
            for iy in range(math.ceil(lo[1] / grid_step), math.floor(hi[1] / grid_step) + 1):
                x, y = ix * grid_step, iy * grid_step
                for z in heights(x, y):
                    q = (x, y, z)
                    if lo[2] - 0.10 <= z <= lo[2] + 0.30 and rooms(q) == {sid}:
                        candidates.append(((x - center[0]) ** 2 + (y - center[1]) ** 2, q))
        safe_candidates = []
        for candidate_index, (_, q) in enumerate(sorted(candidates)):
            if candidate_index % 100 == 0:
                report("observations", f"Checking space {sid}", space_index, len(wanted),
                       space_id=sid, candidates_checked=candidate_index, candidates_total=len(candidates))
            if camera_rooms(q) == {sid}:
                safe_candidates.append(q)
        if not safe_candidates:
            raise ValueError(f"No interior observation point for space {sid}")
        # Deterministic farthest-point sampling represents separate accessible
        # pockets and landing entries without exploding the route search.
        selected = [safe_candidates.pop(0)]
        while safe_candidates and len(selected) < 12:
            q = max(safe_candidates, key=lambda q: min(math.dist(q, other) for other in selected))
            safe_candidates.remove(q)
            selected.append(q)
        observation_candidates[sid] = selected
        observations[sid] = selected[0]

    def edge_portal(edge):
        tris = list(edge.get("portal_triangles", []))
        if not tris:
            ids = edge.get("evidence", {}).get("connector_ids", [edge["connector_id"]])
            for identifier in ids:
                tris.extend(connectors.get(identifier, []))
        if not tris:
            raise ValueError(f"Connector {edge['connector_id']} lacks evaluated portal geometry")
        return tris

    route_context = {}

    def solve(a, goals, source, target, edge):
        stair = edge["kind"] == "stair"
        portal = edge_portal(edge)
        portal_bounds = bbox(portal)
        plane_origin = plane_normal = None
        if not stair:
            vertical = []
            for tri in portal:
                n = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))
                area = math.sqrt(norm2(n))
                if area > 1e-9 and abs(n[2]) / area < 1e-5:
                    vertical.append((area, tri, mul(n, 1 / area)))
            if not vertical:
                raise ValueError(f"Connector {edge['connector_id']} lacks a vertical aperture")
            _, face, plane_normal = max(vertical, key=lambda item: item[0])
            plane_origin = face[0]
            # Project all parallel faces to one aperture plane. This supports
            # thick opening volumes without extending past their real width.
            projected = []
            for _, tri, n in vertical:
                if abs(dot(n, plane_normal)) > 1 - 1e-5:
                    projected.append(tuple(sub(v, mul(plane_normal, dot(sub(v, plane_origin), plane_normal))) for v in tri))

        def aperture(q):
            projection = sub(q, mul(plane_normal, dot(sub(q, plane_origin), plane_normal)))
            probe = add(projection, (0, 0, camera_height))
            return any(point_triangle(probe, tri) < 1e-10 for tri in projected)

        def near_tread(q):
            # Step clearance applies only within the physical flight envelope,
            # never to the approach or destination room as a whole.
            return in_bounds(q, portal_bounds, 0.26) and min(point_triangle(q, tri) for tri in portal) <= 0.26 ** 2

        def near_portal(q):
            if stair:
                return near_tread(q)
            return abs(dot(sub(q, plane_origin), plane_normal)) <= 0.40 and aperture(q)

        def crossing(a, b):
            if stair:
                return near_tread(a) or near_tread(b)
            da, db = dot(sub(a, plane_origin), plane_normal), dot(sub(b, plane_origin), plane_normal)
            if da * db > 0 or abs(da - db) < 1e-9:
                return False
            q = add(a, mul(sub(b, a), da / (da - db)))
            return aperture(q)

        def allowed(q):
            member = rooms(q)
            return bool(member & {source, target}) or near_portal(q)

        @lru_cache(maxsize=200000)
        def segment(a, b):
            if abs(a[2] - b[2]) > (0.255 if on_flight(a) or on_flight(b) else 0.04):
                return False
            # Point-camera wall guard, not body clearance. Actual opening
            # voids are traversable; adjacent jambs/internal walls are not.
            eye_a = add(a, (0, 0, camera_height))
            direction = sub(b, a)
            checked = set()
            for x in range(math.floor(min(a[0], b[0])), math.floor(max(a[0], b[0])) + 1):
                for y in range(math.floor(min(a[1], b[1])), math.floor(max(a[1], b[1])) + 1):
                    for tri, _lo, _hi in wall_index.get((x, y), ()):
                        if tri in checked:
                            continue
                        checked.add(tri)
                        hit = ray(eye_a, direction, tri)
                        if hit is not None and 1e-7 < hit < 1 - 1e-7:
                            return False
            n = max(1, math.ceil(math.dist(a, b) / sample_step))
            previous = rooms(a)
            for i in range(n + 1):
                q = tuple(a[k] + (b[k] - a[k]) * i / n for k in range(3))
                member = rooms(q)
                if not allowed(q) or not supported(q, on_flight(q)):
                    return False
                if member != previous and not near_portal(q):
                    return False
                previous = member
            return True

        def connection(a, b):
            if segment(a, b):
                return (a, b)
            # Raise before advancing onto a tread, or retreat before lowering.
            # Keep the existing continuous raise/advance or retreat/lower
            # motion, now without capsule or general-obstacle rejection.
            if 1e-7 < abs(a[2] - b[2]) <= 0.255 and on_flight(a) and on_flight(b):
                mid = (a[0], a[1], b[2]) if b[2] > a[2] else (b[0], b[1], a[2])
                if segment(a, mid) and segment(mid, b):
                    return (a, mid, b)
            return None

        # Prefer the simple observation -> connector -> observation path.
        # Room-volume/portal tests reject concave-room and wrong-wall shortcuts;
        # only those cases, and stairs, need the existing local grid routing.
        if not stair:
            center = tuple((portal_bounds[0][k] + portal_bounds[1][k]) / 2 for k in range(3))
            for goal in goals:
                point = (center[0], center[1], (a[2] + goal[2]) / 2)
                if (aperture(point) and segment(a, point) and segment(point, goal)
                        and (crossing(a, point) or crossing(point, goal))):
                    return [a, point, goal]

        start = (round(a[0] / grid_step), round(a[1] / grid_step), a[2], False)
        points = {start: a}
        costs, parents, parent_paths = {start: 0.0}, {}, {}
        preference = {goal: 5.0 * math.dist(goal, observation_candidates[target][0]) for goal in goals}

        def heuristic(q):
            return min(math.dist(q, goal) + preference[goal] for goal in goals)

        def reconstruct(key, goal):
            path = [goal, points[key]]
            while key in parents:
                path.extend(reversed(parent_paths[key][:-1]))
                key = parents[key]
            return list(reversed(path))

        best_goal = None
        queue = [(heuristic(a), start)]
        cache = {}
        expanded = 0
        settled = set()
        while queue:
            estimate, key = heapq.heappop(queue)
            if best_goal is not None and estimate >= best_goal[0] - 1e-9:
                return reconstruct(best_goal[1], best_goal[2])
            if key in settled:
                continue
            settled.add(key)
            q = points[key]
            expanded += 1
            if expanded == 1 or expanded % 500 == 0:
                report("routing", f"Searching {source} -> {target} via {edge['connector_id']}",
                       **route_context, expanded_nodes=expanded, frontier_size=len(queue))
            if expanded > 150000:
                raise ValueError(f"Connector {edge['connector_id']} route search limit exceeded")
            if key[3]:
                for goal in goals:
                    if math.dist(q, goal) < grid_step * 1.5 and segment(q, goal):
                        score = costs[key] + math.dist(q, goal) + preference[goal]
                        if best_goal is None or score < best_goal[0]:
                            best_goal = (score, key, goal)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ix, iy = key[0] + dx, key[1] + dy
                xy = (ix, iy)
                if xy not in cache:
                    cache[xy] = [(ix * grid_step, iy * grid_step, z) for z in heights(ix * grid_step, iy * grid_step)]
                for dest in cache[xy]:
                    if abs(dest[2] - q[2]) > (0.255 if on_flight(q) or on_flight(dest) else 0.04) or not allowed(dest):
                        continue
                    # A staged step can cross the aperture at a different
                    # height; determine crossing from the checked polyline.
                    link = connection(q, dest)
                    if link is None:
                        continue
                    crossed = key[3] or any(crossing(u, v) for u, v in zip(link, link[1:], strict=False))
                    next_key = (ix, iy, dest[2], crossed)
                    cost = costs[key] + sum(math.dist(u, v) for u, v in zip(link, link[1:], strict=False))
                    if cost >= costs.get(next_key, float("inf")):
                        continue
                    costs[next_key], parents[next_key], points[next_key] = cost, key, dest
                    parent_paths[next_key] = link
                    heapq.heappush(queue, (cost + heuristic(dest), next_key))
        if best_goal is not None:
            return reconstruct(best_goal[1], best_goal[2])
        raise ValueError(f"No connector-guided route from space {source} to {target} through connector {edge['connector_id']}")

    # Select observations jointly with the exact route. Failed later legs can
    # backtrack to another safe room pocket, never skip a room or connector.
    search_attempts = 0
    failure_messages = []
    failed_states = set()
    solved_legs = {}
    future_spaces = [set(route[i:]) for i in range(len(route))]

    def realize(index, start, chosen):
        nonlocal search_attempts
        if index == len(route) - 1:
            return [start], [0], [], chosen
        # Past observations that never recur cannot affect the remaining route.
        state = (index, start, tuple(sorted((sid, q) for sid, q in chosen.items()
                                           if sid in future_spaces[index])))
        if state in failed_states:
            return None
        source, target = route[index:index + 2]
        edges = sorted((e for e in tour_plan["edges"] if {e["source"], e["target"]} == {source, target}), key=lambda e: (e["connector_id"], e["kind"]))
        if not edges:
            raise ValueError(f"Semantic tour has no edge {source} -> {target}")
        for edge in edges:
            variants = edge.get("portal_variants", [None])
            for variant_index, variant in enumerate(variants):
                effective_edge = dict(edge, portal_triangles=variant) if variant is not None else edge
                goals = [chosen[target]] if target in chosen else list(observation_candidates[target])
                while goals:
                    # Geometry depends on these inputs, not the earlier room
                    # observations. Reuse exact successes and exact failures.
                    leg_key = (start, tuple(goals), source, target,
                               edge["connector_id"], edge["kind"], variant_index)
                    if leg_key not in solved_legs:
                        search_attempts += 1
                        if search_attempts > 2000:
                            raise ValueError("Observation reachability search limit exceeded; refusing partial tour")
                        route_context.update(route_index=index + 1, route_total=len(route) - 1,
                                             source_space_id=source, target_space_id=target,
                                             connector_id=edge["connector_id"], search_attempt=search_attempts)
                        report("routing", f"Trying {source} -> {target}; search may backtrack", **route_context)
                        try:
                            solved_legs[leg_key] = solve(start, goals, source, target, effective_edge)
                        except ValueError as exc:
                            solved_legs[leg_key] = str(exc)
                    leg = solved_legs[leg_key]
                    if isinstance(leg, str):
                        failure_messages.append(leg)
                        break
                    goal = leg[-1]
                    result = realize(index + 1, goal, {**chosen, target: goal})
                    if result is not None:
                        tail, visits, transitions, final_chosen = result
                        offset = len(leg) - 1
                        transition = {"source": source, "target": target, "connector_id": edge["connector_id"], "kind": edge["kind"], "portal_variant": variant_index}
                        return leg + tail[1:], [0] + [offset + v for v in visits], [transition] + transitions, final_chosen
                    goals.remove(goal)
        failed_states.add(state)
        return None

    realized = None
    for start in observation_candidates[route[0]]:
        realized = realize(0, start, {route[0]: start})
        if realized is not None:
            break
    if realized is None:
        detail = "; ".join(dict.fromkeys(failure_messages))
        raise ValueError("Required tour transition failed: " + detail)
    path, visit_indices, used_edges, observations = realized
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))
    def sample_tour(path, visit_indices, frame_count, fps):
        # Collision sampling is spatial proof, not the walking speed. Keep every
        # checked vertex; at low fps this can make travel slower than 1 m/s.
        walk_speed, turn_speed, observation_seconds = 1.0, math.radians(60), 0.8
        counts = [max(1, math.ceil(math.dist(a, b) * fps / walk_speed))
                  for a, b in zip(path, path[1:], strict=False)]
        turn_counts = [0] * len(path)
        previous_heading = None
        for i, (a, b) in enumerate(zip(path, path[1:], strict=False)):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if math.hypot(dx, dy) < 1e-8:
                continue
            heading = math.atan2(dy, dx)
            if previous_heading is not None:
                angle = abs((heading - previous_heading + math.pi) % (2 * math.pi) - math.pi)
                if angle > 1e-6:
                    turn_counts[i] = math.ceil(angle * fps / turn_speed - 1e-9)
            previous_heading = heading
        observation_counts = [0] * len(path)
        for index in visit_indices:
            observation_counts[index] += max(1, math.ceil(observation_seconds * fps))
        needed = 1 + sum(counts) + sum(turn_counts) + sum(observation_counts)
        # A longer requested duration adds room observation time, not slow motion.
        extra = max(0, frame_count - needed)
        base, rem = divmod(extra, len(visit_indices))
        for i, index in enumerate(visit_indices):
            observation_counts[index] += base + (i < rem)
        actual_count = needed + extra
        if actual_count > 216000:
            raise ValueError("Complete tour exceeds 216000 frames; refusing truncation")
        samples, path_frames, holds = [path[0]], {0: 0}, []
        for i, q in enumerate(path):
            for kind, count in (("turn", turn_counts[i]), ("observation", observation_counts[i])):
                if count:
                    first = len(samples) - 1
                    samples.extend([q] * count)
                    holds.append(dict(kind=kind, path_index=i, start_frame=first,
                                      end_frame=len(samples) - 1, seconds=count / fps))
            if i < len(counts):
                b, count = path[i + 1], counts[i]
                samples.extend(tuple(q[k] + (b[k] - q[k]) * j / count for k in range(3))
                               for j in range(1, count))
                samples.append(b)  # Exact endpoint, without floating-point drift.
                path_frames[i + 1] = len(samples) - 1
        timing = dict(walk_speed_m_s=walk_speed, turn_speed_degrees_s=60,
                      observation_pause_seconds=observation_seconds, holds=holds,
                      movement_seconds=sum(counts) / fps,
                      turn_seconds=sum(turn_counts) / fps,
                      observation_seconds=sum(observation_counts) / fps,
                      actual_duration_seconds=len(samples) / fps)
        return samples, path_frames, timing

    report("trajectory", "Sampling the complete physical camera trajectory")
    samples, path_frames, timing = sample_tour(path, visit_indices, frame_count, fps)
    frame_spaces = []
    for q in samples:
        found = camera_rooms(q)
        frame_spaces.append(next(iter(found)) if len(found) == 1 else None)
    visits = [{"space_id": sid, "frame": path_frames[index], "observation_point_m": list(observations[sid])} for sid, index in zip(route, visit_indices, strict=True)]
    for index, transition in enumerate(used_edges):
        transition.update(start_frame=path_frames[visit_indices[index]],
                          end_frame=path_frames[visit_indices[index + 1]])
    visited = sorted(set(route))
    room_ids = {sid for sid in component["space_ids"] if nodes[sid]["classification"] == "tourable_interior"}
    covered = room_ids & set(visited)
    if covered != room_ids:
        raise ValueError("Physical route missed a required tourable room")
    metadata = dict(method="semantic_connector_guided_tour", component_id=component["component_id"],
                    tour=list(route), semantic_tour=component["tour"], route_space_ids=list(route), visited_spaces=visited,
                    visits=visits, observation_points_m={str(k): list(v) for k, v in observations.items()},
                    tourable_rooms_visited=len(covered), tourable_rooms_total=len(room_ids),
                    coverage=len(covered) / len(room_ids) if room_ids else 1.0,
                    route_length_m=length, path_length_m=length, frame_space_ids=frame_spaces,
                    transitions=used_edges, door_collision_policy="IfcDoor_ignored", collision_policy="point_camera_walls_only", human_radius_m=0.0,
                    maximum_step_m=0.25, support_sample_spacing_m=sample_step, timing=timing)
    return [(q[0], q[1], q[2] + camera_height) for q in samples], metadata


def _ego_validate(params):
    """Validate again at the trust boundary; the TCP bridge is independently callable."""
    import math
    from pathlib import Path

    defaults = dict(duration_seconds=30, fps=10, width=1280, height=720,
                    camera_height=1.65, seed=0, component_id=None)
    unknown = set(params) - set(defaults) - {"output_path"}
    if unknown:
        raise ValueError(f"Unknown generate_ego_video arguments: {sorted(unknown)}")
    values = {key: params.get(key, default) for key, default in defaults.items()}
    for key, upper in (("duration_seconds", 3600), ("camera_height", 10)):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"{key} must be finite and in (0, {upper}]")
    for key, lower, upper in (("fps", 1, 60), ("width", 64, 4096),
                              ("height", 64, 4096), ("seed", 0, 2147483647)):
        if type(values[key]) is not int or not lower <= values[key] <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    if values["component_id"] is not None and (type(values["component_id"]) is not int or values["component_id"] < 0):
        raise ValueError("component_id must be a nonnegative integer or null")
    if values["width"] % 2 or values["height"] % 2:
        raise ValueError("width and height must be even for H.264")
    raw = params.get("output_path")
    if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
        raise ValueError("output_path must be a nonempty .mp4 path without NUL")
    path = Path(bpy.path.abspath(raw)).expanduser().absolute()
    if path.suffix.lower() != ".mp4":
        raise ValueError("output_path must end in .mp4")
    if not path.parent.is_dir():
        raise ValueError("output_path parent directory must already exist")
    sidecar = path.with_name(path.stem + "_poses.json")
    progress_path = path.with_name(path.stem + "_progress.json")
    if any(os.path.lexists(p) for p in (path, sidecar, progress_path)):
        raise FileExistsError("Refusing to overwrite video, poses or progress sidecar")
    values.update(output_path=path, poses_path=sidecar, progress_path=progress_path,
                  frame_count=max(1, int(math.ceil(values["duration_seconds"] * values["fps"]))))
    return values


def _ego_evaluated_meshes(scene, depsgraph, scale, progress=None):
    """Extract actual world triangles, including hidden obstacles and instances.

    Navigation uses SI metres. No bbox is substituted for collision geometry.
    Moving geometry is rejected because the navigation mesh is a static snapshot.
    """
    import math

    objects = [obj for obj in scene.objects if _ifc_class_for_object(obj) != "IfcDoor"]
    for obj in objects:
        if getattr(obj, "animation_data", None) or getattr(getattr(obj, "data", None), "animation_data", None):
            raise ValueError("Ego navigation requires a static scene (animated objects/data found)")
        if getattr(obj, "rigid_body", None) or any(m.type in {"CLOTH", "SOFT_BODY", "FLUID", "PARTICLE_SYSTEM", "NODES", "MESH_SEQUENCE_CACHE"} for m in getattr(obj, "modifiers", ())):
            raise ValueError("Ego navigation does not support simulated geometry")
        for modifier in getattr(obj, "modifiers", ()):
            if modifier.show_viewport != modifier.show_render:
                raise ValueError("Navigation requires identical viewport/render modifier visibility")
    entries = [(obj.evaluated_get(depsgraph), obj.evaluated_get(depsgraph).matrix_world.copy(), obj) for obj in objects]
    for instance in depsgraph.object_instances:
        if instance.is_instance:
            evaluated = instance.object
            entries.append((evaluated, instance.matrix_world.copy(), evaluated.original))
    meshes = []
    count = 0
    for index, (evaluated, matrix, original) in enumerate(entries):
        if progress is not None:
            progress("scene_geometry", f"Evaluating {original.name}", index, len(entries), triangles=count)
        if _ifc_class_for_object(original) == "IfcDoor":
            continue
        if evaluated.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue
        mesh = None
        try:
            mesh = evaluated.to_mesh()
            if mesh is None:
                continue
            mesh.calc_loop_triangles()
            if count + len(mesh.loop_triangles) > 250000:
                raise ValueError("Navigation mesh exceeds 250,000 triangles; simplify the scene")
            vertices = [tuple(float(v) * scale for v in (matrix @ vertex.co)) for vertex in mesh.vertices]
            if any(not math.isfinite(v) for vertex in vertices for v in vertex):
                raise ValueError("Scene contains non-finite evaluated mesh coordinates")
            triangles = [tuple(vertices[i] for i in tri.vertices) for tri in mesh.loop_triangles]
            count += len(triangles)
            if triangles:
                element = _element_for_object(original)
                meshes.append(dict(name=original.name, ifc_class=_ifc_class_for_object(original),
                                   ifc_id=element.id() if element is not None else None, triangles=triangles))
        finally:
            if mesh is not None:
                evaluated.to_mesh_clear()
    if not meshes:
        raise ValueError("No evaluated mesh geometry available for interior navigation")
    return meshes


def _ego_yaws(positions, fps):
    """Level forward headings; turn during stationary corner holds, at <=60 deg/s."""
    import math

    next_move = [len(positions) - 1] * len(positions)
    for index in range(len(positions) - 2, -1, -1):
        if math.dist(positions[index], positions[index + 1]) > 1e-7:
            next_move[index] = index + 1
        else:
            next_move[index] = next_move[index + 1]
    yaws = []
    for index, position in enumerate(positions):
        stationary = index == 0 or math.dist(positions[index - 1], position) <= 1e-7
        if stationary and next_move[index] > index + 1:
            start, end = position, positions[next_move[index]]
        else:
            # Use the current checked segment, never look ahead across a
            # corner while moving. Stationary holds above turn toward departure.
            start = positions[max(0, index - 1)]
            end = position
            if math.dist(start, end) < 1e-7:
                start, end = position, positions[next_move[index]]
        dx, dy = end[0] - start[0], end[1] - start[1]
        target = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-8 else (yaws[-1] if yaws else 0.0)
        if yaws:
            delta = (target - yaws[-1] + math.pi) % (2 * math.pi) - math.pi
            limit = math.radians(60) / fps
            target = yaws[-1] + max(-limit, min(limit, delta))
        yaws.append(target)
    return yaws


def _ego_linear_camera_animation(action):
    """Keep subframe motion on validated line segments, including Blender 5 slots."""
    curves = list(getattr(action, "fcurves", ()))
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for bag in getattr(strip, "channelbags", ()):
                curves.extend(bag.fcurves)
    if not curves:
        raise RuntimeError("Cannot access camera animation curves for safe linear interpolation")
    seen = set()
    for curve in curves:
        identifier = curve.as_pointer() if hasattr(curve, "as_pointer") else id(curve)
        if identifier in seen:
            continue
        seen.add(identifier)
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"


class _EgoProgress:
    """Rate-limited, atomic progress snapshots; no Blender API or planner state."""

    def __init__(self, path):
        import time

        self.path = path
        self.started = time.monotonic()
        self.last_write = float("-inf")
        self.data = {}
        self.warned = False

    def __enter__(self):
        # Match the movie/poses no-overwrite policy, including stale runs.
        with open(self.path, "x", encoding="utf-8"):
            pass
        self("starting", "Checking scene and render settings", force=True)
        return self

    def __call__(self, stage, message, completed=None, total=None, *, status="running", force=False, **details):
        import contextlib
        import json
        import os
        import tempfile
        import time
        from datetime import datetime, timezone

        now = time.monotonic()
        changed = stage != self.data.get("stage")
        self.data = dict(status=status, stage=stage, message=message,
                         completed=completed, total=total,
                         elapsed_seconds=round(now - self.started, 2),
                         updated_at=datetime.now(timezone.utc).isoformat(), details=details)
        if not (force or changed or now - self.last_write >= 0.5):
            return
        temporary = None
        self.last_write = now
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".ego-progress-", suffix=".json", delete=False) as stream:
                temporary = stream.name
                json.dump(self.data, stream, allow_nan=False)
            os.replace(temporary, self.path)
        except OSError as exc:
            # Reporting failure must not abort an otherwise valid tour/render.
            if not self.warned:
                print(f"[bonsai-mcp-bridge] Progress update failed: {exc}")
                self.warned = True
        finally:
            if temporary is not None and os.path.exists(temporary):
                with contextlib.suppress(OSError):
                    os.unlink(temporary)

    def __exit__(self, exc_type, exc, _traceback):
        if exc_type is None:
            self("completed", "Video and poses saved", 1, 1, status="completed", force=True)
        else:
            self(self.data.get("stage", "starting"), f"{exc_type.__name__}: {exc}",
                 status="failed", force=True, **self.data.get("details", {}))
        return False


def _h_generate_ego_video(params):
    """Render an ego tour with physical IfcDoor objects ignored and hidden."""
    import math

    args = _ego_validate(params)
    with _EgoProgress(args["progress_path"]) as progress:
        scene = bpy.context.scene
        if _get_loaded_ifc() is None:
            raise ValueError("No IFC project is loaded; load a Bonsai project before generating an ego video")
        scale = float(scene.unit_settings.scale_length)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Scene unit scale must be positive and finite")
        if bpy.app.handlers.frame_change_pre or bpy.app.handlers.frame_change_post:
            raise ValueError("Disable frame-change handlers before ego rendering; they can move collision geometry")
        if not getattr(bpy.app.build_options, "codec_ffmpeg", False):
            raise RuntimeError("This Blender build has no FFmpeg/H.264 support. Install a Blender build with FFmpeg enabled.")
        progress("connectivity", "Building deterministic room connectivity and tour")
        semantic = _house_tour(_get_loaded_ifc(), seed=args["seed"])
        components = semantic.get("component_tours", [])
        selected_id = args["component_id"] if args["component_id"] is not None else min((c["component_id"] for c in components), default=None)
        selected = next((c for c in components if c["component_id"] == selected_id), None)
        if selected is None:
            raise ValueError("No eligible tour component for requested component_id")
        return _ego_render_scene(args, scene, scale, semantic, selected, progress)


def _ego_render_scene(args, scene, scale, semantic, selected, progress=None):
    import math
    import tempfile

    from mathutils import Vector

    report = progress or (lambda *args, **kwargs: None)
    report("scene_geometry", "Extracting evaluated scene geometry", component_id=selected.get("component_id"))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = _ego_evaluated_meshes(scene, depsgraph, scale, progress)
    selected_spaces = set(selected["route_space_ids"])
    report("portals", "Resolving actual openings and stair portals")
    semantic["edges"] = _ego_portal_geometry(_get_loaded_ifc(), [e for e in semantic["edges"] if {e["source"], e["target"]} <= selected_spaces], meshes)
    positions, navigation = _ego_tour_plan(meshes, semantic, args["camera_height"], args["frame_count"], args["fps"], args["component_id"], progress)
    args["frame_count"] = len(positions)
    render = scene.render
    saved = []
    camera_obj = camera_data = None
    camera_action = None
    original_frame, original_subframe = scene.frame_current, scene.frame_subframe
    reserved = []
    success = False
    render_progress = None
    def setting(owner, key, value):
        saved.append((owner, key, getattr(owner, key)))
        setattr(owner, key, value)

    # Reserve both paths exclusively, then copy completed outputs into our own
    # files. A competing request cannot overwrite an existing output.
    try:
        if hasattr(render, "use_lock_interface"):
            setting(render, "use_lock_interface", True)
        for path in (args["output_path"], args["poses_path"]):
            with open(path, "xb"):
                pass
            reserved.append(path)
        with tempfile.TemporaryDirectory(prefix=".bonsai-ego-", dir=args["output_path"].parent) as temp:
            movie = os.path.join(temp, "walkthrough.mp4")
            try:
                setting(render, "engine", "BLENDER_WORKBENCH")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Workbench rendering is unavailable in this Blender build") from exc
            for key, value in dict(filepath=movie, resolution_x=args["width"], resolution_y=args["height"],
                                   resolution_percentage=100, fps=args["fps"], fps_base=1.0,
                                   pixel_aspect_x=1.0, pixel_aspect_y=1.0, use_file_extension=True,
                                   use_sequencer=False, use_compositing=False, use_border=False,
                                   use_crop_to_border=False).items():
                setting(render, key, value)
            if hasattr(render, "use_multiview"):
                setting(render, "use_multiview", False)
            setting(render.image_settings, "color_mode", "RGB")
            try:
                # Blender 5.2 gates file_format by media_type. Snapshot the
                # format first so media_type is restored before the old format.
                if hasattr(render.image_settings, "media_type"):
                    saved.append((render.image_settings, "file_format", render.image_settings.file_format))
                    setting(render.image_settings, "media_type", "VIDEO")
                    render.image_settings.file_format = "FFMPEG"
                else:
                    setting(render.image_settings, "file_format", "FFMPEG")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("This Blender version has no native FFmpeg video output. Use a Blender version with MPEG4/H.264 rendering support.") from exc
            try:
                setting(render.ffmpeg, "format", "MPEG4")
                setting(render.ffmpeg, "codec", "H264")
                setting(render.ffmpeg, "audio_codec", "NONE")
                setting(render.ffmpeg, "constant_rate_factor", "MEDIUM")
                setting(render.ffmpeg, "ffmpeg_preset", "GOOD")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("This Blender build cannot encode MPEG4/H.264; use an FFmpeg-enabled Blender build") from exc
            for key, value in dict(light="STUDIO", color_type="MATERIAL", show_shadows=True,
                                   show_cavity=True, show_xray=False).items():
                setting(scene.display.shading, key, value)
            for obj in scene.objects:
                if _ifc_class_for_object(obj) in {"IfcSpace", "IfcOpeningElement", "IfcVirtualElement", "IfcDoor"}:
                    setting(obj, "hide_render", True)
            camera_data = bpy.data.cameras.new("Bonsai Ego Camera")
            camera_obj = bpy.data.objects.new("Bonsai Ego Camera", camera_data)
            scene.collection.objects.link(camera_obj)
            camera_obj.rotation_mode = "QUATERNION"
            camera_data.lens = 20
            camera_data.clip_start = 0.05 / scale
            camera_data.clip_end = 1000 / scale
            setting(scene, "camera", camera_obj)
            setting(scene, "frame_start", 1)
            setting(scene, "frame_end", args["frame_count"])
            setting(scene, "frame_step", 1)
            report("camera", "Creating camera keyframes", 0, args["frame_count"])
            previous_rotation = None
            yaws = _ego_yaws(positions, args["fps"])
            for index, (position, yaw) in enumerate(zip(positions, yaws, strict=True)):
                if index % 25 == 0:
                    report("camera", "Creating camera keyframes", index, args["frame_count"])
                rotation = Vector((math.cos(yaw), math.sin(yaw), 0)).to_track_quat("-Z", "Y")
                if previous_rotation is not None and previous_rotation.dot(rotation) < 0:
                    rotation.negate()
                camera_obj.location = Vector(position) / scale
                camera_obj.rotation_quaternion = rotation
                camera_obj.keyframe_insert(data_path="location", frame=index + 1)
                camera_obj.keyframe_insert(data_path="rotation_quaternion", frame=index + 1)
                previous_rotation = rotation.copy()
            camera_action = camera_obj.animation_data.action
            _ego_linear_camera_animation(camera_action)
            poses = []
            report("poses", "Evaluating camera poses", 0, args["frame_count"])
            for frame in range(1, args["frame_count"] + 1):
                scene.frame_set(frame, subframe=0)
                bpy.context.view_layer.update()
                matrix = camera_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
                quaternion = matrix.to_quaternion().normalized()
                poses.append(dict(frame=frame - 1, blender_frame=frame, timestamp=(frame - 1) / args["fps"],
                                  position=list(matrix.translation), rotation_quaternion=list(quaternion),
                                  space_id=navigation["frame_space_ids"][frame - 1]))
                if frame % 25 == 0 or frame == args["frame_count"]:
                    report("poses", "Evaluating camera poses", frame, args["frame_count"])
            result = dict(success=True, video_path=str(args["output_path"]), poses_path=str(args["poses_path"]),
                          frame_count=args["frame_count"], fps=args["fps"],
                          duration_seconds=args["frame_count"] / args["fps"],
                          requested_duration_seconds=args["duration_seconds"], width=args["width"], height=args["height"],
                          camera_height=args["camera_height"], seed=args["seed"], navigation=navigation,
                          coordinate_system="Blender world; right-handed Z-up; camera forward -Z, up +Y",
                          position_units="scene units", metres_per_scene_unit=scale,
                          quaternion_order="wxyz", frames=poses)
            result.update({key: navigation[key] for key in (
                "tour", "visited_spaces", "tourable_rooms_visited", "tourable_rooms_total",
                "coverage", "route_length_m", "component_id")})
            if "progress_path" in args:
                result["progress_path"] = str(args["progress_path"])
            result["warnings"] = []
            if navigation.get("multi_room_attempted") and len(navigation.get("visited_space_indices", [])) < 2:
                result["warnings"].append("No safely connected multi-room route found; staying in one interior region.")
            if navigation.get("path_length_m", 0) / max(result["duration_seconds"], 0.001) < 0.2:
                result["warnings"].append("Available safe route is short; movement is slower than normal walking.")
            if progress is not None:
                def render_progress(render_scene, *_args):
                    report("rendering", f"Rendered frame {render_scene.frame_current}",
                           render_scene.frame_current - scene.frame_start + 1,
                           args["frame_count"])
                bpy.app.handlers.render_post.append(render_progress)
            report("rendering", "Rendering continuous ego video", 0, args["frame_count"])
            status = bpy.ops.render.render(animation=True, scene=scene.name)
            if "FINISHED" not in status or not os.path.isfile(movie) or os.path.getsize(movie) == 0:
                raise RuntimeError("Workbench H.264 render did not produce a video; check Blender's render log")
            report("saving", "Saving MP4 and camera poses")
            shutil.copyfile(movie, args["output_path"])
            with open(args["poses_path"], "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            success = True
            return {key: value for key, value in result.items() if key != "frames"}
    finally:
        if render_progress is not None:
            with contextlib.suppress(ValueError):
                bpy.app.handlers.render_post.remove(render_progress)
        # Restore every changed property, even if a previous restoration fails.
        for owner, key, value in reversed(saved):
            with contextlib.suppress(Exception):
                setattr(owner, key, value)
        with contextlib.suppress(Exception):
            scene.frame_set(original_frame, subframe=original_subframe)
        if camera_obj is not None:
            if camera_action is None and camera_obj.animation_data:
                camera_action = camera_obj.animation_data.action
            with contextlib.suppress(Exception):
                bpy.data.objects.remove(camera_obj, do_unlink=True)
        if camera_data is not None:
            with contextlib.suppress(Exception):
                bpy.data.cameras.remove(camera_data)
        if camera_action is not None and camera_action.users == 0:
            with contextlib.suppress(Exception):
                bpy.data.actions.remove(camera_action)
        if not success:
            for path in reserved:
                with contextlib.suppress(OSError):
                    path.unlink()


def _house_tour_virtual_portals(ifc, min_width=0.5, min_height=1.8, tolerance=1e-5):
    boundaries = sorted(ifc.by_type("IfcRelSpaceBoundary"), key=lambda item: item.id())
    boundaries = [
        b
        for b in boundaries
        if getattr(b, "PhysicalOrVirtualBoundary", None) == "VIRTUAL"
        and getattr(b, "RelatedBuildingElement", None) is None
    ]
    if not boundaries:
        return []
    import numpy as np
    from ifcopenshell.util.placement import get_axis2placement, get_local_placement
    from ifcopenshell.util.unit import calculate_unit_scale

    scale = calculate_unit_scale(ifc)
    records = []
    for boundary in boundaries:
        if (
            boundary.PhysicalOrVirtualBoundary != "VIRTUAL"
            or boundary.RelatedBuildingElement is not None
        ):
            continue
        try:
            surface = boundary.ConnectionGeometry.SurfaceOnRelatingElement
            if not surface.is_a("IfcSurfaceOfLinearExtrusion"):
                continue
            curve = surface.SweptCurve.Curve
            if not curve.is_a("IfcPolyline") or len(curve.Points) != 2:
                continue
            mat = get_local_placement(boundary.RelatingSpace.ObjectPlacement) @ get_axis2placement(
                surface.Position
            )
            points = [
                np.array(tuple(p.Coordinates) + (0.0,) * (3 - len(p.Coordinates)))
                for p in curve.Points
            ]
            direction = np.array(surface.ExtrudedDirection.DirectionRatios, dtype=float)
            direction /= np.linalg.norm(direction)
            vertices = np.array(
                [
                    (mat @ np.append(v, 1))[:3] * scale
                    for v in points + [v + direction * surface.Depth for v in points[::-1]]
                ]
            )
            u = vertices[1] - vertices[0]
            height = vertices[3] - vertices[0]
            width = np.linalg.norm(u)
            if (
                width <= tolerance
                or abs(u[2]) > tolerance
                or np.linalg.norm(height[:2]) > tolerance
                or height[2] <= tolerance
            ):
                continue
            u /= width
            normal = np.cross(u, [0, 0, 1])
            records.append((boundary, vertices, u, normal))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
    evidence = []
    for index, (a, av, u, n) in enumerate(records):
        for b, bv, _bu, bn in records[index + 1 :]:
            if a.RelatingSpace.id() == b.RelatingSpace.id():
                continue
            # Opposing authored boundary winding, same plane, positive area.
            if np.dot(n, bn) > -1 + 1e-8 or np.max(np.abs((bv - av[0]) @ n)) > tolerance:
                continue
            axis_a = (av - av[0]) @ u
            axis_b = (bv - av[0]) @ u
            lo = max(min(axis_a), min(axis_b))
            hi = min(max(axis_a), max(axis_b))
            bottom = max(min(av[:, 2]), min(bv[:, 2]))
            top = min(max(av[:, 2]), max(bv[:, 2]))
            width = hi - lo
            height = top - bottom
            if width < min_width or height < min_height:
                continue
            evidence.append(
                dict(
                    space_ids=[a.RelatingSpace.id(), b.RelatingSpace.id()],
                    boundary_ids=[a.id(), b.id()],
                    kind="paired_virtual_boundary_polygon_overlap",
                    width_m=float(width),
                    height_m=float(height),
                    area_m2=float(width * height),
                )
            )
    return evidence


def _house_tour_triangle_overlap_area_xy(a, b):
    """Convex clipping of actual projected triangles, not AABBs."""

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    import numpy as np

    a = np.asarray(a)[:, :2]
    b = np.asarray(b)[:, :2]
    if cross(b[1] - b[0], b[2] - b[0]) < 0:
        b = b[::-1]
    polygon = list(a)
    for i in range(3):
        start, end = b[i], b[(i + 1) % 3]
        output = []
        if not polygon:
            return 0.0
        previous = polygon[-1]
        pd = cross(end - start, previous - start)
        for current in polygon:
            cd = cross(end - start, current - start)
            if (cd >= 0) != (pd >= 0):
                output.append(previous + (current - previous) * (pd / (pd - cd)))
            if cd >= 0:
                output.append(current)
            previous, pd = current, cd
        polygon = output
    if len(polygon) < 3:
        return 0.0
    return (
        abs(sum(cross(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))))
        / 2
    )


def _house_tour_stair_endpoints(ifc, max_step=0.25, min_tread_area=0.05, min_overlap_area=0.002, storey_lookup=None):
    """Read actual Body meshes; match extreme substantial treads to space floors.

    No bbox/nearest-room evidence. Require unique matches, distinct spaces,
    and distinct storeys. This establishes semantic adjacency only, not a
    clearance-checked walk path. Thin riser caps are not walking treads.
    Unsupported/ambiguous flights return diagnostic records, never edges.
    """
    if not ifc.by_type("IfcStairFlight"):
        return [], []
    import ifcopenshell.geom
    import numpy as np

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    def triangles(product):
        body = next(
            (
                r
                for r in product.Representation.Representations
                if r.RepresentationIdentifier == "Body"
            ),
            None,
        )
        if body is None:
            raise ValueError("missing Body representation")
        shape = ifcopenshell.geom.create_shape(settings, product, body)
        geometry = shape.geometry
        return np.array(geometry.verts).reshape(-1, 3)[np.array(geometry.faces).reshape(-1, 3)]

    def horizontal_groups(mesh, up):
        groups = {}
        for t in mesh:
            normal = np.cross(t[1] - t[0], t[2] - t[0])
            if np.ptp(t[:, 2]) > 1e-5 or (normal[2] if up else -normal[2]) <= 1e-8:
                continue
            z = round(float(t[0, 2]), 5)
            groups.setdefault(z, []).append(t)
        return groups

    def storey(space):
        if storey_lookup is not None:
            parent = storey_lookup(space)
            return parent.id() if parent is not None else None
        parents = [r.RelatingObject for r in space.Decomposes]
        return next((p.id() for p in parents if p.is_a("IfcBuildingStorey")), None)

    floors = []
    diagnostics = []
    for space in sorted(ifc.by_type("IfcSpace"), key=lambda item: item.id()):
        try:
            groups = horizontal_groups(triangles(space), False)
            if not groups:
                continue
            # Only lowest floor: do not confuse roof/ceiling bottom faces with floor.
            z = min(groups)
            floors.append((space, z, groups[z]))
        except Exception as error:
            diagnostics.append(
                dict(
                    space_id=space.id(), reason="space_body_geometry_unavailable", detail=str(error)
                )
            )
    evidence = []
    for flight in sorted(ifc.by_type("IfcStairFlight"), key=lambda item: item.id()):
        try:
            groups = horizontal_groups(triangles(flight), True)
            groups = {
                z: ts
                for z, ts in groups.items()
                if sum(np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) / 2 for t in ts)
                >= min_tread_area
            }
            if len(groups) < 2:
                raise ValueError("fewer than two substantial horizontal treads")
            endpoints = []
            for z in [min(groups), max(groups)]:
                matches = []
                for space, floor_z, floor_triangles in floors:
                    if abs(floor_z - z) > max_step:
                        continue
                    overlap = sum(
                        _house_tour_triangle_overlap_area_xy(t, f)
                        for t in groups[z]
                        for f in floor_triangles
                    )
                    if overlap >= min_overlap_area:
                        matches.append(
                            dict(
                                space_id=space.id(),
                                storey_id=storey(space),
                                tread_z_m=z,
                                floor_z_m=floor_z,
                                overlap_area_m2=float(overlap),
                            )
                        )
                endpoints.append(matches)
            if any(len(m) != 1 for m in endpoints):
                diagnostics.append(
                    dict(
                        flight_id=flight.id(),
                        reason="ambiguous_or_unmatched_stair_endpoints",
                        endpoint_candidates=endpoints,
                    )
                )
                continue
            lo, hi = endpoints[0][0], endpoints[1][0]
            if (
                lo["space_id"] == hi["space_id"]
                or lo["storey_id"] is None
                or hi["storey_id"] is None
                or lo["storey_id"] == hi["storey_id"]
            ):
                diagnostics.append(
                    dict(
                        flight_id=flight.id(),
                        reason="endpoints_not_distinct_storeys",
                        endpoint_candidates=endpoints,
                    )
                )
                continue
            evidence.append(
                dict(
                    kind="stair_tread_space_floor_polygon_overlap",
                    flight_id=flight.id(),
                    space_ids=[lo["space_id"], hi["space_id"]],
                    endpoints=[lo, hi],
                )
            )
        except Exception as error:
            diagnostics.append(
                dict(flight_id=flight.id(), reason="unsupported_stair_geometry", detail=str(error))
            )
    return evidence, diagnostics


def _house_tour_door_opening_evidence(ifc, door, candidate_spaces=None, mesh_loader=None):
    """Return (one pair evidence or None, diagnostic records).

    candidate_spaces must contain *all* semantic candidates, not a desired pair.
    Omit it to test all spaces. mesh_loader optionally returns (vertices, faces)
    in world metres, and allows callers to cache native Body tessellations.
    """
    ids = {"door_id": door.id()}
    diagnostics = []

    def reject(reason, **extra):
        diagnostics.append(dict(ids, reason=reason, **extra))
        return None, diagnostics

    try:
        from collections import Counter

        import numpy as np

        if mesh_loader is None:
            import ifcopenshell.geom

            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)

            def mesh_loader(product):
                body = next(
                    (
                        r
                        for r in product.Representation.Representations
                        if r.RepresentationIdentifier == "Body"
                    ),
                    None,
                )
                if body is None:
                    raise ValueError("missing Body representation")
                shape = ifcopenshell.geom.create_shape(settings, product, body)
                return shape.geometry.verts, shape.geometry.faces
    except Exception as error:
        return reject("geometry_backend_unavailable", detail=str(error))

    def mesh(product):
        verts, faces = mesh_loader(product)
        verts = np.asarray(verts, dtype=float).reshape(-1, 3)
        faces = np.asarray(faces, dtype=int).reshape(-1, 3)
        if not len(faces) or not np.isfinite(verts).all():
            raise ValueError("empty or nonfinite Body mesh")
        # Merge duplicate coordinates before checking closed manifold topology.
        _, inv = np.unique(np.round(verts, 7), axis=0, return_inverse=True)
        edges = Counter(
            tuple(sorted((int(a), int(b))))
            for f in inv[faces]
            for a, b in zip(f, np.roll(f, -1), strict=False)
        )
        if any(n != 2 for n in edges.values()):
            raise ValueError("Body mesh is not a closed two-manifold")
        return verts, verts[faces]

    def inside(point, triangles):
        a, b, c = np.moveaxis(triangles - point, 1, 0)
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        winding = abs(float(np.sum(2 * np.arctan2(numerator, denominator))))
        return abs(winding - 4 * np.pi) < 1e-4

    def floor_at(point, triangles, bottom):
        for t in triangles:
            if np.ptp(t[:, 2]) > 1e-5 or abs(t[0, 2] - bottom) > 0.25:
                continue
            normal = np.cross(t[1] - t[0], t[2] - t[0])
            if normal[2] >= -1e-8:
                continue
            a, b, c = t[:, :2]
            q = point[:2]

            def cross(x, y):
                return x[0] * y[1] - x[1] * y[0]

            signs = [cross(b - a, q - a), cross(c - b, q - b), cross(a - c, q - c)]
            if max(signs) <= 1e-8 or min(signs) >= -1e-8:
                return float(t[0, 2])
        return None

    try:
        openings = list(door.FillsVoids)
        if len(openings) != 1:
            return reject("door_not_one_filled_opening")
        opening = openings[0].RelatingOpeningElement
        ids["opening_id"] = opening.id()
        hosts = list(opening.VoidsElements)
        if len(hosts) != 1 or not hosts[0].RelatingBuildingElement.is_a("IfcWall"):
            return reject("opening_not_one_host_wall")
        host = hosts[0].RelatingBuildingElement
        ids["host_wall_id"] = host.id()
        ov, ot = mesh(opening)
        hv, ht = mesh(host)
        # Recover basis from real horizontal opening mesh edges, not world AABB.
        directions = []
        for t in ot:
            for a, b in zip(t, np.roll(t, -1, axis=0), strict=False):
                delta = b - a
                length = np.linalg.norm(delta)
                if abs(delta[2]) < 1e-6 and length > 1e-6:
                    directions.append(delta / length)
        basis = None
        for u in directions:
            n = np.cross(u, [0.0, 0.0, 1.0])
            projected = np.column_stack((ov @ u, ov @ n, ov[:, 2]))
            lo = projected.min(0)
            hi = projected.max(0)
            sizes = hi - lo
            # All vertices must be at the eight corners of an orthogonal prism.
            corner = np.minimum(abs(projected - lo), abs(projected - hi))
            if (
                (corner < 1e-5).all()
                and sizes[0] >= 0.5
                and 0.03 <= sizes[1] <= 0.8
                and sizes[2] >= 1.8
                and sizes[0] > sizes[1] * 1.5
            ):
                basis = (u, n, lo, hi)
                break
        if basis is None:
            return reject("unsupported_nonrectangular_or_small_opening")
        u, n, lo, hi = basis
        width, depth, height = hi - lo

        def world(a, b, z):
            return u * a + n * b + np.array([0.0, 0.0, z])

        center = (lo + hi) / 2
        # At least one actual wall jamb must surround the opening at mid-height.
        jambs = [world(a, center[1], center[2]) for a in (lo[0] - 0.03, hi[0] + 0.03)]
        jamb_hits = [inside(q, ht) for q in jambs]
        if not any(jamb_hits):
            return reject("host_wall_has_no_solid_jamb", jamb_probes_m=[q.tolist() for q in jambs])
        spaces = (
            list(candidate_spaces)
            if candidate_spaces is not None
            else list(ifc.by_type("IfcSpace"))
        )
        loaded = []
        for space in spaces:
            try:
                _, ts = mesh(space)
                loaded.append((space, ts))
            except Exception as error:
                # Missing any candidate can hide a competing match: never guess.
                return reject(
                    "candidate_space_geometry_unavailable", space_id=space.id(), detail=str(error)
                )
        side_matches = []
        side_records = []
        for side, offset in ((-1, lo[1] - 0.02), (1, hi[1] + 0.02)):
            xy = [world(lo[0] + width * f, offset, lo[2]) for f in (0.2, 0.5, 0.8)]
            probes = [q + np.array([0.0, 0.0, z]) for q in xy for z in (0.3, 0.9, 1.7)]
            matches = []
            for space, ts in loaded:
                count = sum(inside(q, ts) for q in probes)
                floors = [floor_at(q, ts, lo[2]) for q in xy]
                valid = count == len(probes) and all(z is not None for z in floors)
                diagnostics.append(
                    dict(
                        ids,
                        space_id=space.id(),
                        side=side,
                        reason="matched_opening_side" if valid else "rejected_opening_side",
                        volume_probe_hits=count,
                        volume_probe_count=len(probes),
                        floor_z_m=floors,
                    )
                )
                # Partial volume intersections are ambiguous, even if another space passes.
                if count and not valid:
                    return reject(
                        "partial_or_unsupported_space_side", space_id=space.id(), side=side
                    )
                if valid:
                    matches.append(space.id())
            side_matches.append(matches)
            side_records.append(
                dict(side=side, space_ids=matches, probes_m=[q.tolist() for q in probes])
            )
        if any(len(m) != 1 for m in side_matches) or side_matches[0] == side_matches[1]:
            return reject("opening_sides_not_two_unique_spaces", sides=side_records)
        evidence = dict(
            ids,
            kind="door_opening_host_wall_space_volume_sides",
            space_ids=sorted([m[0] for m in side_matches]),
            width_m=float(width),
            height_m=float(height),
            thickness_m=float(depth),
            sides=side_records,
            host_jamb_hits=jamb_hits,
            fills_voids_relation_id=openings[0].id(),
            voids_element_relation_id=hosts[0].id(),
            note="Sampled geometric adjacency only; not a traversability or collision-path guarantee",
        )
        return evidence, diagnostics
    except Exception as error:
        return reject("unsupported_door_geometry", detail=str(error))


def _house_tour_wall_separation_evidence(ifc, side_a_ids, side_b_ids):
    """Return shared-wall relations, not a proof of disconnected physical units.

    Shared facade walls also qualify. Geometry and wall role must be inspected.
    Absence of openings is only absence in this export.
    """
    rows = []
    a, b = set(side_a_ids), set(side_b_ids)
    for wall in ifc.by_type("IfcWall"):
        boundaries = [r for r in ifc.get_inverse(wall) if r.is_a("IfcRelSpaceBoundary")]
        members = {r.RelatingSpace.id() for r in boundaries}
        if not members.intersection(a) or not members.intersection(b):
            continue
        rows.append(
            {
                "wall_id": wall.id(),
                "name": wall.Name,
                "side_a_space_ids": sorted(members.intersection(a)),
                "side_b_space_ids": sorted(members.intersection(b)),
                "boundary_ids": [r.id() for r in boundaries],
                "openings": [
                    {
                        "opening_id": r.RelatedOpeningElement.id(),
                        "fillings": [
                            {
                                "id": f.RelatedBuildingElement.id(),
                                "type": f.RelatedBuildingElement.is_a(),
                            }
                            for f in r.RelatedOpeningElement.HasFillings
                        ],
                    }
                    for r in wall.HasOpenings
                ],
                "limitation": "Shared wall is not necessarily a dividing wall; verify geometry and role.",
            }
        )
    return rows


def _house_tour_physical_unit_separation(ifc, side_a_ids, side_b_ids, tolerance=0.03):
    """Conservative, model-evidenced internal-unit separation, not world proof.

    Requires distinct exported occupancy-zone strings, a demising-role wall,
    spaces touching opposite geometric wall faces, no hosted wall voids, and
    distinct external doors. Missing evidence returns unproven, never merged.
    """
    import ifcopenshell.geom
    from ifcopenshell.util.element import get_psets

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    cache = {}
    vertex_cache = {}
    face_cache = {}

    def bounds(entity):
        if entity.id() not in cache:
            rep = next(
                (
                    r
                    for r in entity.Representation.Representations
                    if r.RepresentationIdentifier == "Body"
                ),
                None,
            )
            if rep is None:
                raise ValueError("No Body representation")
            shape = ifcopenshell.geom.create_shape(settings, entity, rep)
            verts = shape.geometry.verts
            vertex_cache[entity.id()] = verts
            face_cache[entity.id()] = shape.geometry.faces
            cache[entity.id()] = (
                [min(verts[i::3]) for i in range(3)],
                [max(verts[i::3]) for i in range(3)],
            )
        return cache[entity.id()]

    def zones(ids):
        values = set()
        for sid in ids:
            found = {
                str(v)
                for p in get_psets(ifc.by_id(sid)).values()
                if isinstance(p, dict)
                for k, v in p.items()
                if k.lower().replace(" ", "") == "occupancyzonename" and v
            }
            if not found:
                return set()
            values.update(found)
        return values

    def exterior_doors(ids):
        result = set()
        for sid in ids:
            for rel in getattr(ifc.by_id(sid), "BoundedBy", ()):
                element = rel.RelatedBuildingElement
                if (
                    element
                    and element.is_a("IfcDoor")
                    and rel.InternalOrExternalBoundary == "EXTERNAL"
                ):
                    result.add(element.id())
        return sorted(result)

    za, zb = zones(side_a_ids), zones(side_b_ids)
    da, db = exterior_doors(side_a_ids), exterior_doors(side_b_ids)
    result = {
        "status": "unproven",
        "occupancy_zones_a": sorted(za),
        "occupancy_zones_b": sorted(zb),
        "external_doors_a": da,
        "external_doors_b": db,
        "dividing_walls": [],
        "errors": [],
        "limitations": [
            "Evidence supports separate internal touring units, not global physical disconnection.",
            "No hosted voids means none exported in the audited wall, not proof of IFC completeness.",
            "Opposite-face test uses Body world AABBs; only axis-aligned thin walls qualify.",
            "Occupancy zone strings and demising role are exporter metadata, not legal unit certification.",
        ],
    }
    for row in _house_tour_wall_separation_evidence(ifc, side_a_ids, side_b_ids):
        wall = ifc.by_id(row["wall_id"])
        role = " ".join(
            str(x or "") for x in (wall.Name, getattr(wall, "ObjectType", None))
        ).lower()
        if not any(word in role for word in ("party wall", "demising", "dimising")):
            continue
        if row["openings"]:
            continue
        try:
            low, high = bounds(wall)
            axis = min((0, 1), key=lambda i: high[i] - low[i])
            other = 1 - axis
            # A thin diagonal AABB is not a planar axis-aligned divider.
            wall_vertices = vertex_cache[wall.id()]
            wall_faces = face_cache[wall.id()]
            aligned = True
            for k in range(0, len(wall_faces), 3):
                points = [wall_vertices[3 * idx : 3 * idx + 3] for idx in wall_faces[k : k + 3]]
                u = [points[1][i] - points[0][i] for i in range(3)]
                v = [points[2][i] - points[0][i] for i in range(3)]
                nx = u[1] * v[2] - u[2] * v[1]
                ny = u[2] * v[0] - u[0] * v[2]
                horizontal_norm = (nx * nx + ny * ny) ** 0.5
                if horizontal_norm > 1e-10 and min(abs(nx), abs(ny)) / horizontal_norm > 1e-4:
                    aligned = False
                    break
            if not aligned:
                continue
            if high[axis] - low[axis] >= 0.25 * (high[other] - low[other]):
                continue

            def face(sid, low=low, high=high, axis=axis, other=other):
                sl, sh = bounds(ifc.by_id(sid))
                overlap = all(
                    min(sh[i], high[i]) - max(sl[i], low[i]) > tolerance for i in (other, 2)
                )
                if not overlap:
                    return None
                if abs(sh[axis] - low[axis]) <= tolerance:
                    return "low"
                if abs(sl[axis] - high[axis]) <= tolerance:
                    return "high"
                return None

            fa = {sid: face(sid) for sid in row["side_a_space_ids"]}
            fb = {sid: face(sid) for sid in row["side_b_space_ids"]}
            pairs = [
                [a, b] for a, va in fa.items() for b, vb in fb.items() if va and vb and va != vb
            ]
            if pairs:
                result["dividing_walls"].append(
                    dict(
                        row,
                        bbox=[low, high],
                        normal_axis=axis,
                        opposite_face_pairs=pairs,
                        method="demising_role_body_opposite_faces_no_voids",
                    )
                )
        except Exception as exc:
            result["errors"].append({"wall_id": wall.id(), "error": str(exc)})
    if (
        za
        and zb
        and za.isdisjoint(zb)
        and da
        and db
        and set(da).isdisjoint(db)
        and result["dividing_walls"]
    ):
        result["status"] = "physically_separate"
        result["confidence"] = "evidence_supported_internal_unit_separation"
    return result


def _house_tour_component_separation_evidence(ifc, component_ids_list):
    """Audit every pair of tour components without hardcoded space/unit IDs."""
    pairs = []
    for i, side_a in enumerate(component_ids_list):
        for j in range(i + 1, len(component_ids_list)):
            evidence = _house_tour_physical_unit_separation(ifc, side_a, component_ids_list[j])
            evidence["component_a_index"] = i
            evidence["component_b_index"] = j
            pairs.append(evidence)
    return {
        "status": "physically_separate"
        if pairs and all(pair["status"] == "physically_separate" for pair in pairs)
        else "unresolved"
        if pairs
        else "single_component",
        "pairs": pairs,
    }


def _house_tour_classifications(ifc, spaces):
    """Classify all spaces; IFC metadata outranks descriptive name hints."""
    import re

    properties = {space.id(): [] for space in spaces}
    try:
        relations = list(ifc.by_type("IfcRelDefinesByProperties"))
    except RuntimeError:
        relations = []
    for space in spaces:
        for rel in getattr(space, "IsDefinedBy", ()) or ():
            if rel not in relations:
                relations.append(rel)
    for rel in relations:
        pset = getattr(rel, "RelatingPropertyDefinition", None)
        for prop in getattr(pset, "HasProperties", ()) or ():
            value = getattr(prop, "NominalValue", None)
            value = getattr(value, "wrappedValue", value)
            for space in getattr(rel, "RelatedObjects", ()) or ():
                if space.id() in properties:
                    properties[space.id()].append(
                        (
                            str(getattr(pset, "Name", "")) + "." + str(getattr(prop, "Name", "")),
                            value,
                            rel.id(),
                        )
                    )

    def usage(value):
        words = set(re.findall(r"[a-z]+", str(value).lower()))
        if words & {
            "corridor",
            "hallway",
            "circulation",
            "stair",
            "stairs",
            "stairwell",
            "stairway",
            "foyer",
            "landing",
            "lobby",
        }:
            return "circulation"
        if words & {
            "roof",
            "rooftop",
            "exterior",
            "outdoor",
            "balcony",
            "terrace",
            "void",
            "shaft",
        }:
            return "non_tourable"
        if words & {
            "utility",
            "bedroom",
            "bathroom",
            "kitchen",
            "living",
            "dining",
            "office",
            "study",
            "laundry",
            "closet",
            "storage",
            "toilet",
            "garage",
        }:
            return "tourable_interior"
        return None

    result = {}
    for space in spaces:
        evidence, categories, internal = [], set(), []
        service_use, occupied_use = False, False
        values = [
            (attr, getattr(space, attr, None), None)
            for attr in ("InteriorOrExteriorSpace", "PredefinedType", "ObjectType")
        ] + properties[space.id()]
        for field, value, rid in values:
            if value is None:
                continue
            token = str(value).upper()
            key = field.rsplit(".", 1)[-1].lower()
            if key not in {
                "interiororexteriorspace",
                "predefinedtype",
                "objecttype",
                "isexternal",
                "category",
                "category description",
                "category code",
                "omniclass table 13 category",
                "occupancytype",
                "spaceusage",
                "spacetype",
                "function",
                "isaccessible",
                "isoccupied",
                "accessible",
                "occupied",
            }:
                continue
            category = None
            if key in {"category description", "omniclass table 13 category"}:
                service_use = service_use or "general facility service spaces" in str(value).lower()
            if key in {"isaccessible", "isoccupied", "accessible", "occupied"} and token in {
                "TRUE",
                "1",
            }:
                occupied_use = True
            if key in {"occupancytype", "spaceusage", "function", "objecttype"} and (
                usage(value) in {"tourable_interior", "circulation"}
                or "terrace" in str(value).lower()
                or "occupied" in str(value).lower()
            ):
                occupied_use = True
            if key == "isexternal" and token in {"TRUE", "FALSE", "1", "0"}:
                internal.append(token in {"FALSE", "0"})
            elif key in {"interiororexteriorspace", "predefinedtype"} and token in {
                "INTERNAL",
                "EXTERNAL",
                "EXTERNAL_EARTH",
                "EXTERNAL_WATER",
                "EXTERNAL_FIRE",
            }:
                internal.append(token == "INTERNAL")
            elif key in {
                "objecttype",
                "predefinedtype",
                "category",
                "category description",
                "omniclass table 13 category",
                "occupancytype",
                "spaceusage",
                "spacetype",
                "function",
            }:
                category = usage(value)
            if category:
                categories.add(category)
            evidence.append(
                {
                    "source": "ifc_metadata",
                    "field": field,
                    "value": str(value),
                    "relationship_id": rid,
                }
            )
        if False in internal:
            categories.add("non_tourable")
        conflict = len(categories) > 1 or (True in internal and False in internal)
        if conflict:
            classification, reason = "unknown", "conflicting_ifc_metadata"
        elif categories:
            classification, reason = (
                next(iter(categories)),
                "explicit_ifc_usage_or_exterior_metadata",
            )
        else:
            names = [getattr(space, "Name", None), getattr(space, "LongName", None)]
            hints = {usage(name) for name in names if name}
            hints.discard(None)
            evidence.extend({"source": "name_hint", "value": str(name)} for name in names if name)
            if len(hints) == 1 and "non_tourable" not in hints:
                classification, reason = next(iter(hints)), "name_hint_without_conflicting_metadata"
            else:
                classification, reason = (
                    "unknown",
                    "insufficient_or_ambiguous_interior_usage_evidence",
                )
        policy_exclusion = False
        if service_use and not conflict:
            # Combined authored use + roof role + no portal evidence is a touring
            # policy, never a claim that the roof is physically inaccessible.
            parents = []
            try:
                for rel in ifc.by_type("IfcRelAggregates"):
                    if space in (getattr(rel, "RelatedObjects", ()) or ()):
                        parent = getattr(rel, "RelatingObject", None)
                        if parent is not None:
                            parents.append(parent)
            except RuntimeError:
                pass
            roof_role = any(
                "roof" in str(getattr(parent, "Name", "")).lower() for parent in parents
            )
            boundaries = [
                rel
                for rel in ifc.by_type("IfcRelSpaceBoundary")
                if getattr(rel, "RelatingSpace", None) == space
            ]
            portal = any(
                getattr(rel, "PhysicalOrVirtualBoundary", None) == "VIRTUAL"
                or (
                    getattr(rel, "RelatedBuildingElement", None) is not None
                    and rel.RelatedBuildingElement.is_a()
                    in {"IfcDoor", "IfcOpeningElement", "IfcStair", "IfcStairFlight"}
                )
                for rel in boundaries
            )
            wall_only = bool(boundaries) and all(
                getattr(rel, "RelatedBuildingElement", None) is not None
                and (
                    rel.RelatedBuildingElement.is_a("IfcWall")
                    or rel.RelatedBuildingElement.is_a("IfcWallStandardCase")
                )
                and getattr(rel, "InternalOrExternalBoundary", None) == "EXTERNAL"
                for rel in boundaries
            )
            evidence.append(
                {
                    "source": "service_roof_policy",
                    "roof_storey_ids": sorted(
                        parent.id()
                        for parent in parents
                        if "roof" in str(getattr(parent, "Name", "")).lower()
                    ),
                    "boundary_ids": sorted(rel.id() for rel in boundaries),
                    "external_wall_boundaries_only": wall_only,
                    "portal_evidenced": portal,
                    "occupied_or_accessible_use_evidenced": occupied_use,
                }
            )
            if occupied_use:
                classification, reason = (
                    "tourable_interior",
                    "explicit_occupied_or_accessible_use_overrides_service_roof_policy",
                )
            elif roof_role and wall_only and not portal:
                classification, reason = (
                    "non_tourable",
                    "conservative_service_roof_touring_policy_not_proven_inaccessible",
                )
                policy_exclusion = True
            else:
                classification, reason = "unknown", "service_space_usage_requires_review"
        result[space.id()] = {
            "classification": classification,
            "reason": reason,
            "evidence": evidence,
            "policy_exclusion": policy_exclusion,
            "confidence": "policy_exclusion_not_physical_accessibility"
            if policy_exclusion
            else "metadata_or_label_evidence_not_accessibility_proof",
        }
    return result


def _house_tour(ifc, start_space_id=None, seed=0):
    """Read-only IFC topology. Connectivity is not a collision/passability claim."""
    import random
    from collections import defaultdict, deque

    if ifc is None:
        raise RuntimeError("No IFC project is loaded. Open an IFC file in Bonsai first.")
    if start_space_id is not None and (type(start_space_id) is not int or start_space_id <= 0):
        raise ValueError("start_space_id must be a positive integer or null.")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("seed must be an integer from 0 to 2147483647.")

    def entities(kind):
        # Some optional relationship types do not exist in older IFC schemas.
        try:
            return sorted(ifc.by_type(kind), key=lambda x: x.id())
        except RuntimeError:
            return []

    spaces = entities("IfcSpace")
    if not spaces:
        raise ValueError("The loaded IFC contains no IfcSpace entities; a house tour needs spaces.")
    space_ids = {x.id() for x in spaces}
    if start_space_id is not None and start_space_id not in space_ids:
        raise ValueError("start_space_id does not identify an IfcSpace in the loaded IFC.")
    parents = defaultdict(list)
    children = defaultdict(list)
    for kind, parent_attr, child_attr in [
        ("IfcRelAggregates", "RelatingObject", "RelatedObjects"),
        ("IfcRelNests", "RelatingObject", "RelatedObjects"),
        ("IfcRelContainedInSpatialStructure", "RelatingStructure", "RelatedElements"),
    ]:
        for rel in entities(kind):
            parent = getattr(rel, parent_attr, None)
            if parent is not None:
                for child in getattr(rel, child_attr, ()) or ():
                    parents[child.id()].append((parent, rel.id()))
                    children[parent.id()].append((child, rel.id()))

    def storey_of(entity):
        todo, seen = deque([entity]), set()
        while todo:
            item = todo.popleft()
            if item.id() in seen:
                continue
            seen.add(item.id())
            if item.is_a("IfcBuildingStorey"):
                return item
            todo.extend(p for p, _ in sorted(parents[item.id()], key=lambda x: x[0].id()))
        return None

    classifications = _house_tour_classifications(ifc, spaces)
    nodes = []
    for space in spaces:
        storey = storey_of(space)
        nodes.append(
            {
                "id": space.id(),
                "name": getattr(space, "Name", None),
                "long_name": getattr(space, "LongName", None),
                **classifications[space.id()],
                "storey_id": storey.id() if storey else None,
                "storey_name": getattr(storey, "Name", None) if storey else None,
            }
        )
    connectors = {
        x.id(): x
        for kind in ("IfcDoor", "IfcOpeningElement", "IfcStair", "IfcStairFlight")
        for x in entities(kind)
    }
    boundary_spaces = defaultdict(set)
    boundary_evidence = defaultdict(list)
    for rel in entities("IfcRelSpaceBoundary"):
        space = getattr(rel, "RelatingSpace", None)
        element = getattr(rel, "RelatedBuildingElement", None)
        if space is not None and space.id() in space_ids and element is not None:
            boundary_spaces[element.id()].add(space.id())
            boundary_evidence[element.id()].append(rel.id())
    fills = defaultdict(list)
    for rel in entities("IfcRelFillsElement"):
        opening = getattr(rel, "RelatingOpeningElement", None)
        filling = getattr(rel, "RelatedBuildingElement", None)
        if opening is not None and filling is not None:
            fills[opening.id()].append((filling, rel.id()))
    voids = defaultdict(list)
    for rel in entities("IfcRelVoidsElement"):
        opening = getattr(rel, "RelatedOpeningElement", None)
        if opening is not None:
            voids[opening.id()].append(rel.id())

    edges, unresolved, processed = [], [], set()
    adjacency = {sid: set() for sid in space_ids}

    def record(connector, members, relation_ids, connector_ids, kind):
        members = sorted(members)
        evidence = {
            "method": "explicit_ifc_relationships",
            "relationship_ids": sorted(set(relation_ids)),
            "connector_ids": sorted(set(connector_ids)),
        }
        if len(members) > 2 and kind == "door":
            try:
                geometric, diagnostics = _house_tour_door_opening_evidence(ifc, connector)
            except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
                geometric, diagnostics = (
                    None,
                    [{"reason": "door_geometry_unavailable", "detail": str(exc)}],
                )
            evidence["geometry_diagnostics"] = diagnostics
            if geometric and set(geometric["space_ids"]) <= set(members):
                evidence["original_candidate_space_ids"] = members
                evidence["geometric_resolution"] = geometric
                evidence["method"] = (
                    "explicit_candidates_resolved_by_opening_host_and_space_volumes"
                )
                members = sorted(geometric["space_ids"])
        if len(members) != 2:
            unresolved.append(
                {
                    "connector_id": connector.id(),
                    "ifc_class": connector.is_a(),
                    "space_ids": members,
                    "evidence": evidence,
                    "reason": "ambiguous_multiple_spaces"
                    if len(members) > 2
                    else "fewer_than_two_explicit_spaces",
                }
            )
            return
        a, b = members
        edges.append(
            {
                "source": a,
                "target": b,
                "kind": kind,
                "connector_id": connector.id(),
                "evidence": evidence,
                "passability": "not_verified",
            }
        )
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Normalize a filled opening and its door to one portal. Never use the
    # host wall's boundaries: a wall shared by spaces does not prove a doorway.
    for opening in entities("IfcOpeningElement"):
        filling_rels = fills[opening.id()]
        processed.add(opening.id())
        doors = [f for f, _ in filling_rels if f.is_a("IfcDoor")]
        if filling_rels and (len(doors) != len(filling_rels) or len(doors) != 1):
            unresolved.append(
                {
                    "connector_id": opening.id(),
                    "ifc_class": opening.is_a(),
                    "space_ids": sorted(boundary_spaces[opening.id()]),
                    "reason": "window_or_other_non_door_filling",
                }
            )
            continue
        group = [opening] + doors
        ids = [x.id() for x in group]
        processed.update(ids)
        members = set().union(*(boundary_spaces[i] for i in ids))
        evidence = [r for i in ids for r in boundary_evidence[i]]
        evidence += [r for _, r in filling_rels] + voids[opening.id()]
        record(
            doors[0] if doors else opening, members, evidence, ids, "door" if doors else "opening"
        )
    for cid, connector in sorted(connectors.items()):
        if cid in processed:
            continue
        members = set(boundary_spaces[cid])
        evidence = list(boundary_evidence[cid])
        ids = [cid]
        stair = connector.is_a("IfcStair") or connector.is_a("IfcStairFlight")
        if stair:
            # A stair may aggregate flights. Only explicit space relationships
            # on those parts count, never all rooms on the stair's storey.
            todo, seen = deque([connector]), set()
            while todo:
                part = todo.popleft()
                if part.id() in seen:
                    continue
                seen.add(part.id())
                ids.append(part.id())
                members.update(boundary_spaces[part.id()])
                evidence.extend(boundary_evidence[part.id()])
                for parent, rid in parents[part.id()]:
                    if parent.id() in space_ids:
                        members.add(parent.id())
                        evidence.append(rid)
                for child, rid in children[part.id()]:
                    if child.is_a("IfcStairFlight") or child.is_a("IfcStair"):
                        todo.append(child)
                        evidence.append(rid)
        record(connector, members, evidence, ids, "stair" if stair else "door")

    # Two opposed, overlapping authored VIRTUAL boundary surfaces provide
    # explicit open-portal evidence. This is not proximity of object bounds.
    virtual_boundaries = [
        b
        for b in entities("IfcRelSpaceBoundary")
        if getattr(b, "PhysicalOrVirtualBoundary", None) == "VIRTUAL"
        and getattr(b, "RelatedBuildingElement", None) is None
    ]
    virtual_error = None
    try:
        virtual_portals = _house_tour_virtual_portals(ifc)
    except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
        virtual_portals = []
        virtual_error = str(exc)
    partners = defaultdict(set)
    for portal in virtual_portals:
        for boundary_id in portal["boundary_ids"]:
            partners[boundary_id].add(tuple(sorted(portal["space_ids"])))
    resolved_boundaries = set()
    virtual_edges = {}
    node_storeys = {n["id"]: n["storey_id"] for n in nodes}
    for portal in virtual_portals:
        a, b = sorted(portal["space_ids"])
        if (
            a not in space_ids
            or b not in space_ids
            or any(len(partners[bid]) != 1 for bid in portal["boundary_ids"])
        ):
            continue
        if node_storeys[a] is None or node_storeys[a] != node_storeys[b]:
            continue
        evidence = dict(portal)
        evidence["method"] = portal["kind"]
        evidence["relationship_ids"] = sorted(portal["boundary_ids"])
        if (a, b) in virtual_edges:
            virtual_edges[a, b]["evidence"]["patches"].append(evidence)
            virtual_edges[a, b]["evidence"]["relationship_ids"] = sorted(
                set(virtual_edges[a, b]["evidence"]["relationship_ids"] + portal["boundary_ids"])
            )
        else:
            edge = {
                "source": a,
                "target": b,
                "kind": "opening",
                "connector_id": min(portal["boundary_ids"]),
                "evidence": {
                    "method": portal["kind"],
                    "relationship_ids": sorted(portal["boundary_ids"]),
                    "patches": [evidence],
                },
                "passability": "not_verified",
            }
            virtual_edges[a, b] = edge
            edges.append(edge)
        adjacency[a].add(b)
        adjacency[b].add(a)
        resolved_boundaries.update(portal["boundary_ids"])
    for boundary in virtual_boundaries:
        if boundary.id() not in resolved_boundaries:
            space = getattr(boundary, "RelatingSpace", None)
            unresolved.append(
                {
                    "connector_id": boundary.id(),
                    "ifc_class": boundary.is_a(),
                    "space_ids": [space.id()] if space is not None else [],
                    "reason": "unpaired_unsupported_or_ambiguous_virtual_boundary",
                    "detail": virtual_error,
                }
            )

    # A supported flight Body mesh may prove unique endpoints by actual
    # tread/floor polygon intersection. Never infer stairs from storey proximity.
    geometry_diagnostics = []
    try:
        stair_portals, geometry_diagnostics = _house_tour_stair_endpoints(
            ifc, storey_lookup=storey_of
        )
    except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
        stair_portals = []
        if entities("IfcStairFlight"):
            geometry_diagnostics.append(
                {"reason": "stair_geometry_unavailable", "detail": str(exc)}
            )
    resolved_flights = set()
    for portal in stair_portals:
        flight_id = portal["flight_id"]
        # Keep explicit ambiguity visible. Geometry must not override it.
        if len(boundary_spaces[flight_id]) > 2:
            continue
        a, b = sorted(portal["space_ids"])
        if a not in space_ids or b not in space_ids:
            continue
        if any(
            e["kind"] == "stair" and flight_id in e["evidence"].get("connector_ids", [])
            for e in edges
        ):
            continue
        connector_ids = [flight_id]
        relationship_ids = []
        for parent, rid in parents[flight_id]:
            if parent.is_a("IfcStair"):
                connector_ids.append(parent.id())
                relationship_ids.append(rid)
        evidence = dict(
            portal,
            method=portal["kind"],
            connector_ids=sorted(connector_ids),
            relationship_ids=sorted(relationship_ids),
            max_endpoint_step_m=0.25,
            min_tread_area_m2=0.05,
            min_overlap_area_m2=0.002,
        )
        edges.append(
            {
                "source": a,
                "target": b,
                "kind": "stair",
                "connector_id": flight_id,
                "evidence": evidence,
                "passability": "not_verified",
            }
        )
        adjacency[a].add(b)
        adjacency[b].add(a)
        resolved_flights.update(connector_ids)
    unresolved = [
        item
        for item in unresolved
        if item["connector_id"] not in resolved_flights
        or item["reason"] != "fewer_than_two_explicit_spaces"
    ]

    def connected(graph):
        result, remaining = [], set(graph)
        while remaining:
            todo, group = [min(remaining)], set()
            while todo:
                sid = todo.pop()
                if sid in group:
                    continue
                group.add(sid)
                todo.extend(sorted(graph[sid] - group, reverse=True))
            remaining -= group
            result.append(sorted(group))
        return sorted(result, key=lambda group: (-len(group), group[0]))

    raw_components = connected(adjacency)
    raw_edges, raw_adjacency = edges, adjacency
    by_id = {node["id"]: node for node in nodes}
    room_ids = {
        sid for sid in space_ids if classifications[sid]["classification"] == "tourable_interior"
    }
    circulation_ids = {
        sid for sid in space_ids if classifications[sid]["classification"] == "circulation"
    }
    eligible = room_ids | circulation_ids
    if start_space_id is not None and start_space_id not in eligible:
        raise ValueError(
            "start_space_id must identify a classified interior room or circulation space."
        )
    edges = [edge for edge in raw_edges if {edge["source"], edge["target"]} <= eligible]
    adjacency = {sid: raw_adjacency[sid] & eligible for sid in eligible}
    components = connected(adjacency)
    # Prune circulation dead ends, but never remove an isolated room target.
    targets = set(eligible)
    changed = True
    while changed:
        leaves = {
            sid
            for sid in targets & circulation_ids
            if len(adjacency[sid] & targets) <= 1 and sid != start_space_id
        }
        changed = bool(leaves)
        targets -= leaves
    targets = {
        sid
        for group in connected({sid: adjacency[sid] & targets for sid in targets})
        if set(group) & room_ids
        for sid in group
    }
    if start_space_id is not None:
        targets.add(start_space_id)
    rng = random.Random(seed)
    order = sorted(eligible)
    rng.shuffle(order)
    rank = {sid: i for i, sid in enumerate(order)}

    def ancestry(entity, kind):
        todo, seen, found = [entity], set(), set()
        while todo:
            item = todo.pop()
            if item.id() in seen:
                continue
            seen.add(item.id())
            if item.is_a(kind):
                found.add(item.id())
            todo.extend(parent for parent, _ in parents[item.id()])
        return found

    buildings = {space.id(): ancestry(space, "IfcBuilding") for space in spaces}
    separation = {"status": "single_component", "pairs": []}
    if len(components) > 1:
        try:
            separation = _house_tour_component_separation_evidence(ifc, components)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            separation = {
                "status": "unresolved",
                "pairs": [],
                "reason": "separation_evidence_unavailable",
                "detail": str(exc),
            }
    tours = []
    for index, component in enumerate(components):
        component_targets = targets & set(component)
        if not component_targets:
            continue
        start = start_space_id if start_space_id in component else min(component_targets)
        route, visited = [start], {start}
        while not component_targets <= visited:
            todo, previous, destination = deque([route[-1]]), {route[-1]: None}, None
            while todo:
                current = todo.popleft()
                if current in component_targets - visited:
                    destination = current
                    break
                for neighbor in sorted(adjacency[current], key=lambda sid: (rank[sid], sid)):
                    if neighbor not in previous:
                        previous[neighbor] = current
                        todo.append(neighbor)
            if destination is None:
                break
            path = []
            while previous[destination] is not None:
                path.append(destination)
                destination = previous[destination]
            route.extend(reversed(path))
            visited.update(path)
        building_ids = set().union(*(buildings[sid] for sid in component))
        others = eligible - set(component)
        separate = (
            len(building_ids) == 1
            and all(buildings[sid] for sid in component)
            and bool(others)
            and all(buildings[sid] and not (buildings[sid] & building_ids) for sid in others)
        )
        pair_evidence = [
            pair
            for pair in separation["pairs"]
            if index in (pair["component_a_index"], pair["component_b_index"])
        ]
        unit_separate = (
            len(pair_evidence) == len(components) - 1
            and bool(pair_evidence)
            and all(pair["status"] == "physically_separate" for pair in pair_evidence)
        )
        isolated = len(component) == 1 and bool(set(component) & room_ids)
        status = (
            "unresolved_missing_connection"
            if isolated
            else "physically_separate"
            if separate or unit_separate
            else "unresolved_missing_connection"
            if len(components) > 1
            else "connected"
        )
        tours.append(
            {
                "component_id": index,
                "space_ids": component,
                "target_space_ids": sorted(component_targets),
                "start_space_id": start,
                "route_space_ids": route,
                "tour": [
                    {
                        "space_id": sid,
                        **{
                            key: by_id[sid][key]
                            for key in (
                                "id",
                                "name",
                                "long_name",
                                "storey_id",
                                "storey_name",
                                "classification",
                            )
                        },
                    }
                    for sid in route
                ],
                "covered_target_space_ids": sorted(visited & component_targets),
                "visited_space_ids": sorted(visited),
                "coverage": len(visited & component_targets) / len(component_targets),
                "target_count": len(component_targets),
                "covered_target_count": len(visited & component_targets),
                "tourable_room_count": len(set(component) & room_ids),
                "covered_tourable_room_count": len(visited & room_ids),
                "tourable_room_coverage": len(visited & room_ids) / len(set(component) & room_ids)
                if set(component) & room_ids
                else None,
                "graph_validation_status": status,
                "separation_evidence": {
                    "building_ids": sorted(building_ids),
                    "unit_separation_pairs": pair_evidence,
                    "reason": "distinct_explicit_ifc_buildings"
                    if separate
                    else "evidence_supported_internal_unit_separation"
                    if unit_separate
                    else "isolated_room_requires_connection_review"
                    if isolated
                    else "no_proven_physical_separation"
                    if len(components) > 1
                    else "single_connected_graph",
                },
            }
        )
    # Multiple component plans are independent. A flat route must never concatenate them.
    flat = tours[0] if len(components) == 1 and len(tours) == 1 else None
    route = flat["route_space_ids"] if flat else []
    visited = set().union(*(set(t["visited_space_ids"]) for t in tours)) if tours else set()
    covered = visited & targets
    selected = next(
        (t for t in tours if start_space_id in t["space_ids"]), tours[0] if tours else None
    )
    reachable_from_start = selected["space_ids"] if selected else []
    reachable = sorted({sid for tour in tours for sid in tour["space_ids"]})
    unresolved_spaces = [node for node in nodes if node["classification"] == "unknown"]
    classification_complete = not unresolved_spaces
    component_index = {sid: index for index, group in enumerate(components) for sid in group}
    blocking_connectors, connector_audit = [], []
    for item in unresolved:
        members = set(item.get("space_ids", []))
        candidate_components = {component_index[sid] for sid in members if sid in component_index}
        exterior = False
        connector = connectors.get(item["connector_id"])
        if connector is not None and connector.is_a("IfcDoor") and len(members) == 1:
            related = [
                rel
                for rel in entities("IfcRelSpaceBoundary")
                if getattr(rel, "RelatedBuildingElement", None) is not None
                and rel.RelatedBuildingElement.id()
                in item.get("evidence", {}).get("connector_ids", [item["connector_id"]])
            ]
            exterior = bool(related) and all(
                getattr(rel, "InternalOrExternalBoundary", None) == "EXTERNAL" for rel in related
            )
        nonportal = item["reason"] == "window_or_other_non_door_filling"
        crosses = (
            len(candidate_components) > 1 or bool(members - eligible) and bool(members & eligible)
        )
        # A connector with no space endpoints is an export audit issue, not
        # evidence of an omitted room. Disconnected room components independently
        # remain unresolved unless physical separation is positively supported.
        blocking = not (nonportal or exterior) and crosses
        audit = dict(
            item,
            blocks_graph_validation=blocking,
            audit_reason="explicit_nonportal"
            if nonportal
            else "explicit_exterior_door"
            if exterior
            else "unresolved_cross_component_or_unknown_space_connection"
            if crosses
            else "no_endpoint_evidence"
            if not members
            else "within_component_or_excluded_space_diagnostic",
        )
        (blocking_connectors if blocking else connector_audit).append(audit)
    graph_status = (
        "unresolved"
        if unresolved_spaces
        or blocking_connectors
        or any(t["graph_validation_status"] == "unresolved_missing_connection" for t in tours)
        else "validated_topology"
    )

    unresolved_graph_spaces = [
        {
            "space_id": sid,
            "component_id": tour["component_id"],
            "reason": "unresolved_missing_connection",
            "evidence": tour["separation_evidence"],
        }
        for tour in tours
        if tour["graph_validation_status"] == "unresolved_missing_connection"
        for sid in tour["space_ids"]
    ]

    def label(sid):
        node = by_id[sid]
        names = [str(value) for value in (node["name"], node["long_name"]) if value]
        return f"#{sid} {' / '.join(dict.fromkeys(names)) or 'Unnamed space'}"

    graph_lines = [
        "Raw IFC space graph (includes excluded and unknown spaces; not passability proof):"
    ]
    for node in nodes:
        links = []
        for edge in raw_edges:
            if node["id"] in (edge["source"], edge["target"]):
                other = edge["target"] if node["id"] == edge["source"] else edge["source"]
                links.append(f"{edge['kind']} #{edge['connector_id']} -> {label(other)}")
        graph_lines.append(label(node["id"]) + " -> " + ("; ".join(links) or "isolated"))
    return {
        "nodes": nodes,
        "spaces": nodes,
        "raw_spaces": nodes,
        "raw_edges": raw_edges,
        "raw_components": raw_components,
        "raw_graph": {"nodes": nodes, "edges": raw_edges, "components": raw_components},
        "edges": edges,
        "connections": edges,
        "target_graph": {"space_ids": sorted(eligible), "edges": edges, "components": components},
        "tourable_spaces": [node for node in nodes if node["id"] in room_ids],
        "circulation_spaces": [node for node in nodes if node["id"] in circulation_ids],
        "excluded_spaces": [node for node in nodes if node["classification"] == "non_tourable"],
        "unresolved_spaces": unresolved_spaces,
        "classification_complete": classification_complete,
        "classification_completeness": (len(nodes) - len(unresolved_spaces)) / len(nodes),
        "graph_validation_status": graph_status,
        "graph_validation_scope": "classified_target_connectivity_and_evidenced_component_separation_not_ifc_export_completeness_or_passability",
        "unresolved_graph_spaces": unresolved_graph_spaces,
        "unresolved_graph_space_ids": sorted(item["space_id"] for item in unresolved_graph_spaces),
        "component_separation": separation,
        "component_tours": tours,
        "tours": tours,
        "component_details": tours,
        "components": components,
        "tour": flat["tour"] if flat else [],
        "route_space_ids": route,
        "legacy_flat_tour_available": flat is not None,
        "legacy_flat_tour_reason": "single_component"
        if flat
        else "independent_component_tours_or_no_eligible_targets",
        "isolated_space_ids": sorted(sid for sid in eligible if not adjacency[sid]),
        "raw_isolated_space_ids": sorted(sid for sid in space_ids if not raw_adjacency[sid]),
        "start_space_id": selected["start_space_id"] if selected else None,
        "seed": seed,
        "reachable_space_ids": reachable,
        "visited_space_ids": sorted(visited),
        "unreachable_space_ids": sorted(targets - visited),
        "reachable_from_requested_start_space_ids": reachable_from_start
        if start_space_id is not None
        else None,
        "unreachable_from_requested_start_space_ids": sorted(eligible - set(reachable_from_start))
        if start_space_id is not None
        else None,
        "target_space_ids": sorted(targets),
        "covered_target_space_ids": sorted(covered),
        "target_count": len(targets),
        "covered_target_count": len(covered),
        "tourable_room_count": len(room_ids),
        "covered_tourable_room_count": len(visited & room_ids),
        "coverage": len(covered) / len(targets) if targets else None,
        "tourable_room_coverage": len(visited & room_ids) / len(room_ids) if room_ids else None,
        "coverage_complete": bool(targets)
        and covered == targets
        and graph_status == "validated_topology",
        "coverage_denominator_policy": "all_tourable_rooms_and_non_dead_end_circulation_in_room_components_plus_explicit_start",
        "unresolved_connectors": unresolved,
        "blocking_unresolved_connectors": blocking_connectors,
        "connector_audit_notes": connector_audit,
        "geometry_diagnostics": geometry_diagnostics,
        "total_space_count": len(nodes),
        "reachable_space_count": len(reachable),
        "unique_visited_count": len(visited),
        "all_spaces_visited": visited == space_ids,
        "whole_model_coverage": len(visited) / len(space_ids),
        "storeys": [
            {
                "id": x.id(),
                "name": getattr(x, "Name", None),
                "space_ids": [n["id"] for n in nodes if n["storey_id"] == x.id()],
            }
            for x in entities("IfcBuildingStorey")
        ],
        "represented_storey_ids": sorted(
            {n["storey_id"] for n in nodes if n["storey_id"] is not None}
        ),
        "stats": {
            "space_count": len(nodes),
            "edge_count": len(edges),
            "component_count": len(components),
            "reachable_count": len(reachable),
            "visited_count": len(visited),
            "unreachable_count": len(targets - visited),
            "route_step_count": sum(len(t["route_space_ids"]) for t in tours),
            "unresolved_connector_count": len(unresolved),
        },
        "graph_text": "\n".join(graph_lines),
        "tour_text": " -> ".join(label(sid) for sid in route),
        "warnings": [
            "Semantic connectivity does not prove current door passability or a collision-free route.",
            "Unknown spaces remain unresolved and are not used as interior shortcuts.",
            "Component tours are independent; no continuous whole-house route is implied.",
            "Coverage is combinatorial; check classification completeness and graph validation separately.",
        ],
    }


def _h_plan_house_tour(params):
    params = params or {}
    return _house_tour(_get_loaded_ifc(), params.get("start_space_id"), params.get("seed", 0))


_HANDLERS = {
    "plan_house_tour": _h_plan_house_tour,
    "generate_ego_video": _h_generate_ego_video,
    "ping": _h_ping,
    "get_scene_info": _h_get_scene_info,
    "get_selected_objects": _h_get_selected_objects,
    "list_elements": _h_list_elements,
    "execute_code": _h_execute_code,
    "execute_ifc_code": _h_execute_ifc_code,
    "get_viewport_screenshot": _h_get_viewport_screenshot,
    "get_ifc_project_info": _h_get_ifc_project_info,
    "get_spatial_structure": _h_get_spatial_structure,
    "get_quantities": _h_get_quantities,
    "get_psets": _h_get_psets,
    "save_ifc_file": _h_save_ifc_file,
    "refresh_view": _h_refresh_view,
    "refresh_geometry": _h_refresh_geometry,
    "reload_project": _h_reload_project,
}

class _PendingRequest:
    """Bundle a request with an Event that is set when the result is ready.

    `cancelled` is set by the connection handler when the client is gone or
    timed out: the drain loop then skips execution. `finished` is set by the
    bridge only when it produced a definitive outcome (result or error), so
    the handler can distinguish "ran" from "was skipped".
    """

    __slots__ = (
        "command",
        "params",
        "result",
        "error",
        "traceback",
        "done",
        "cancelled",
        "cancel_reason",
        "finished",
    )

    def __init__(self, command: str, params: dict) -> None:
        self.command = command
        self.params = params
        self.result = None
        self.error: str | None = None
        self.traceback: str | None = None
        self.done = threading.Event()
        self.cancelled = False
        self.cancel_reason: str | None = None
        self.finished = False


def _edits_allowed() -> bool:
    """Whether EDIT commands may run (the add-on's 'Allow edits' preference)."""
    try:
        return bool(_prefs().allow_edits)
    except Exception:
        return True


def _record_activity(pending: _PendingRequest, duration_s: float) -> None:
    _STATE["requests_served"] = int(_STATE.get("requests_served", 0)) + 1  # type: ignore[arg-type]
    _STATE["last_command"] = pending.command
    log = _STATE.get("activity")
    if isinstance(log, collections.deque):
        stamp = time.strftime("%H:%M:%S")
        outcome = "err" if pending.error else "ok"
        log.appendleft(f"{stamp} {outcome} {pending.command} {duration_s * 1000.0:.0f}ms")


def _redraw_view3d_areas() -> None:
    """Repaint 3D viewports so the panel's counters update without mouse-over."""
    with contextlib.suppress(Exception):
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()


def _drain_queue() -> float | None:
    """Timer callback. Runs on the main thread, executes queued commands."""
    if _STATE.get("server") is None:
        # bridge stopped: fail anything still queued and let Blender
        # unregister this timer (handler threads may have re-registered it
        # in the window between shutdown and their own exit)
        _flush_queue("bridge stopped before the request could run")
        _STATE["timer_registered"] = False
        return None
    q: queue.Queue = _STATE["request_queue"]  # type: ignore[assignment]
    drained = 0
    while drained < 8:
        try:
            pending: _PendingRequest = q.get_nowait()
        except queue.Empty:
            break
        drained += 1
        if pending.cancelled:
            # the client already gave up on this request; never execute it
            pending.done.set()
            continue
        _invalidate_ifc_cache()
        started = time.perf_counter()
        handler = _HANDLERS.get(pending.command)
        try:
            if handler is None:
                raise ValueError(f"Unknown command: {pending.command!r}")
            if pending.command in _EDIT_COMMANDS and not _edits_allowed():
                raise PermissionError(
                    "Editing is disabled in the Bonsai MCP add-on settings. "
                    "Enable it in the 3D viewport sidebar (N) > Bonsai MCP > "
                    "Allow edits. Query tools keep working in read-only mode."
                )
            pending.result = handler(pending.params)
        except (Exception, SystemExit) as exc:
            # SystemExit too: exec()'d code calling sys.exit() must not kill
            # this timer (that would strand every future request)
            pending.error = f"{type(exc).__name__}: {exc}"
            pending.traceback = traceback.format_exc()
        finally:
            pending.finished = True
            pending.done.set()
            _record_activity(pending, time.perf_counter() - started)
    if drained:
        _redraw_view3d_areas()
    return 0.05


def _flush_queue(reason: str) -> None:
    """Fail every queued request; called when the bridge stops."""
    q: queue.Queue = _STATE["request_queue"]  # type: ignore[assignment]
    while True:
        try:
            pending: _PendingRequest = q.get_nowait()
        except queue.Empty:
            return
        pending.error = reason
        pending.finished = True
        pending.done.set()


def _ensure_timer_running() -> None:
    if _STATE.get("server") is None:
        # never (re)register the drain timer for a stopped bridge; handler
        # threads that outlive _stop_server call this on every request
        return
    if bpy.app.timers.is_registered(_drain_queue):
        _STATE["timer_registered"] = True
        return
    # persistent=True: file loads silently unregister non-persistent timers
    bpy.app.timers.register(_drain_queue, persistent=True)
    _STATE["timer_registered"] = True


def _ensure_timer_stopped() -> None:
    if bpy.app.timers.is_registered(_drain_queue):
        bpy.app.timers.unregister(_drain_queue)
    _STATE["timer_registered"] = False

class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


# Idle timeout for an open client connection; the persistent client
# reconnects transparently when it expires.
_IDLE_TIMEOUT_SECONDS = 300.0


class _BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        """Read framed requests until the client disconnects (socketserver hook)."""
        sock: socket.socket = self.request
        sock.settimeout(_IDLE_TIMEOUT_SECONDS)
        try:
            while True:
                try:
                    message = _read_message(sock)
                except _ProtocolError as exc:
                    with contextlib.suppress(OSError):
                        _send_message(
                            sock, {"success": False, "error": f"Protocol error: {exc}"}
                        )
                    self._drain_and_close(sock)
                    return
                if message is None:
                    return

                request_id = message.get("id")

                def _reply(payload: dict, _rid=request_id) -> None:
                    # echo the client's request id so a persistent connection
                    # can detect out-of-sync replies
                    if _rid is not None:
                        payload["id"] = _rid
                    _send_message(sock, payload)

                if _STATE.get("server") is not self.server:
                    # the bridge was stopped while this connection stayed open;
                    # refuse instead of executing commands on a "stopped" bridge
                    with contextlib.suppress(OSError):
                        _reply({"success": False, "error": "bridge stopped"})
                    return

                expected_token = str(_STATE.get("token") or "")
                if expected_token:
                    provided = message.get("token")
                    if not isinstance(provided, str) or not hmac.compare_digest(
                        provided, expected_token
                    ):
                        with contextlib.suppress(OSError):
                            _reply(
                                {
                                    "success": False,
                                    "error": (
                                        "Invalid or missing bridge token. This "
                                        "bridge requires a shared secret: set "
                                        "BONSAI_MCP_TOKEN on the client to the "
                                        "token configured in the add-on "
                                        "preferences."
                                    ),
                                }
                            )
                        self._drain_and_close(sock)
                        return

                command = str(message.get("command", ""))
                params = message.get("params") or {}

                pending = _PendingRequest(command, params)
                _STATE["request_queue"].put(pending)  # type: ignore[union-attr]

                # re-register the drain timer if something killed it;
                # bpy.app.timers is safe to call from this thread
                with contextlib.suppress(Exception):
                    _ensure_timer_running()

                if not self._await_result(sock, pending):
                    # client is gone; the request was cancelled, send nothing
                    return

                if pending.finished:
                    if pending.error is not None:
                        _reply(
                            {
                                "success": False,
                                "error": pending.error,
                                "traceback": pending.traceback,
                            }
                        )
                    else:
                        _reply({"success": True, "result": pending.result})
                elif pending.cancel_reason == "client_closed":
                    _reply(
                        {
                            "success": False,
                            "error": (
                                "The request was cancelled because the client "
                                "closed the connection before a result was ready; "
                                "it will not run."
                            ),
                        }
                    )
                else:
                    _reply(
                        {
                            "success": False,
                            "error": (
                                f"Timed out after {3600.0 if command == 'generate_ego_video' else MAIN_THREAD_WAIT_SECONDS:.0f}s waiting "
                                "for Blender's main thread; the queued request was "
                                "cancelled and will not run (unless it had already "
                                "started). Blender may be busy with a modal operation "
                                "or a long-running script."
                            ),
                        }
                    )
        except OSError:
            # normal client disconnect (WinError 10053 on Windows), not an error
            return

    @staticmethod
    def _drain_and_close(sock: socket.socket) -> None:
        """Half-close and drain unread input so the error reply is delivered.

        Closing a socket with unread received data (e.g. the body of an
        oversized frame) sends an RST that can destroy the just-sent reply.
        Draining (bounded) lets the close end with an orderly FIN instead.
        """
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(2.0)
            drained = 0
            while drained < 1024 * 1024:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                drained += len(chunk)

    @staticmethod
    def _await_result(sock: socket.socket, pending: _PendingRequest) -> bool:
        """Wait for the main thread, watching for client disconnect and timeout.

        Returns False when the client is gone and nothing should be sent.
        On timeout or bridge stop, sets `cancelled` so the drain loop skips
        the request, gives the main thread a short grace period in case it
        finished concurrently, and returns True so the caller reports the
        outcome.
        """
        deadline = time.monotonic() + (3600.0 if pending.command == "generate_ego_video" else MAIN_THREAD_WAIT_SECONDS)
        poll_socket = True
        while not pending.done.wait(timeout=0.5):
            if _STATE.get("server") is None:
                # bridge stopped while this request waited; nothing will run it
                pending.cancelled = True
                if not pending.done.is_set():
                    pending.error = "bridge stopped before the request could run"
                    pending.finished = True
                    pending.done.set()
                return True
            if time.monotonic() >= deadline:
                pending.cancelled = True
                pending.cancel_reason = pending.cancel_reason or "timeout"
                pending.done.wait(timeout=0.05)
                return True
            if not poll_socket:
                continue
            try:
                readable, _, _ = select.select([sock], [], [], 0)
                if readable and sock.recv(1, socket.MSG_PEEK) == b"":
                    # EOF: the client closed its side - either a full close
                    # (gone for good) or a half-close (legal for a one-shot
                    # send/shutdown(SHUT_WR)/recv exchange, still reading).
                    # TCP cannot distinguish the two, so: cancel the request
                    # if it has not started (safety: no ghost edits), but
                    # keep waiting for an in-flight result and attempt to
                    # deliver the reply; a fully-closed peer surfaces as
                    # OSError on send, which the caller swallows.
                    pending.cancelled = True
                    pending.cancel_reason = "client_closed"
                    poll_socket = False
            except OSError:
                pending.cancelled = True
                return False
        return True


def _start_server(host: str, port: int, token: str = "") -> bool:
    """Start the bridge. Returns False when it was already running.

    The token is snapshotted here (main thread) so handler threads never
    read bpy preferences; changing the preference therefore requires a
    bridge restart to take effect.
    """
    if _STATE.get("server") is not None:
        return False
    server = _ThreadedTCPServer((host, port), _BridgeHandler)
    thread = threading.Thread(
        target=server.serve_forever, name="bonsai-mcp-bridge", daemon=True
    )
    _STATE["server"] = server
    _STATE["thread"] = thread
    _STATE["bound"] = tuple(server.server_address[:2])
    _STATE["last_error"] = None
    _STATE["token"] = str(token or "")
    thread.start()
    _ensure_timer_running()
    bound = _STATE["bound"]
    print(f"[bonsai-mcp-bridge] listening on {bound[0]}:{bound[1]}")  # type: ignore[index]
    return True


def _bound_address() -> tuple[str, int] | None:
    bound = _STATE.get("bound")
    if isinstance(bound, tuple) and len(bound) == 2:
        return bound  # type: ignore[return-value]
    return None


def _cleanup_screenshot_dir() -> None:
    tmp = _STATE.pop("screenshot_dir", None)
    if isinstance(tmp, str):
        shutil.rmtree(tmp, ignore_errors=True)


def _stop_server() -> None:
    server = _STATE.get("server")
    if server is None:
        _ensure_timer_stopped()
        _flush_queue("bridge stopped before the request could run")
        _cleanup_screenshot_dir()
        return
    try:
        server.shutdown()  # type: ignore[attr-defined]
        server.server_close()  # type: ignore[attr-defined]
    finally:
        _STATE["server"] = None
        _STATE["thread"] = None
        _STATE["bound"] = None
        _STATE["token"] = ""
        _ensure_timer_stopped()
        # requests queued but never run must not leave clients waiting
        _flush_queue("bridge stopped before the request could run")
        _cleanup_screenshot_dir()
        _cleanup_reload_dir()
        print("[bonsai-mcp-bridge] stopped")

class BONSAI_MCP_AddonPrefs(AddonPreferences):
    bl_idname = __name__

    host: StringProperty(  # type: ignore[valid-type]
        name="Host",
        default=DEFAULT_HOST,
        description="Bind address. Leave as 127.0.0.1 unless you know what you're doing.",
    )
    port: IntProperty(  # type: ignore[valid-type]
        name="Port",
        default=DEFAULT_PORT,
        min=1024,
        max=65535,
    )
    allow_edits: BoolProperty(  # type: ignore[valid-type]
        name="Allow edits",
        default=True,
        description=(
            "Allow the MCP EDIT tools (execute_ifc_code, execute_blender_code, "
            "save_ifc_file) to run. Disable for a read-only session: queries and "
            "screenshots keep working, code execution and saving are blocked"
        ),
    )
    token: StringProperty(  # type: ignore[valid-type]
        name="Token",
        default="",
        subtype="PASSWORD",
        description=(
            "Optional shared secret. When set, every bridge request must "
            "carry the same token (client side: BONSAI_MCP_TOKEN). Useful on "
            "shared machines where loopback is not a user boundary. Applied "
            "on bridge start; restart the bridge after changing it"
        ),
    )

    def draw(self, _context) -> None:
        layout = self.layout
        layout.label(text="Bridge bind address (local only):")
        layout.prop(self, "host")
        layout.prop(self, "port")
        layout.prop(self, "allow_edits")
        layout.prop(self, "token")
        layout.label(text="Token changes apply on the next bridge start.")
        layout.label(text="Warning: do not bind to a non-loopback address.", icon="ERROR")


def _prefs():
    return bpy.context.preferences.addons[__name__].preferences  # type: ignore[index]


class BONSAI_MCP_OT_StartBridge(Operator):
    bl_idname = "bonsai_mcp.start_bridge"
    bl_label = "Start Bridge"
    bl_description = "Start the local TCP bridge for the bonsai-mcp server."

    def execute(self, _context):
        prefs = _prefs()
        try:
            started = _start_server(prefs.host, prefs.port, getattr(prefs, "token", ""))
        except OSError as exc:
            message = (
                f"Could not start bridge on {prefs.host}:{prefs.port}: {exc}. "
                "The port may be in use by another application (for example a "
                "different Blender MCP bridge). Pick another port here and set "
                "BONSAI_MCP_PORT to match on the client side."
            )
            if getattr(exc, "errno", None) == errno.EADDRINUSE:
                message += (
                    " If you just stopped the bridge, the port can stay reserved "
                    "for a short time (TIME_WAIT); wait a few seconds and press "
                    "Start again."
                )
            _STATE["last_error"] = f"Start failed: port {prefs.port} unavailable"
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        bound = _bound_address()
        if not started and bound is not None:
            self.report(
                {"INFO"},
                f"Bridge already running on {bound[0]}:{bound[1]} "
                "(stop it first to rebind with new preferences)",
            )
            return {"FINISHED"}
        if bound is not None:
            self.report({"INFO"}, f"Bridge listening on {bound[0]}:{bound[1]}")
        else:
            self.report({"INFO"}, f"Bridge listening on {prefs.host}:{prefs.port}")
        return {"FINISHED"}


class BONSAI_MCP_OT_StopBridge(Operator):
    bl_idname = "bonsai_mcp.stop_bridge"
    bl_label = "Stop Bridge"
    bl_description = "Stop the local TCP bridge."

    def execute(self, _context):
        _stop_server()
        self.report({"INFO"}, "Bridge stopped")
        return {"FINISHED"}


class BONSAI_MCP_PT_Panel(Panel):
    bl_label = "Bonsai MCP"
    bl_idname = "BONSAI_MCP_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bonsai MCP"

    def draw(self, _context) -> None:
        layout = self.layout
        prefs = _prefs()
        running = _STATE.get("server") is not None
        bound = _bound_address()

        col = layout.column(align=True)
        col.label(
            text=f"Status: {'running' if running else 'stopped'}",
            icon="CHECKMARK" if running else "X",
        )
        if running and bound is not None:
            # the address actually bound, not the (editable) preference
            col.label(text=f"Listening on {bound[0]}:{bound[1]}")
        else:
            col.label(text=f"Bind: {prefs.host}:{prefs.port}")
        if running and _STATE.get("token"):
            col.label(text="Token required by clients", icon="LOCKED")
        last_error = _STATE.get("last_error")
        if last_error:
            col.label(text=str(last_error), icon="ERROR")
        col.separator()

        row = col.row(align=True)
        start = row.row(align=True)
        start.enabled = not running
        start.operator(BONSAI_MCP_OT_StartBridge.bl_idname, icon="PLAY")
        stop = row.row(align=True)
        stop.enabled = running
        stop.operator(BONSAI_MCP_OT_StopBridge.bl_idname, icon="PAUSE")
        col.separator()

        col.prop(prefs, "allow_edits")
        if prefs.allow_edits:
            warn = col.row()
            warn.alert = True
            warn.label(text="Edits on: AI can modify the model.", icon="UNLOCKED")
        else:
            col.label(text="Read-only: EDIT tools are blocked.", icon="LOCKED")

        if running:
            col.separator()
            activity = _STATE.get("activity")
            if isinstance(activity, collections.deque) and activity:
                col.label(text="Recent:")
                for line in list(activity):
                    col.label(text=f"  {line}")

        col.separator()
        col.label(text="Local trusted use only.", icon="ERROR")


_CLASSES = (
    BONSAI_MCP_AddonPrefs,
    BONSAI_MCP_OT_StartBridge,
    BONSAI_MCP_OT_StopBridge,
    BONSAI_MCP_PT_Panel,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    _stop_server()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
