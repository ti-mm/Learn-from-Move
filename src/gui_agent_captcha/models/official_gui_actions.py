"""Parse model-native calls without executing generated Python or changing their vocabulary."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from ..actions import AtomicAction, PrimitiveAction
from ..core import Observation

OFFICIAL_TOOL_ACTIONS = {
    "gui_owl": (
        "key",
        "type",
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "scroll",
        "hscroll",
        "wait",
        "interact",
        "terminate",
        "answer",
    ),
    "evocua_s2": (
        "key",
        "type",
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "scroll",
        "wait",
        "terminate",
        "key_down",
        "key_up",
    ),
}
EXECUTOR_ALIASES = {
    "gui_owl": ("click", "drag"),
    "evocua_s2": ("click",),
}
# Parser/executor acceptance includes aliases used by the desktop adapters;
# these names are deliberately kept separate from the upstream prompt enum.
TOOL_ACTIONS = {
    style: OFFICIAL_TOOL_ACTIONS[style] + EXECUTOR_ALIASES.get(style, ())
    for style in OFFICIAL_TOOL_ACTIONS
}
EXECUTOR_ACTIONS = TOOL_ACTIONS

UITARS_ACTIONS = (
    "click",
    "left_double",
    "right_single",
    "drag",
    "hotkey",
    "type",
    "scroll",
    "wait",
    "finished",
    "move_to",
)
UITARS_OFFICIAL_ACTIONS = UITARS_ACTIONS[:-1]
BUTTON_EXTENSION_STYLES = frozenset({"uitars", "dart_gui", "gui_owl", "evocua_s2"})
BUTTON_EXTENSION_ACTIONS = ("mouse_down", "mouse_up")


def current_position_click_action(style: str) -> str:
    return "left_click_current" if style == "gui_owl" else "left_click"


def missing_mouse_actions(style: str) -> tuple[str, ...]:
    """Add current-position click only to templates lacking that operation."""
    if style not in BUTTON_EXTENSION_STYLES:
        raise ValueError(f"mouse extension is not defined for {style}")
    if style in {"uitars", "dart_gui", "gui_owl"}:
        return (*BUTTON_EXTENSION_ACTIONS, current_position_click_action(style))
    return BUTTON_EXTENSION_ACTIONS
HOLO_ACTIONS = (
    "click",
    "write",
    "answer",
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
)
PY_AUTO_GUI_PARAMETERS = {
    "moveTo": ("x", "y", "duration"),
    "moveRel": ("xOffset", "yOffset", "duration"),
    "dragTo": ("x", "y", "duration", "button"),
    "dragRel": ("xOffset", "yOffset", "duration", "button"),
    "click": ("x", "y", "clicks", "interval", "button"),
    "doubleClick": ("x", "y", "interval", "button"),
    "tripleClick": ("x", "y", "interval", "button"),
    "rightClick": ("x", "y", "interval"),
    "middleClick": ("x", "y", "interval"),
    "mouseDown": ("x", "y", "button"),
    "mouseUp": ("x", "y", "button"),
    "scroll": ("clicks", "x", "y"),
    "hscroll": ("clicks", "x", "y"),
    "write": ("message", "interval"),
    "typewrite": ("message", "interval"),
    "press": ("keys", "presses", "interval"),
    "keyDown": ("key",),
    "keyUp": ("key",),
    "sleep": ("seconds",),
}


class OfficialGuiProtocolError(ValueError):
    """The generated response violates the selected model's action contract."""


@dataclass(frozen=True)
class OfficialGuiCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class OfficialGuiAction:
    style: str
    calls: tuple[OfficialGuiCall, ...]
    coordinate_size: tuple[int, int]

    @property
    def kind(self) -> str:
        return self.calls[0].name if len(self.calls) == 1 else "native_batch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "style": self.style,
            "calls": [call.to_dict() for call in self.calls],
            "coordinate_size": list(self.coordinate_size),
        }


def without_thinking(raw: str) -> str:
    # Qwen's generation prefix may already contain the opening <think> tag.
    return raw.rsplit("</think>", 1)[-1].strip()


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric coordinate, got {value!r}")
    if not math.isfinite(value):
        raise ValueError("coordinates must be finite")
    return float(value)


def _point(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        value = value.replace("<|box_start|>", "").replace("<|box_end|>", "")
        value = ast.literal_eval(value)
    if not isinstance(value, (tuple, list)) or len(value) not in {2, 4}:
        raise ValueError("expected a point or bounding box")
    values = [_number(item) for item in value]
    if len(values) == 4:
        return ((values[0] + values[2]) / 2, (values[1] + values[3]) / 2)
    return values[0], values[1]


def _validate_arguments(call: OfficialGuiCall, style: str) -> None:
    name, args = call.name, call.arguments
    if style in TOOL_ACTIONS:
        # Upstream executors read recognized fields and ignore unused fields.
        # Preserve them in the trace without inventing additional mouse effects.
        if style == "evocua_s2" and name == "left_click_drag":
            if "duration" in args and _number(args["duration"]) < 0:
                raise ValueError("drag duration must be non-negative")
        required = {
            "key": "keys",
            "key_down": "keys",
            "key_up": "keys",
            "type": "text",
            "scroll": "pixels",
            "hscroll": "pixels",
            "wait": "time",
            "terminate": "status",
            "answer": "text",
            "interact": "text",
        }.get(name)
        if required and required not in args:
            raise ValueError(f"{name} requires {required}")
        if "keys" in args and (
            not isinstance(args["keys"], list)
            or not args["keys"]
            or not all(isinstance(key, str) and key for key in args["keys"])
        ):
            raise ValueError("keys must be a non-empty array of strings")
    for key in {"content", "text", "element", "key", "direction"} & args.keys():
        if not isinstance(args[key], str):
            raise ValueError(f"{key} must be a string")
    if "status" in args and args["status"] not in {"success", "failure"}:
        raise ValueError("status must be success or failure")
    if "press_enter" in args and not isinstance(args["press_enter"], bool):
        raise ValueError("press_enter must be boolean")
    for key in {
        "x",
        "y",
        "xOffset",
        "yOffset",
        "pixels",
        "duration",
        "time",
        "seconds",
        "interval",
    } & args.keys():
        if args[key] is None and key in {"x", "y"} and style == "opencua_pyautogui":
            continue
        value = _number(args[key])
        if key in {"duration", "time", "seconds", "interval"} and value < 0:
            raise ValueError(f"{key} must be non-negative")
        if style == "holo3_tool" and key in {"x", "y"} and not isinstance(args[key], int):
            raise ValueError("Holo coordinates must be integers")
    if "button" in args and args["button"] not in {
        "left",
        "middle",
        "right",
        "primary",
        "secondary",
    }:
        raise ValueError("invalid mouse button")


def _python_calls(code: str, *, pyautogui: bool) -> list[OfficialGuiCall]:
    calls = []
    for node in ast.parse(code.strip()).body:
        if isinstance(node, ast.Import) and pyautogui:
            if all(alias.name == "pyautogui" and alias.asname is None for alias in node.names):
                continue
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("expected literal action calls; Python control flow is unsupported")
        call = node.value
        if pyautogui:
            if not isinstance(call.func, ast.Attribute) or not isinstance(
                call.func.value, ast.Name
            ):
                raise ValueError("expected pyautogui.* or computer.* calls")
            namespace, name = call.func.value.id, call.func.attr
            if namespace == "computer":
                params = {"terminate": ("status",), "triple_click": ("x", "y"), "wait": ()}
            elif namespace == "pyautogui":
                params = {**PY_AUTO_GUI_PARAMETERS, "hotkey": ()}
            else:
                raise ValueError(f"unsupported Python namespace: {namespace}")
            if name not in params:
                raise ValueError(f"unsupported native Python function: {namespace}.{name}")
            values = [ast.literal_eval(arg) for arg in call.args]
            if name == "hotkey":
                arguments = {"keys": values}
            else:
                if len(values) > len(params[name]):
                    raise ValueError(f"too many arguments for {name}")
                arguments = dict(zip(params[name], values))
            allowed = set(params[name]) | ({"interval"} if name == "hotkey" else set())
            for kw in call.keywords:
                if kw.arg not in allowed or kw.arg in arguments:
                    raise ValueError(f"invalid or duplicate argument for {name}: {kw.arg}")
                arguments[kw.arg] = ast.literal_eval(kw.value)
            name = f"{namespace}.{name}"
        else:
            if not isinstance(call.func, ast.Name) or call.args:
                raise ValueError("expected a native action with keyword arguments")
            name = call.func.id
            arguments = {}
            for kw in call.keywords:
                if kw.arg is None or kw.arg in arguments:
                    raise ValueError("expanded/duplicate action arguments are unsupported")
                arguments[kw.arg] = ast.literal_eval(kw.value)
        calls.append(OfficialGuiCall(name, arguments))
    return calls


def parse_official_response(
    raw: str,
    style: str,
    *,
    coordinate_size: tuple[int, int],
    button_extension: bool = False,
) -> OfficialGuiAction:
    if button_extension and style not in BUTTON_EXTENSION_STYLES:
        raise ValueError(f"button extension is not defined for {style}")
    extra_actions = missing_mouse_actions(style) if button_extension else ()
    body = without_thinking(raw)
    if style == "ui_venus2_computer":
        from .ui_venus2_official import parse_response

        _, _, call = parse_response(body)
        children = call.actions if call.name == "Sequence" else (call,)
        return OfficialGuiAction(
            style, tuple(OfficialGuiCall(c.name, c.kwargs) for c in children), (999, 999)
        )
    if style in {"uitars", "dart_gui"}:
        match = re.search(r"(?:^|\n)Action:\s*(.*)", body, re.S)
        if not match:
            raise ValueError("missing native Action: field")
        calls = _python_calls(match.group(1), pyautogui=False)
        if len(calls) != 1 or calls[0].name not in (*UITARS_ACTIONS, *extra_actions):
            raise ValueError("expected one declared UI-TARS/DART action")
        call = calls[0]
        required = {
            "click": {"start_box"},
            "left_double": {"start_box"},
            "right_single": {"start_box"},
            "move_to": {"start_box"},
            "drag": {"start_box", "end_box"},
            "hotkey": {"key"},
            "type": {"content"},
            "scroll": {"start_box", "direction"},
            "wait": set(),
            "finished": {"content"},
            "mouse_down": set(),
            "mouse_up": set(),
            "left_click": set(),
        }[call.name]
        if set(call.arguments) != required:
            raise ValueError(f"{call.name} requires exactly {sorted(required)}")
        for key in required & {"start_box", "end_box"}:
            _point(call.arguments[key])
    elif style in TOOL_ACTIONS:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", body, re.S)
        calls = []
        for block in blocks:
            obj = json.loads(block)
            if obj.get("name") != "computer_use" or not isinstance(obj.get("arguments"), dict):
                raise ValueError("expected computer_use arguments")
            args = dict(obj["arguments"])
            name = args.pop("action", None)
            if name not in (*TOOL_ACTIONS[style], *extra_actions):
                raise ValueError(f"undeclared {style} action: {name}")
            if name == "left_click_current" and args:
                raise ValueError("left_click_current takes no arguments")
            if name in extra_actions and args not in ({}, {"button": "left"}):
                raise ValueError(f"{name} accepts only an optional button=left")
            if button_extension and style == "gui_owl" and name in {"left_click", "click"} and "coordinate" not in args:
                raise ValueError(f"{name} requires coordinate; use left_click_current for a click in place")
            if name in {"mouse_move", "left_click_drag", "drag"} and "coordinate" not in args:
                raise ValueError(f"{name} requires coordinate")
            if "coordinate" in args:
                if not isinstance(args["coordinate"], list) or len(args["coordinate"]) != 2:
                    raise ValueError("computer_use coordinate must be [x, y]")
                _point(args["coordinate"])
            calls.append(OfficialGuiCall(name, args))
    elif style == "holo3_tool":
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", body, re.S)
        calls = []
        for block in blocks:
            match = re.fullmatch(r"<function=(\w+)>\s*(.*?)\s*</function>", block, re.S)
            if not match or match[1] not in HOLO_ACTIONS:
                raise ValueError("expected a declared Holo3.1 native function")
            params = re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", match[2], re.S)
            remainder = re.sub(r"<parameter=\w+>.*?</parameter>", "", match[2], flags=re.S)
            if remainder.strip():
                raise ValueError("malformed native function parameters")
            args = {}
            point_parameters = []
            for key, value in params:
                is_drag_points = match[1] == "drag" and key == "points"
                if key in args and not is_drag_points:
                    raise ValueError("duplicate native function parameter")
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value
                if is_drag_points:
                    point_parameters.append(args[key])
            if len(point_parameters) > 1:
                # Repeated pair-valued points are an ordered drag path.
                args["points"] = point_parameters
            required = {
                "click": {"element", "x", "y"},
                "move_to": {"x", "y"},
                "drag": {"points"},
                "answer": {"content"},
                "write": {"content"},
                "mouse_down": set(),
                "mouse_up": set(),
                "left_click": set(),
            }[match[1]]
            optional = {"press_enter"} if match[1] == "write" else set()
            if not required <= args.keys() or args.keys() - required - optional:
                raise ValueError(f"invalid {match[1]} arguments")
            if match[1] == "drag":
                points = args["points"]
                # Normalize flat integer paths, optionally wrapped once, without
                # inferring missing coordinates or repairing numeric values.
                if isinstance(points, list) and len(points) == 1 and isinstance(points[0], list):
                    points = points[0]
                if (
                    len(point_parameters) == 1
                    and isinstance(points, list)
                    and len(points) >= 4
                    and len(points) % 2 == 0
                    and all(type(v) is int for v in points)
                ):
                    points = [points[i:i + 2] for i in range(0, len(points), 2)]
                if not isinstance(points, list) or len(points) < 2:
                    raise ValueError("Holo drag requires at least two points")
                for point in points:
                    if (
                        not isinstance(point, list)
                        or len(point) != 2
                        or any(type(v) is not int or not 0 <= v <= 1000 for v in point)
                    ):
                        raise ValueError("Holo drag points must be pairs of integers in [0,1000]")
                args["points"] = points
            calls.append(OfficialGuiCall(match[1], args))
    elif style == "opencua_pyautogui":
        blocks = re.findall(r"```(?:python|code)?\s*\n(.*?)```", body, re.S)
        code = "\n".join(blocks)
        if not blocks:
            code = body.split("Code:", 1)[-1].strip()
        calls = _python_calls(code, pyautogui=True)
    else:
        raise ValueError(f"unsupported official GUI style: {style}")
    if not calls:
        raise ValueError("no native action returned")
    for call in calls:
        _validate_arguments(call, style)
    if button_extension and len(calls) != 1 and any(
        call.name in (*BUTTON_EXTENSION_ACTIONS, "left_click", "left_click_current") for call in calls
    ):
        raise ValueError("button extension actions require one call per response")
    if any(size <= 0 for size in coordinate_size):
        raise ValueError("invalid coordinate dimensions")
    return OfficialGuiAction(style, tuple(calls), coordinate_size)


def environment_actions(
    call: OfficialGuiCall,
    action: OfficialGuiAction,
    observation: Observation,
) -> tuple[list[PrimitiveAction | AtomicAction], str]:
    """Convert a call at execution time, using the cursor after preceding calls.

    Keyboard/scroll/non-left mouse actions have no effect in these six isolated
    canvases. They remain legal native decisions, recorded as no-effect actions.
    """
    name, args = call.name, call.arguments
    width, height = action.coordinate_size

    def point(value: Any) -> tuple[float, float]:
        x, y = _point(value)
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError(f"native coordinate {value!r} is outside {width}x{height}")
        return x * 1000 / width, y * 1000 / height

    def cursor() -> tuple[float, float]:
        if observation.cursor_xy is None:
            raise ValueError("current cursor is required for this native action")
        return tuple(v * 1000 / s for v, s in zip(observation.cursor_xy, observation.size_px))

    def move(xy: tuple[float, float]) -> PrimitiveAction:
        return PrimitiveAction("move_to", x=xy[0], y=xy[1])

    if action.style == "ui_venus2_computer":
        if name == "Finished":
            return [], "model_terminated"
        if name == "CallUser":
            return [], "model_requested_user"
        if name == "Wait":
            return [], "wait"
        target = point(args["box"]) if "box" in args else None
        if name == "Hover":
            return [move(target)], "mouse"
        if name in {"MouseDown", "MouseUp"}:
            prefix = [move(target)] if target is not None else []
            return prefix + [PrimitiveAction(
                "mouse_down" if name == "MouseDown" else "mouse_up"
            )], "mouse"
        if name in {"Click", "DoubleClick", "TripleClick"}:
            command = AtomicAction("click", [target]) if target is not None else PrimitiveAction("left_click")
            return [command] * {"Click": 1, "DoubleClick": 2, "TripleClick": 3}[name], "mouse"
        if name == "Drag":
            start = point(args["start"]) if "start" in args else cursor()
            return [AtomicAction("drag", [start, point(args["end"])])], "mouse"
        return [], "no_effect"

    if name in {"finished", "terminate", "answer", "computer.terminate"}:
        return [], "model_terminated"
    if name in {"wait", "computer.wait", "pyautogui.sleep"}:
        return [], "wait"
    if name in {"interact"}:
        return [], "model_requested_user"
    if action.style in BUTTON_EXTENSION_STYLES and name in BUTTON_EXTENSION_ACTIONS:
        allowed = ({}, {"button": "left"}) if action.style in TOOL_ACTIONS else ({},)
        if args not in allowed:
            raise ValueError(f"invalid arguments for {name}")
        return [PrimitiveAction(name)], "mouse"
    if action.style in {"uitars", "dart_gui"}:
        if name == "left_click":
            if args:
                raise ValueError("left_click takes no coordinates or other arguments")
            return [PrimitiveAction("left_click")], "mouse"
        if name in {"click", "left_double", "move_to", "drag"}:
            start = point(args["start_box"])
            if name == "move_to":
                return [move(start)], "mouse"
            if name == "drag":
                return [AtomicAction("drag", [start, point(args["end_box"])])], "mouse"
            return [AtomicAction("click", [start])] * (2 if name == "left_double" else 1), "mouse"
    elif action.style in TOOL_ACTIONS:
        if action.style == "gui_owl" and name == "left_click_current":
            if args:
                raise ValueError("left_click_current takes no arguments")
            return [PrimitiveAction("left_click")], "mouse"
        target = point(args["coordinate"]) if "coordinate" in args else None
        if name == "mouse_move":
            return [move(target)], "mouse"
        if name in {"left_click_drag", "drag"}:
            return [AtomicAction("drag", [cursor(), target])], "mouse"
        if name in {"left_click", "click", "double_click", "triple_click"}:
            click = AtomicAction("click", [target]) if target else PrimitiveAction("left_click")
            count = {"left_click": 1, "click": 1, "double_click": 2, "triple_click": 3}[name]
            return [click] * count, "mouse"
    elif action.style == "holo3_tool":
        if name == "move_to":
            return [move(point([args["x"], args["y"]]))], "mouse"
        if name == "click":
            return [AtomicAction("click", [point([args["x"], args["y"]])])], "mouse"
        if name == "drag":
            points = args["points"]
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError("drag requires at least two points")
            return [AtomicAction("drag", [point(p) for p in points])], "mouse"
        if name in {"mouse_down", "mouse_up", "left_click"}:
            return [PrimitiveAction(name)], "mouse"
    else:
        method = name.split(".", 1)[-1]
        if method in {
            "moveTo",
            "dragTo",
            "moveRel",
            "dragRel",
            "click",
            "doubleClick",
            "tripleClick",
            "triple_click",
            "mouseDown",
            "mouseUp",
        }:
            if args.get("button", "left") not in {"left", "primary"}:
                return [], "no_effect"
            target = None
            if method.endswith("Rel"):
                old = cursor()
                target = (
                    old[0] + _number(args.get("xOffset", 0)) * 1000 / width,
                    old[1] + _number(args.get("yOffset", 0)) * 1000 / height,
                )
                if any(not 0 <= value <= 1000 for value in target):
                    raise ValueError("relative movement endpoint outside screenshot")
            elif "x" in args or "y" in args:
                old = cursor()
                target = point(
                    [
                        args.get("x") if args.get("x") is not None else old[0] * width / 1000,
                        args.get("y") if args.get("y") is not None else old[1] * height / 1000,
                    ]
                )
            if method in {"moveTo", "moveRel", "dragTo", "dragRel"}:
                if target is None:
                    raise ValueError(f"{method} requires coordinates")
                if method.startswith("move"):
                    return [move(target)], "mouse"
                return [AtomicAction("drag", [cursor(), target])], "mouse"
            if method in {"mouseDown", "mouseUp"}:
                prefix = [move(target)] if target else []
                return prefix + [
                    PrimitiveAction("mouse_down" if method == "mouseDown" else "mouse_up")
                ], "mouse"
            count = args.get(
                "clicks", {"doubleClick": 2, "tripleClick": 3, "triple_click": 3}.get(method, 1)
            )
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("click count must be a positive integer")
            click = AtomicAction("click", [target]) if target else PrimitiveAction("left_click")
            return [click] * count, "mouse"
    return [], "no_effect"
