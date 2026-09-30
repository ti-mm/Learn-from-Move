from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Iterable, Literal

from ..actions import Action, AtomicAction, PrimitiveAction, WebAction
from ..core import Observation, StepResult
from ..prompt_contracts import render_raw_unified_contract_block
from ..protocol_tracks import (
    OSWORLD_MOUSE_ACTION_KINDS,
    canonicalize_task_requirement,
    is_self_built_osworld_mouse_task,
)
from .native_baseline_protocols import (
    native_action_kinds,
    native_model_action_kinds,
    native_system_prompt,
    native_task_prompt,
)
from .openai_chat import build_action_prompt_text, parse_action_blob

BaselinePromptStyle = Literal[
    "repo_json",
    "uitars",
    "dart_gui",
    "evocua_s2",
    "gui_owl",
    "holo3_tool",
    "opencua_pyautogui",
    "mai_ui",
    "ui_voyager",
    "ui_venus",
]
PyAutoGUICoordinateMode = Literal[
    "relative_0_1000",
    "relative_0_999",
    "pixel",
    "qwen25_smart_resize",
    "evocua_0_999",
]

_TOOL_CALL_STYLES = {"evocua_s2", "gui_owl"}
_THOUGHT_ACTION_STYLES = {"uitars", "dart_gui"}
_SELF_BUILT_MOUSE_ACTIONS = (
    "click",
    "drag",
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
)


def render_for_baseline(
    style: BaselinePromptStyle,
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    budget: int | None = None,
    condition: str | None = None,
) -> str:
    """Render the current repo task contract through a baseline-specific shell."""

    if style == "repo_json":
        return build_action_prompt_text(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
            condition=condition,
        )

    allowed = (
        tuple(dict.fromkeys(str(kind) for kind in allowed_kinds if str(kind)))
        if style == "ui_venus"
        else native_action_kinds(allowed_kinds)
    )
    system = native_system_prompt(style, allowed_kinds=allowed)
    task = native_task_prompt(style, obs, allowed_kinds=allowed)
    if system == "You are a helpful assistant.":
        return task
    return f"{system}\n\n{task}"


def normalize_baseline_output(
    raw: str,
    style: BaselinePromptStyle,
    *,
    allowed_kinds: Iterable[str] | None = None,
    image_size_px: tuple[int, int] | None = None,
    native_image_size_px: tuple[int, int] | None = None,
    current_cursor_xy: tuple[float, float] | None = None,
    pyautogui_coordinate_mode: PyAutoGUICoordinateMode = "relative_0_1000",
    strict: bool = False,
) -> Action:
    """Normalize native baseline output into this repo's Action dataclasses."""

    requested_kinds = allowed_kinds or _SELF_BUILT_MOUSE_ACTIONS
    native_names = native_action_kinds(requested_kinds)
    allowed = _normalize_allowed_kinds(
        requested_kinds if style == "repo_json" else native_names
    )
    if style == "repo_json":
        return parse_action_blob(raw)
    if style == "mai_ui":
        action = _normalize_mobile_use_output(
            raw,
            allowed_kinds=allowed,
            model_action_kinds=native_model_action_kinds(style, requested_kinds),
            coordinate_mode="relative_0_999",
            output_shell="mai_ui",
            drag_action_name="drag",
            drag_coordinate_fields=("start_coordinate", "end_coordinate"),
            strict=strict,
        )
        return _validate_action_coordinates(action, strict=strict)
    if style == "ui_voyager":
        action = _normalize_mobile_use_output(
            raw,
            allowed_kinds=allowed,
            model_action_kinds=native_model_action_kinds(style, requested_kinds),
            coordinate_mode="relative_0_999",
            output_shell="ui_voyager",
            drag_action_name="swipe",
            drag_coordinate_fields=("coordinate", "coordinate2"),
            strict=strict,
        )
        return _validate_action_coordinates(action, strict=strict)
    if style == "ui_venus":
        action = _normalize_ui_venus_output(
            raw,
            allowed_kinds=allowed,
            image_size_px=image_size_px,
            native_image_size_px=native_image_size_px,
            strict=strict,
        )
        return _validate_action_coordinates(action, strict=strict)
    if style in _TOOL_CALL_STYLES:
        strict_action_names = native_model_action_kinds(style, requested_kinds)
        action = _normalize_tool_call_output(
            raw,
            allowed_kinds=allowed,
            image_size_px=image_size_px,
            current_cursor_xy=current_cursor_xy,
            coordinate_mode=("evocua_0_999" if style == "evocua_s2" else "relative_0_1000"),
            strict_action_names=strict_action_names,
            allow_left_click_drag_duration=style == "evocua_s2",
            strict=strict,
        )
        return _validate_action_coordinates(action, strict=strict)
    if style == "holo3_tool":
        action = _normalize_holo3_structured_output(
            raw,
            allowed_kinds=allowed,
            image_size_px=image_size_px,
            current_cursor_xy=current_cursor_xy,
            strict_action_names=native_names,
            strict=strict,
        )
        return _validate_action_coordinates(action, strict=strict)
    if style in _THOUGHT_ACTION_STYLES:
        action = _normalize_thought_action_output(raw, allowed_kinds=allowed, strict=strict)
        if strict and native_image_size_px is None:
            raise ValueError(f"{style} strict parsing requires the Qwen2.5 native image size")
        converted = _convert_action_coordinates(
            action,
            native_image_size_px=native_image_size_px,
            mode="qwen25_smart_resize",
            strict=strict,
        )
        return _validate_action_coordinates(converted, strict=strict)
    if style == "opencua_pyautogui":
        action = _normalize_pyautogui_output(
            raw,
            allowed_kinds=allowed,
            image_size_px=image_size_px,
            native_image_size_px=native_image_size_px,
            current_cursor_xy=current_cursor_xy,
            coordinate_mode=pyautogui_coordinate_mode,
            strict=strict,
        )
        return _validate_action_coordinates(action, strict=strict)
    raise ValueError(f"unknown baseline prompt style: {style!r}")


def _normalize_mobile_use_output(
    raw: str,
    *,
    allowed_kinds: tuple[str, ...],
    model_action_kinds: tuple[str, ...],
    coordinate_mode: PyAutoGUICoordinateMode,
    output_shell: Literal["mai_ui", "ui_voyager"],
    drag_action_name: str,
    drag_coordinate_fields: tuple[str, str],
    strict: bool,
) -> Action:
    """Parse the native MAI-UI/UI-Voyager ``mobile_use`` response shell."""

    if not strict:
        payload = _extract_tool_call_payload(raw)
        if payload is None:
            return parse_action_blob(raw)
        arguments = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(arguments, dict):
            raise ValueError("mobile_use arguments must be a JSON object")
        normalized = _normalize_mobile_use_arguments(
            arguments,
            drag_action_name=drag_action_name,
            drag_coordinate_fields=drag_coordinate_fields,
        )
        return _action_from_tool_args(
            normalized,
            allowed_kinds=allowed_kinds,
            image_size_px=None,
            current_cursor_xy=None,
            coordinate_mode=coordinate_mode,
            strict=False,
        )

    payload = _extract_strict_mobile_use_payload(
        raw,
        output_shell=output_shell,
    )
    if set(payload) != {"name", "arguments"}:
        raise ValueError("strict mobile_use tool call requires exactly name and arguments")
    if payload.get("name") != "mobile_use":
        raise ValueError("strict mobile_use tool call name must be mobile_use")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("strict mobile_use arguments must be a JSON object")
    _validate_mobile_use_arguments(
        arguments,
        allowed_kinds=model_action_kinds,
        coordinate_max=999 if coordinate_mode == "relative_0_999" else 1000,
        drag_action_name=drag_action_name,
        drag_coordinate_fields=drag_coordinate_fields,
    )
    normalized = _normalize_mobile_use_arguments(
        arguments,
        drag_action_name=drag_action_name,
        drag_coordinate_fields=drag_coordinate_fields,
    )
    return _action_from_tool_args(
        normalized,
        allowed_kinds=allowed_kinds,
        image_size_px=None,
        current_cursor_xy=None,
        coordinate_mode=coordinate_mode,
        strict_action_names=allowed_kinds,
        strict=True,
    )


def _normalize_ui_venus_output(
    raw: str,
    *,
    allowed_kinds: tuple[str, ...],
    image_size_px: tuple[int, int] | None,
    native_image_size_px: tuple[int, int] | None,
    strict: bool,
) -> Action:
    """Normalize UI-Venus web navigation calls into the benchmark action contract."""

    if strict:
        match = re.fullmatch(
            r"\s*<think>\s*(\S(?:.*?\S)?)\s*</think>"
            r"\s*<action>\s*(\S(?:.*?\S)?)\s*</action>"
            r"\s*<conclusion>\s*(\S(?:.*?\S)?)\s*</conclusion>\s*",
            raw,
            re.DOTALL,
        )
        if match is None:
            raise ValueError(
                "strict UI-Venus output requires exactly think, action and conclusion tags"
            )
        action_text = match.group(2).strip()
    else:
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", raw, re.DOTALL)
        action_text = action_match.group(1).strip() if action_match else raw.strip()

    try:
        expression = ast.parse(action_text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"invalid UI-Venus action call: {action_text!r}") from exc
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.args
        or any(keyword.arg is None for keyword in expression.keywords)
    ):
        raise ValueError(f"UI-Venus action must be one named function call: {action_text!r}")
    try:
        arguments = {
            str(keyword.arg): ast.literal_eval(keyword.value)
            for keyword in expression.keywords
        }
    except (ValueError, TypeError) as exc:
        raise ValueError("UI-Venus action arguments must be literal values") from exc
    if len(arguments) != len(expression.keywords):
        raise ValueError("UI-Venus action contains duplicate arguments")

    action_name = expression.func.id
    expected_fields_by_action = {
        "Hover": {"box"},
        "Click": {"box"},
        "Drag": {"start", "end"},
        "Scroll": {"direction"},
        "Type": {"content"},
        "Launch": None,
        "Wait": set(),
        "Finished": {"content"},
        "CallUser": {"content"},
        "LongPress": {"box"},
        "PressBack": set(),
        "PressHome": set(),
        "PressEnter": set(),
        "PressRecent": set(),
        "DoubleClick": {"box"},
        "Hotkey": {"keys"},
    }
    if action_name not in expected_fields_by_action:
        raise ValueError(f"unsupported UI-Venus action: {action_name!r}")
    expected_fields = expected_fields_by_action[action_name]
    if action_name == "Launch":
        if not arguments or not set(arguments).issubset({"app", "url"}):
            raise ValueError("UI-Venus Launch requires app and/or url")
    elif set(arguments) != expected_fields:
        raise ValueError(
            f"UI-Venus {action_name} requires exactly {sorted(expected_fields)!r}"
        )
    if action_name == "Hover":
        if "move_to" not in allowed_kinds:
            raise ValueError("UI-Venus Hover is not enabled for this evaluator")
        _validate_exact_numeric_point(arguments["box"], name="Hover box")
        x_value, y_value = arguments["box"]
        action: Action = PrimitiveAction(kind="move_to", x=x_value, y=y_value)
    elif action_name == "Click":
        if "click" not in allowed_kinds:
            raise ValueError("UI-Venus Click requires the ClickXY action setting")
        _validate_exact_numeric_point(arguments["box"], name="Click box")
        action = AtomicAction(kind="click", points=[tuple(arguments["box"])])
    elif action_name == "Drag":
        if "drag" not in allowed_kinds:
            raise ValueError("UI-Venus Drag is not enabled for this evaluator")
        _validate_exact_numeric_point(arguments["start"], name="Drag start")
        _validate_exact_numeric_point(arguments["end"], name="Drag end")
        action = AtomicAction(
            kind="drag",
            points=[tuple(arguments["start"]), tuple(arguments["end"])],
        )
    elif action_name == "Scroll":
        direction = str(arguments["direction"])
        if direction not in {"up", "down"}:
            raise ValueError("UI-Venus Scroll direction must be 'up' or 'down'")
        action = WebAction(kind="scroll", direction=direction)
    elif action_name == "Type":
        if not isinstance(arguments["content"], str):
            raise ValueError("UI-Venus Type content must be a string")
        action = WebAction(kind="type", content=arguments["content"])
    elif action_name == "Launch":
        app = arguments.get("app")
        url = arguments.get("url")
        if app is not None and not isinstance(app, str):
            raise ValueError("UI-Venus Launch app must be a string")
        if url is not None and not isinstance(url, str):
            raise ValueError("UI-Venus Launch url must be a string")
        action = WebAction(kind="launch", app=app, url=url)
    elif action_name == "Wait":
        action = WebAction(kind="wait")
    elif action_name == "Finished":
        if not isinstance(arguments["content"], str):
            raise ValueError("UI-Venus Finished content must be a string")
        action = WebAction(kind="finished", content=arguments["content"])
    elif action_name == "CallUser":
        if not isinstance(arguments["content"], str):
            raise ValueError("UI-Venus CallUser content must be a string")
        action = WebAction(kind="call_user", content=arguments["content"])
    elif action_name == "LongPress":
        _validate_exact_numeric_point(arguments["box"], name="LongPress box")
        action = WebAction(kind="long_press", points=[tuple(arguments["box"])])
    elif action_name in {"PressBack", "PressHome", "PressEnter", "PressRecent"}:
        action = WebAction(
            kind={
                "PressBack": "press_back",
                "PressHome": "press_home",
                "PressEnter": "press_enter",
                "PressRecent": "press_recent",
            }[action_name]
        )
    elif action_name == "DoubleClick":
        _validate_exact_numeric_point(arguments["box"], name="DoubleClick box")
        action = WebAction(kind="left_double", points=[tuple(arguments["box"])])
    else:
        keys = arguments["keys"]
        if (
            not isinstance(keys, list)
            or not 1 <= len(keys) <= 3
            or not all(isinstance(key, str) and key for key in keys)
        ):
            raise ValueError("UI-Venus Hotkey keys must contain one to three strings")
        action = WebAction(kind="hotkey", key="+".join(keys))

    has_coordinates = (
        isinstance(action, PrimitiveAction) and action.kind == "move_to"
    ) or bool(getattr(action, "points", ()))
    if has_coordinates and strict:
        if (
            native_image_size_px is None
            or image_size_px is None
            or any(value <= 0 for value in (*native_image_size_px, *image_size_px))
        ):
            raise ValueError(
                "strict UI-Venus parsing requires original and native smart-resized image sizes"
            )
    return _convert_ui_venus_action_coordinates(
        action,
        image_size_px=image_size_px,
        native_image_size_px=native_image_size_px,
        strict=strict,
    )


def _extract_strict_mobile_use_payload(
    raw: str,
    *,
    output_shell: Literal["mai_ui", "ui_voyager"],
) -> dict[str, Any]:
    if output_shell == "mai_ui":
        match = re.fullmatch(
            r"\s*<thinking>\s*(\S(?:.*?\S)?)\s*</thinking>"
            r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*",
            raw,
            re.DOTALL,
        )
        payload_group = 2
        expected_shell = "thinking and one tool call in the native XML order"
    else:
        match = re.fullmatch(
            r"\s*Thought:\s*(\S(?:.*?\S)?)\s*"
            r"Action:\s*(\S(?:.*?\S)?)\s*"
            r"<tool_call>\s*(.*?)\s*</tool_call>\s*",
            raw,
            re.DOTALL,
        )
        payload_group = 3
        expected_shell = "Thought, Action and one tool call in the native order"
    if match is None:
        raise ValueError(
            f"strict mobile_use output requires exactly {expected_shell}"
        )
    try:
        payload = json.loads(match.group(payload_group))
    except json.JSONDecodeError as exc:
        raise ValueError(f"strict mobile_use tool call must contain valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("strict mobile_use tool call payload must be a JSON object")
    return payload


def _validate_mobile_use_arguments(
    arguments: dict[str, Any],
    *,
    allowed_kinds: tuple[str, ...],
    coordinate_max: int,
    drag_action_name: str,
    drag_coordinate_fields: tuple[str, str],
) -> None:
    action = arguments.get("action")
    if not isinstance(action, str) or action not in allowed_kinds:
        raise ValueError(f"strict mobile_use action is not advertised by the prompt: {action!r}")
    expected_fields_by_action = {
        "move_to": {"action", "coordinate"},
        "mouse_down": {"action"},
        "mouse_up": {"action"},
        "left_click": {"action"},
        "click": {"action", "coordinate"},
        drag_action_name: {"action", *drag_coordinate_fields},
    }
    expected_fields = expected_fields_by_action.get(action)
    if expected_fields is None:
        raise ValueError(f"unsupported strict mobile_use action: {action!r}")
    if set(arguments) != expected_fields:
        raise ValueError(
            f"strict mobile_use {action} requires exactly {sorted(expected_fields)}"
        )
    coordinate_fields = (
        drag_coordinate_fields
        if action == drag_action_name
        else (("coordinate",) if action in {"move_to", "click"} else ())
    )
    for field in coordinate_fields:
        _validate_exact_integer_point(arguments[field], name=field)
        x_value, y_value = arguments[field]
        _require_coordinate_range(x_value, 0, coordinate_max, name=f"{field} x")
        _require_coordinate_range(y_value, 0, coordinate_max, name=f"{field} y")


def _normalize_mobile_use_arguments(
    arguments: dict[str, Any],
    *,
    drag_action_name: str,
    drag_coordinate_fields: tuple[str, str],
) -> dict[str, Any]:
    normalized = dict(arguments)
    if normalized.get("action") == drag_action_name:
        start_field, end_field = drag_coordinate_fields
        normalized = {
            "action": "drag",
            "points": [normalized.get(start_field), normalized.get(end_field)],
        }
    return normalized


def _normalize_allowed_kinds(action_kinds: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for kind in action_kinds:
        normalized = str(kind).strip()
        if not normalized or normalized in seen:
            continue
        if normalized in _SELF_BUILT_MOUSE_ACTIONS:
            ordered.append(normalized)
            seen.add(normalized)
    return tuple(ordered) or OSWORLD_MOUSE_ACTION_KINDS


def _task_type_for_observation(obs: Observation) -> str | None:
    metadata = obs.metadata or {}
    for key in ("task_type", "benchmark", "task_id", "episode_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _schema_line(kind: str) -> str:
    if kind == "click":
        return "click"
    if kind == "drag":
        return "drag"
    if kind == "move_to":
        return "move_to"
    if kind == "mouse_down":
        return "mouse_down"
    if kind == "mouse_up":
        return "mouse_up"
    if kind == "left_click":
        return "left_click"
    return f"- {kind}"


def _format_history_step(index: int, step: StepResult) -> str:
    action_text = "null" if step.action is None else json.dumps(
        step.action.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    details = [f"done={str(step.done).lower()}"]
    success = step.info.get("success")
    if success is not None:
        details.append(f"success={str(bool(success)).lower()}")
    executed_kind = step.info.get("executed_kind")
    if executed_kind is not None:
        details.append(f"executed_kind={executed_kind}")
    return f"{index}. action={action_text}; " + "; ".join(details)


def _render_task_action_block(
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: tuple[str, ...],
    budget: int | None,
    task_type: str | None,
) -> str:
    del obs, history, allowed_kinds, budget, task_type
    return render_raw_unified_contract_block()


def _render_tool_call_prompt(block: str, *, allowed_kinds: tuple[str, ...]) -> str:
    del block, allowed_kinds
    return render_raw_unified_contract_block()


def _render_holo3_xml_tool_prompt(block: str, *, allowed_kinds: tuple[str, ...]) -> str:
    del block, allowed_kinds
    return render_raw_unified_contract_block()


def _render_thought_action_prompt(block: str, *, allowed_kinds: tuple[str, ...]) -> str:
    del block, allowed_kinds
    return render_raw_unified_contract_block()


def _thought_action_signature(kind: str) -> str:
    if kind in {"click", "drag", "move_to", "mouse_down", "mouse_up", "left_click"}:
        return kind
    return f"{kind}()"


def _render_opencua_pyautogui_prompt(block: str, *, allowed_kinds: tuple[str, ...]) -> str:
    del block, allowed_kinds
    return render_raw_unified_contract_block()


def _pyautogui_signature(kind: str) -> str:
    if kind in {"click", "drag", "move_to", "mouse_down", "mouse_up", "left_click"}:
        return kind
    return f"- {kind}: pyautogui.{kind}()"


def _normalize_tool_call_output(
    raw: str,
    *,
    allowed_kinds: tuple[str, ...],
    image_size_px: tuple[int, int] | None,
    current_cursor_xy: tuple[float, float] | None,
    coordinate_mode: PyAutoGUICoordinateMode = "relative_0_1000",
    strict_action_names: tuple[str, ...] | None = None,
    allow_left_click_drag_duration: bool = False,
    strict: bool = False,
) -> Action:
    parse_raw = _strip_native_reasoning_prefix(raw) if strict else raw
    if strict:
        payloads = _extract_strict_tool_call_payloads(parse_raw)
        if payloads is None:
            raise ValueError(
                "strict computer_use output requires exactly one Action line and "
                "one tool call, or the exact mouse_move + left_click_drag macro"
            )
        if len(payloads) == 2:
            return _action_from_native_drag_macro(
                payloads,
                allowed_kinds=allowed_kinds,
                image_size_px=image_size_px,
                coordinate_mode=coordinate_mode,
                strict_action_names=strict_action_names,
                allow_left_click_drag_duration=allow_left_click_drag_duration,
            )
        if len(payloads) != 1:
            raise ValueError(
                "strict computer_use output permits only one tool call or the exact "
                "two-call native drag macro"
            )
        payload = payloads[0]
    else:
        payload = _extract_tool_call_payload(parse_raw)
    if payload is None:
        return parse_action_blob(raw)
    name = str(payload.get("name") or payload.get("function") or "computer_use")
    args = payload.get("arguments") or payload.get("args")
    if args is None and any(key in payload for key in ("action", "type", "kind")):
        args = payload
    if args is None:
        args = {}
    if not isinstance(args, dict):
        if strict:
            raise ValueError("computer_use arguments must be a JSON object")
        args = {}
    if strict and name != "computer_use":
        raise ValueError(f"strict tool call name must be 'computer_use', got {name!r}")
    if name != "computer_use" and "action" not in args:
        args = {"action": name, **args}
    return _action_from_tool_args(
        args,
        allowed_kinds=allowed_kinds,
        image_size_px=image_size_px,
        current_cursor_xy=current_cursor_xy,
        coordinate_mode=coordinate_mode,
        strict_action_names=strict_action_names,
        allow_left_click_drag_duration=allow_left_click_drag_duration,
        strict=strict,
    )


def _normalize_holo3_structured_output(
    raw: str,
    *,
    allowed_kinds: tuple[str, ...],
    image_size_px: tuple[int, int] | None,
    current_cursor_xy: tuple[float, float] | None,
    strict_action_names: tuple[str, ...] | None = None,
    strict: bool = False,
) -> Action:
    cleaned = _strip_native_reasoning_prefix(raw)
    payload: dict[str, Any] | None = None
    starts = [re.match(r"\A\s*\{", cleaned)] if strict else re.finditer(r"\{", cleaned)
    for match in starts:
        if match is None:
            continue
        try:
            candidate, end = json.JSONDecoder().raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and (not strict or not cleaned[match.start() + end:].strip()):
            payload = candidate
            break
    if payload is not None:
        if strict:
            _validate_holo3_payload(payload)
        tool_call = payload.get("tool_call")
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("tool_name") or tool_call.get("name")
            args = {key: value for key, value in tool_call.items() if key not in {"tool_name", "name"}}
            args["action"] = tool_name
            return _action_from_tool_args(
                args,
                allowed_kinds=allowed_kinds,
                image_size_px=image_size_px,
                current_cursor_xy=current_cursor_xy,
                coordinate_mode="relative_0_1000",
                strict_action_names=strict_action_names,
                strict=strict,
            )
    if strict:
        raise ValueError("Holo3 strict output must be one structured JSON object")
    # Retain XML parsing as a compatibility fallback for older Holo checkpoints.
    return _normalize_tool_call_output(
        raw,
        allowed_kinds=allowed_kinds,
        image_size_px=image_size_px,
        current_cursor_xy=current_cursor_xy,
        coordinate_mode="relative_0_1000",
        strict_action_names=None,
        strict=False,
    )


def _strip_native_reasoning_prefix(raw: str) -> str:
    text = raw.strip()
    if text.startswith("<think>"):
        closing_index = text.find("</think>")
        if closing_index >= 0:
            return text[closing_index + len("</think>"):].strip()
    closing_index = text.find("</think>")
    if closing_index >= 0 and not text.startswith("{"):
        return text[closing_index + len("</think>"):].strip()
    return text


def _extract_strict_tool_call_payloads(raw: str) -> list[dict[str, Any]] | None:
    action_match = re.match(
        r"\A[ \t\r\n]*Action:[ \t]*([^\r\n]*\S)[ \t]*\r?\n",
        raw,
    )
    if action_match is None:
        return None

    body = raw[action_match.end():]
    block_pattern = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    payloads: list[dict[str, Any]] = []
    position = 0
    for match in block_pattern.finditer(body):
        if body[position:match.start()].strip():
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"strict tool call must contain valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("strict tool call payload must be a JSON object")
        if set(payload) != {"name", "arguments"}:
            raise ValueError("strict tool call payload requires exactly name and arguments")
        if payload.get("name") != "computer_use":
            raise ValueError("strict tool call name must be computer_use")
        if not isinstance(payload.get("arguments"), dict):
            raise ValueError("strict tool call arguments must be a JSON object")
        payloads.append(payload)
        position = match.end()
    if not payloads or body[position:].strip():
        return None
    return payloads


def _action_from_native_drag_macro(
    payloads: list[dict[str, Any]],
    *,
    allowed_kinds: tuple[str, ...],
    image_size_px: tuple[int, int] | None,
    coordinate_mode: PyAutoGUICoordinateMode,
    strict_action_names: tuple[str, ...] | None,
    allow_left_click_drag_duration: bool,
) -> AtomicAction:
    if len(payloads) != 2:
        raise ValueError("native drag macro requires exactly two tool calls")
    first_args = payloads[0]["arguments"]
    second_args = payloads[1]["arguments"]
    if (
        first_args.get("action") != "mouse_move"
        or second_args.get("action") != "left_click_drag"
    ):
        raise ValueError(
            "multiple tool calls are allowed only for mouse_move followed by "
            "left_click_drag"
        )

    move_action = _action_from_tool_args(
        first_args,
        allowed_kinds=allowed_kinds,
        image_size_px=image_size_px,
        current_cursor_xy=None,
        coordinate_mode=coordinate_mode,
        strict_action_names=strict_action_names,
        allow_left_click_drag_duration=allow_left_click_drag_duration,
        strict=True,
    )
    if not isinstance(move_action, PrimitiveAction) or move_action.kind != "move_to":
        raise ValueError("native drag macro must start with mouse_move")
    assert move_action.x is not None and move_action.y is not None
    drag_action = _action_from_tool_args(
        second_args,
        allowed_kinds=allowed_kinds,
        image_size_px=image_size_px,
        current_cursor_xy=None,
        current_cursor_relative_xy=(move_action.x, move_action.y),
        coordinate_mode=coordinate_mode,
        strict_action_names=strict_action_names,
        allow_left_click_drag_duration=allow_left_click_drag_duration,
        strict=True,
    )
    if not isinstance(drag_action, AtomicAction) or drag_action.kind != "drag":
        raise ValueError("native drag macro must end with left_click_drag")
    return drag_action


def _validate_holo3_payload(payload: dict[str, Any]) -> None:
    if not set(payload).issubset({"note", "thought", "tool_call"}):
        raise ValueError("Holo3 output contains unsupported top-level fields")
    if "thought" not in payload or not isinstance(payload["thought"], str):
        raise ValueError("Holo3 output requires a thought string")
    if "tool_call" not in payload or not isinstance(payload["tool_call"], dict):
        raise ValueError("Holo3 output requires one tool_call object")
    if "note" in payload and payload["note"] is not None and not isinstance(payload["note"], str):
        raise ValueError("Holo3 note must be a string or null")
    tool_call = payload["tool_call"]
    if "tool_name" not in tool_call or not isinstance(tool_call["tool_name"], str):
        raise ValueError("Holo3 tool_call requires tool_name")
    tool_name = tool_call["tool_name"]
    expected_fields = {
        "move_to": {"tool_name", "x", "y"},
        "mouse_down": {"tool_name"},
        "mouse_up": {"tool_name"},
        "left_click": {"tool_name"},
        "drag": {"tool_name", "points"},
        "click": {"tool_name", "x", "y"},
    }.get(tool_name)
    if expected_fields is None:
        raise ValueError(f"Holo3 tool_name is not supported: {tool_name!r}")
    if set(tool_call) != expected_fields:
        raise ValueError(
            f"Holo3 {tool_name} tool_call requires exactly {sorted(expected_fields)}"
        )


def _extract_tool_call_payload(raw: str) -> dict[str, Any] | None:
    json_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    if json_match:
        tool_call_text = json_match.group(1).strip()
        try:
            decoder = json.JSONDecoder()
            payload, _end = decoder.raw_decode(tool_call_text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            recovered = _recover_malformed_tool_call_payload(tool_call_text)
            if recovered is not None:
                return recovered
    xml_match = re.search(r"<function=([^>\s]+)>(.*?)</function>", raw, re.DOTALL)
    if xml_match:
        params: dict[str, Any] = {}
        for key, value in re.findall(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", xml_match.group(2), re.DOTALL):
            parsed = _parse_jsonish(value.strip())
            if key in params:
                existing = params[key]
                if isinstance(existing, list) and existing and all(isinstance(item, (list, tuple)) for item in existing):
                    params[key] = [*existing, parsed]
                else:
                    params[key] = [existing, parsed]
            else:
                params[key] = parsed
        return {"name": xml_match.group(1), "arguments": params}
    for brace_match in re.finditer(r"\{", raw):
        candidate = raw[brace_match.start():].strip()
        try:
            decoder = json.JSONDecoder()
            payload, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            recovered = _recover_malformed_tool_call_payload(candidate)
            if recovered is not None:
                return recovered
            continue
        if isinstance(payload, dict) and (
            "arguments" in payload
            or "args" in payload
            or any(key in payload for key in ("action", "type", "kind"))
        ):
            return payload
    return _recover_malformed_tool_call_payload(raw)


def _recover_malformed_tool_call_payload(tool_call_text: str) -> dict[str, Any] | None:
    action_match = re.search(
        r'"(?:action|type|kind)"\s*:\s*"([^"]+)"',
        tool_call_text,
    )
    if action_match is None:
        return None
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', tool_call_text)
    name = name_match.group(1) if name_match is not None else "computer_use"
    args: dict[str, Any] = {"action": action_match.group(1)}
    points = _recover_numeric_points(tool_call_text)
    if points:
        if len(points) == 1:
            args["x"], args["y"] = points[0]
        else:
            args["points"] = points
    return {"name": name, "arguments": args}


def _recover_numeric_points(text: str) -> list[tuple[float, float]]:
    number = r"-?\d+(?:\.\d+)?"
    for key in ("point", "coordinate", "coordinates", "location", "x"):
        match = re.search(
            rf'"{key}"\s*:\s*\[\s*({number})\s*,\s*({number})\s*\]',
            text,
        )
        if match is not None:
            return [(float(match.group(1)), float(match.group(2)))]
    x_match = re.search(rf'"?x"?\s*:\s*({number})', text)
    y_match = re.search(rf'"?y"?\s*:\s*({number})', text)
    if x_match is not None and y_match is not None:
        return [(float(x_match.group(1)), float(y_match.group(1)))]
    x_unlabeled_y_match = re.search(rf'"x"\s*:\s*({number})\s*,\s*({number})', text)
    if x_unlabeled_y_match is not None:
        return [(float(x_unlabeled_y_match.group(1)), float(x_unlabeled_y_match.group(2)))]
    points_match = re.search(r'"points"\s*:\s*(\[.*)', text, re.DOTALL)
    if points_match is None:
        return []
    values = [float(value) for value in re.findall(number, points_match.group(1))]
    return [(values[index], values[index + 1]) for index in range(0, len(values) - 1, 2)]


def _parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _action_from_tool_args(
    args: dict[str, Any],
    *,
    allowed_kinds: tuple[str, ...],
    image_size_px: tuple[int, int] | None,
    current_cursor_xy: tuple[float, float] | None,
    current_cursor_relative_xy: tuple[float, float] | None = None,
    coordinate_mode: PyAutoGUICoordinateMode = "relative_0_1000",
    strict_action_names: tuple[str, ...] | None = None,
    allow_left_click_drag_duration: bool = False,
    strict: bool = False,
) -> Action:
    relative_cursor_xy = current_cursor_relative_xy or _cursor_to_relative_bins(
        current_cursor_xy,
        image_size_px,
    )
    external_action = args.get("action") or args.get("type") or args.get("kind")
    action = _normalize_external_kind(external_action)
    if action is None:
        raise ValueError(f"tool call did not include an action: {args!r}")
    if strict:
        _validate_strict_tool_args(
            args,
            action=action,
            external_action=external_action,
            strict_action_names=strict_action_names,
            coordinate_click_enabled="click" in allowed_kinds,
            allow_left_click_drag_duration=allow_left_click_drag_duration,
        )
    if action == "move_to":
        x, y = _convert_tool_point(
            _point_from_payload(args),
            mode=coordinate_mode,
            strict=strict,
        )
        return PrimitiveAction(kind="move_to", x=x, y=y)
    if action in {"mouse_down", "mouse_up", "left_click"}:
        point = _point_from_payload(args, required=False)
        if action == "left_click" and point is not None:
            if "click" not in allowed_kinds:
                if strict:
                    raise ValueError(
                        "coordinate-bearing left_click is not the coordinate-free "
                        "benchmark compatibility action"
                    )
                return PrimitiveAction(kind="left_click")
            converted = _convert_tool_point(point, mode=coordinate_mode, strict=strict)
            return AtomicAction(kind="click", points=[converted])
        return PrimitiveAction(kind=action)
    if action == "click":
        point = _point_from_payload(args, required=False)
        if point is None:
            return PrimitiveAction(kind="left_click")
        if "click" not in allowed_kinds:
            raise ValueError("coordinate click is not enabled for this evaluator")
        converted = _convert_tool_point(point, mode=coordinate_mode, strict=strict)
        return AtomicAction(kind="click", points=[converted])
    if action == "drag":
        points = [
            _convert_tool_point(point, mode=coordinate_mode, strict=strict)
            for point in _points_from_payload(args)
        ]
        if len(points) == 2:
            return AtomicAction(kind="drag", points=points)
        is_native_single_target_drag = external_action == "left_click_drag"
        if len(points) == 1 and relative_cursor_xy is not None and (is_native_single_target_drag or not strict):
            return AtomicAction(kind="drag", points=[relative_cursor_xy, points[0]])
        point = _point_from_payload(args, required=False)
        if point is not None and relative_cursor_xy is not None and (is_native_single_target_drag or not strict):
            converted = _convert_tool_point(point, mode=coordinate_mode, strict=strict)
            return AtomicAction(kind="drag", points=[relative_cursor_xy, converted])
        raise ValueError(f"drag tool call requires two points: {args!r}")
    if action == "wait":
        return WebAction(kind="wait")
    if action in {"terminate", "submit"}:
        return AtomicAction(kind="submit", points=[])
    return parse_action_blob(json.dumps({"kind": action, **args}, ensure_ascii=False))


def _validate_strict_tool_args(
    args: dict[str, Any],
    *,
    action: str,
    external_action: Any,
    strict_action_names: tuple[str, ...] | None,
    coordinate_click_enabled: bool,
    allow_left_click_drag_duration: bool,
) -> None:
    if not isinstance(external_action, str) or not external_action.strip():
        raise ValueError("tool arguments require a string action")
    if strict_action_names is not None and external_action not in strict_action_names:
        raise ValueError(f"strict tool action is not advertised by the prompt: {external_action!r}")
    common = {"action"}
    if action == "move_to":
        allowed = common | {"coordinate", "x", "y"}
    elif action in {"mouse_down", "mouse_up"}:
        allowed = common
    elif action == "left_click":
        allowed = common | ({"coordinate"} if coordinate_click_enabled else set())
    elif action == "click":
        allowed = common | {"coordinate", "x", "y"}
    elif action == "drag":
        allowed = common | {"coordinate", "points", "x", "y"}
        if external_action == "left_click_drag" and allow_left_click_drag_duration:
            allowed.add("duration")
    else:
        allowed = set(args)
    extra = set(args) - allowed
    if extra:
        raise ValueError(f"unsupported arguments for {action}: {sorted(extra)}")
    if action in {"mouse_down", "mouse_up"} and set(args) != {"action"}:
        raise ValueError(f"{action} takes no coordinates")
    point_keys = [key for key in ("coordinate",) if key in args]
    has_x = "x" in args
    has_y = "y" in args
    if has_x != has_y:
        raise ValueError(f"{action} requires both x and y when either is provided")
    native_point_fields = {"action", "coordinate"}
    if (
        external_action == "left_click_drag"
        and allow_left_click_drag_duration
        and "duration" in args
    ):
        native_point_fields.add("duration")
    if external_action in {"mouse_move", "left_click_drag"} and set(args) != native_point_fields:
        raise ValueError(f"{external_action} requires exactly one native coordinate field")
    if action == "left_click" and set(args) != {"action"}:
        if not coordinate_click_enabled or set(args) != {"action", "coordinate"}:
            raise ValueError(
                "coordinate-bearing left_click requires the ClickXY action setting "
                "and exactly one native coordinate field"
            )
        _validate_exact_integer_point(args["coordinate"], name="coordinate")
    if action in {"move_to", "click"}:
        encodings = len(point_keys) + int(has_x and has_y)
        if encodings != 1:
            raise ValueError(f"{action} requires exactly one unambiguous point")
        if point_keys:
            _validate_exact_integer_point(args[point_keys[0]], name=point_keys[0])
        else:
            _validate_integer_scalar(args["x"], name="x")
            _validate_integer_scalar(args["y"], name="y")
    if action == "drag":
        if external_action == "left_click_drag":
            encodings = len(point_keys) + int(has_x and has_y)
            if encodings != 1 or "points" in args:
                raise ValueError("left_click_drag requires exactly one target point")
            if point_keys:
                _validate_exact_integer_point(args[point_keys[0]], name=point_keys[0])
            else:
                _validate_integer_scalar(args["x"], name="x")
                _validate_integer_scalar(args["y"], name="y")
            if "duration" in args:
                _validate_non_negative_finite_number(
                    args["duration"],
                    name="left_click_drag duration",
                )
        elif "points" in args:
            if point_keys or has_x:
                raise ValueError("drag points cannot be combined with another coordinate encoding")
            points = args["points"]
            if not isinstance(points, (list, tuple)) or len(points) != 2:
                raise ValueError("drag points must contain exactly two points")
            _validate_exact_integer_point(points[0], name="drag start")
            _validate_exact_integer_point(points[1], name="drag end")
        else:
            raise ValueError(
                "benchmark compatibility drag requires exactly one "
                "points=[[x1,y1],[x2,y2]] field"
            )


def _validate_exact_integer_point(value: Any, *, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be exactly [x,y]")
    _validate_integer_scalar(value[0], name=f"{name} x")
    _validate_integer_scalar(value[1], name=f"{name} y")


def _validate_exact_numeric_point(value: Any, *, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be exactly [x,y]")
    _validate_non_negative_finite_number(value[0], name=f"{name} x")
    _validate_non_negative_finite_number(value[1], name=f"{name} y")


def _validate_integer_scalar(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer coordinate")
    if not float(value).is_integer():
        raise ValueError(f"{name} must be an integer coordinate")


def _validate_non_negative_finite_number(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _convert_tool_point(
    point: tuple[float, float],
    *,
    mode: PyAutoGUICoordinateMode,
    strict: bool,
) -> tuple[float, float]:
    x_value, y_value = point
    if strict:
        _require_integer_coordinate(x_value)
        _require_integer_coordinate(y_value)
    if mode in {"relative_0_999", "evocua_0_999"}:
        if strict:
            coordinate_label = "EvoCUA" if mode == "evocua_0_999" else "0..999"
            _require_coordinate_range(x_value, 0, 999, name=f"{coordinate_label} x")
            _require_coordinate_range(y_value, 0, 999, name=f"{coordinate_label} y")
        return (
            round(float(x_value) * 1000 / 999),
            round(float(y_value) * 1000 / 999),
        )
    if mode == "relative_0_1000":
        if strict:
            _require_coordinate_range(x_value, 0, 1000, name="x")
            _require_coordinate_range(y_value, 0, 1000, name="y")
        return (_clean_number(x_value), _clean_number(y_value))
    return (_clean_number(x_value), _clean_number(y_value))


def _cursor_to_relative_bins(
    cursor_xy: tuple[float, float] | None,
    image_size_px: tuple[int, int] | None,
) -> tuple[float, float] | None:
    if cursor_xy is None:
        return None
    if image_size_px is None or image_size_px[0] <= 0 or image_size_px[1] <= 0:
        return cursor_xy
    return (
        round(float(cursor_xy[0]) / image_size_px[0] * 1000),
        round(float(cursor_xy[1]) / image_size_px[1] * 1000),
    )


def _normalize_external_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    aliases = {
        "mouse_move": "move_to",
        "move": "move_to",
        "moveTo": "move_to",
        "hover": "move_to",
        "click": "click",
        "left_click": "left_click",
        "leftClick": "left_click",
        "mouseDown": "mouse_down",
        "mouse_down": "mouse_down",
        "mouseUp": "mouse_up",
        "mouse_up": "mouse_up",
        "left_click_drag": "drag",
        "dragTo": "drag",
        "drag": "drag",
        "terminate": "terminate",
        "answer": "submit",
        "submit": "submit",
        "wait": "wait",
    }
    return aliases.get(normalized, normalized)


def _point_from_payload(args: dict[str, Any], *, required: bool = True) -> tuple[float, float] | None:
    for key in ("point", "coordinate", "coordinates", "location"):
        value = args.get(key)
        point = _point_from_sequence(value)
        if point is not None:
            return point
    x_value = args.get("x")
    y_value = args.get("y")
    if y_value is None and isinstance(x_value, (list, tuple)) and len(x_value) >= 2:
        return (float(x_value[0]), float(x_value[1]))
    if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
        return (float(x_value), float(y_value))
    points = args.get("points")
    if isinstance(points, (list, tuple)) and points:
        first = points[0]
        point = _point_from_sequence(first)
        if point is not None:
            return point
        if len(points) >= 2 and all(isinstance(v, (int, float)) for v in points[:2]):
            return (float(points[0]), float(points[1]))
    if required:
        raise ValueError(f"action requires a point: {args!r}")
    return None


def _points_from_payload(args: dict[str, Any]) -> list[tuple[float, float]]:
    points = args.get("points")
    parsed: list[tuple[float, float]] = []
    if isinstance(points, (list, tuple)):
        if (
            len(points) == 1
            and isinstance(points[0], (list, tuple))
            and len(points[0]) >= 2
            and _point_from_sequence(points[0][0]) is not None
            and _point_from_sequence(points[0][1]) is not None
        ):
            points = points[0]
        if len(points) >= 4 and all(isinstance(v, (int, float)) for v in points[:4]):
            return [(float(points[0]), float(points[1])), (float(points[2]), float(points[3]))]
        if len(points) >= 2 and all(isinstance(v, (int, float)) for v in points[:2]):
            return [(float(points[0]), float(points[1]))]
        for point in points:
            parsed_point = _point_from_sequence(point)
            if parsed_point is not None:
                parsed.append(parsed_point)
    start = args.get("start_point") or args.get("start") or args.get("start_box")
    end = args.get("end_point") or args.get("end") or args.get("end_box")
    if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
        return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
    return parsed


def _point_from_sequence(value: Any) -> tuple[float, float] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        return (float(value[0]), float(value[1]))
    if isinstance(value, (list, tuple)) and value:
        box_points = [_point_from_sequence(item) for item in value[:2]]
        parsed = [point for point in box_points if point is not None]
        if len(parsed) == 1:
            return parsed[0]
        if len(parsed) >= 2:
            return (
                (parsed[0][0] + parsed[1][0]) / 2.0,
                (parsed[0][1] + parsed[1][1]) / 2.0,
            )
    return None


def _normalize_thought_action_output(
    raw: str,
    *,
    allowed_kinds: tuple[str, ...],
    strict: bool = False,
) -> Action:
    if strict:
        match = re.fullmatch(
            r"\s*Thought:\s*(\S.*?)\s*\nAction:\s*(\S.*?)\s*",
            raw,
            re.DOTALL,
        )
        if match is None:
            raise ValueError("strict UI-TARS/DART output requires Thought then exactly one Action")
        if "\n" in match.group(2).strip():
            raise ValueError("strict UI-TARS/DART Action must be one action call")
        action_text = match.group(2).strip()
    else:
        action_text = _extract_action_line(raw)
    name_match = re.match(r"\s*([A-Za-z_][\w]*)\s*(?:\((.*)\))?\s*$", action_text, re.DOTALL)
    if not name_match:
        raise ValueError(f"could not parse Action line: {action_text!r}")
    external_action = name_match.group(1)
    action = _normalize_external_kind(external_action)
    body = name_match.group(2) or ""
    points = _extract_tagged_points(body)
    if not points:
        points = _extract_numeric_points(body)
    if strict:
        if external_action not in allowed_kinds:
            raise ValueError(f"strict UI-TARS/DART action name is not enabled: {external_action!r}")
        _validate_strict_thought_action_arguments(
            external_action,
            body,
            point_count=len(points),
        )
    if action == "move_to":
        if not points:
            raise ValueError(f"move_to action needs a point: {action_text!r}")
        if strict:
            _require_integer_point(points[0])
        return PrimitiveAction(kind="move_to", x=points[0][0], y=points[0][1])
    if action in {"mouse_down", "mouse_up", "left_click"}:
        if strict and body.strip():
            raise ValueError(f"{action} must not include arguments")
        return PrimitiveAction(kind=action)
    if action == "click":
        if not points:
            return PrimitiveAction(kind="left_click")
        return AtomicAction(kind="click", points=[points[0]])
    if action == "drag":
        if len(points) < 2:
            raise ValueError(f"drag action needs two points: {action_text!r}")
        if strict and len(points) != 2:
            raise ValueError(f"strict drag action needs exactly two points: {action_text!r}")
        if strict:
            _require_integer_point(points[0])
            _require_integer_point(points[1])
        return AtomicAction(kind="drag", points=points[:2])
    if action in {"finished", "terminate", "submit"}:
        return AtomicAction(kind="submit", points=[])
    raise ValueError(f"unsupported thought/action kind {action!r}; allowed={allowed_kinds}")


def _validate_strict_thought_action_arguments(
    action: str,
    body: str,
    *,
    point_count: int,
) -> None:
    if action in {"mouse_down", "mouse_up", "left_click"}:
        if body.strip():
            raise ValueError(f"{action} must not include arguments")
        return
    if action in {"move_to", "click"}:
        match = re.fullmatch(
            r"\s*(?:start_box|point)\s*=\s*(['\"])(.*?)\1\s*",
            body,
            re.DOTALL,
        )
        if match is None or not _is_native_coordinate_string(match.group(2)) or point_count != 1:
            raise ValueError(f"{action} requires exactly one native coordinate argument")
        return
    if action == "drag":
        match = re.fullmatch(
            r"\s*(?:start_box|start_point)\s*=\s*(['\"])(.*?)\1\s*,\s*"
            r"(?:end_box|end_point)\s*=\s*(['\"])(.*?)\3\s*",
            body,
            re.DOTALL,
        )
        if (
            match is None
            or not _is_native_coordinate_string(match.group(2))
            or not _is_native_coordinate_string(match.group(4))
            or point_count != 2
        ):
            raise ValueError("drag requires one native start coordinate and one native end coordinate")
        return
    raise ValueError(f"unsupported strict UI-TARS/DART action: {action!r}")


def _is_native_coordinate_string(value: str) -> bool:
    number = r"[-+]?\d+(?:\.\d+)?"
    patterns = (
        rf"<\|box_start\|>\s*\(\s*{number}\s*,\s*{number}\s*\)\s*<\|box_end\|>",
        rf"<point>\s*{number}\s+{number}\s*</point>",
        rf"\(\s*{number}\s*,\s*{number}\s*\)",
    )
    return any(re.fullmatch(pattern, value.strip()) is not None for pattern in patterns)


def _extract_action_line(raw: str) -> str:
    match = re.search(r"Action:\s*(.+)", raw, re.DOTALL)
    text = match.group(1) if match else raw
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("no Action content found")
    return lines[0].strip("`")


def _extract_tagged_points(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    patterns = [
        r"<point>\s*([-\d.]+)\s+([-\d.]+)\s*</point>",
        r"<\|box_start\|>\s*\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*<\|box_end\|>",
    ]
    for pattern in patterns:
        for x_value, y_value in re.findall(pattern, text):
            points.append((float(x_value), float(y_value)))
    return points


def _extract_numeric_points(text: str) -> list[tuple[float, float]]:
    numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    return [
        (numbers[index], numbers[index + 1])
        for index in range(0, len(numbers) - 1, 2)
    ]


def _normalize_pyautogui_output(
    raw: str,
    *,
    allowed_kinds: tuple[str, ...],
    image_size_px: tuple[int, int] | None,
    native_image_size_px: tuple[int, int] | None,
    current_cursor_xy: tuple[float, float] | None,
    coordinate_mode: PyAutoGUICoordinateMode,
    strict: bool = False,
) -> Action:
    if strict:
        strict_code = _validate_strict_opencua_shell(raw)
        if coordinate_mode == "qwen25_smart_resize" and native_image_size_px is None:
            raise ValueError("OpenCUA strict parsing requires the Qwen2.5 native image size")
        calls = _extract_strict_pyautogui_calls(strict_code)
    else:
        calls = _extract_pyautogui_calls(raw)
    if not calls:
        if strict:
            raise ValueError("OpenCUA Code section did not contain a supported PyAutoGUI call")
        return parse_action_blob(raw)
    if strict:
        call_names = [name for name, _kwargs in calls]
        if call_names in (["moveTo", "dragTo"], ["moveTo", "dragRel"]):
            _validate_strict_xy_call(calls[0][1], call_name="pyautogui.moveTo")
            start = _convert_point(
                *_xy_from_call_kwargs(calls[0][1]),
                image_size_px=image_size_px,
                native_image_size_px=native_image_size_px,
                mode=coordinate_mode,
                strict=True,
            )
            if call_names[-1] == "dragTo":
                drag_x, drag_y = _validate_strict_drag_call(
                    calls[1][1], call_name="pyautogui.dragTo"
                )
                end = _convert_point(
                    drag_x,
                    drag_y,
                    image_size_px=image_size_px,
                    native_image_size_px=native_image_size_px,
                    mode=coordinate_mode,
                    strict=True,
                )
            else:
                offset_x, offset_y = _validate_strict_drag_call(
                    calls[1][1], call_name="pyautogui.dragRel"
                )
                end = _drag_relative_endpoint(
                    start,
                    offset=(offset_x, offset_y),
                    native_image_size_px=native_image_size_px,
                )
            if "drag" not in allowed_kinds:
                raise ValueError("drag is not enabled for this evaluator")
            return AtomicAction(kind="drag", points=[start, end])
        if len(calls) != 1:
            raise ValueError(
                "OpenCUA strict output requires one PyAutoGUI call, except a "
                "moveTo+dragTo/dragRel drag macro"
            )
        name, kwargs = calls[0]
        if name == "moveTo":
            _validate_strict_xy_call(kwargs, call_name="pyautogui.moveTo")
            point = _convert_point(
                *_xy_from_call_kwargs(kwargs),
                image_size_px=image_size_px,
                native_image_size_px=native_image_size_px,
                mode=coordinate_mode,
                strict=True,
            )
            return PrimitiveAction(kind="move_to", x=point[0], y=point[1])
        if name == "mouseDown":
            _require_no_call_coordinates(kwargs, name)
            return PrimitiveAction(kind="mouse_down")
        if name == "mouseUp":
            _require_no_call_coordinates(kwargs, name)
            return PrimitiveAction(kind="mouse_up")
        if name == "click":
            _validate_strict_click_call(kwargs, coordinate_click_enabled="click" in allowed_kinds)
            point = _xy_from_call_kwargs(kwargs, required=False)
            if point is None:
                return PrimitiveAction(kind="left_click")
            if "click" not in allowed_kinds:
                raise ValueError("coordinate click is not enabled for this evaluator")
            converted = _convert_point(
                *point,
                image_size_px=image_size_px,
                native_image_size_px=native_image_size_px,
                mode=coordinate_mode,
                strict=True,
            )
            return AtomicAction(kind="click", points=[converted])
        if name in {"dragTo", "dragRel"}:
            if "drag" not in allowed_kinds:
                raise ValueError("drag is not enabled for this evaluator")
            start = _current_cursor_relative_point(
                current_cursor_xy,
                image_size_px=image_size_px,
            )
            drag_x, drag_y = _validate_strict_drag_call(
                kwargs,
                call_name=f"pyautogui.{name}",
            )
            if name == "dragTo":
                end = _convert_point(
                    drag_x,
                    drag_y,
                    image_size_px=image_size_px,
                    native_image_size_px=native_image_size_px,
                    mode=coordinate_mode,
                    strict=True,
                )
            else:
                end = _drag_relative_endpoint(
                    start,
                    offset=(drag_x, drag_y),
                    native_image_size_px=native_image_size_px,
                )
            return AtomicAction(kind="drag", points=[start, end])
        raise ValueError(f"unsupported OpenCUA strict PyAutoGUI call: {name}")
    last_move: tuple[float, float] | None = None
    pending_move: tuple[float, float] | None = None
    for name, kwargs in calls:
        if name in {"moveTo", "move_to"}:
            last_move = _convert_point(
                *_xy_from_call_kwargs(kwargs),
                image_size_px=image_size_px,
                native_image_size_px=native_image_size_px,
                mode=coordinate_mode,
                strict=False,
            )
            pending_move = last_move
            continue
        if name in {"mouseDown", "mouse_down"}:
            if pending_move is not None:
                return PrimitiveAction(kind="move_to", x=pending_move[0], y=pending_move[1])
            return PrimitiveAction(kind="mouse_down")
        if name in {"mouseUp", "mouse_up"}:
            if pending_move is not None:
                return PrimitiveAction(kind="move_to", x=pending_move[0], y=pending_move[1])
            return PrimitiveAction(kind="mouse_up")
        if name == "click":
            if pending_move is not None:
                return PrimitiveAction(kind="move_to", x=pending_move[0], y=pending_move[1])
            point = _xy_from_call_kwargs(kwargs, required=False)
            if point is None:
                return PrimitiveAction(kind="left_click")
            converted = _convert_point(
                *point,
                image_size_px=image_size_px,
                native_image_size_px=native_image_size_px,
                mode=coordinate_mode,
                strict=False,
            )
            if "click" in allowed_kinds:
                return AtomicAction(kind="click", points=[converted])
            return PrimitiveAction(kind="move_to", x=converted[0], y=converted[1])
        if name == "dragTo":
            end = _convert_point(
                *_xy_from_call_kwargs(kwargs),
                image_size_px=image_size_px,
                native_image_size_px=native_image_size_px,
                mode=coordinate_mode,
                strict=False,
            )
            if last_move is not None and "drag" in allowed_kinds:
                return AtomicAction(kind="drag", points=[last_move, end])
            return PrimitiveAction(kind="move_to", x=end[0], y=end[1])
        if name in {"terminate", "submit"}:
            if pending_move is not None:
                return PrimitiveAction(kind="move_to", x=pending_move[0], y=pending_move[1])
            return AtomicAction(kind="submit", points=[])
    if pending_move is not None:
        return PrimitiveAction(kind="move_to", x=pending_move[0], y=pending_move[1])
    raise ValueError(f"no supported pyautogui call found in response: {raw!r}")


def _validate_strict_opencua_shell(raw: str) -> str:
    match = re.fullmatch(
        r"\s*# Step\s+[^:\n]+:\s*\n"
        r"## Thought:\s*\n(.+?\S)\s*\n+"
        r"## Action:\s*\n([^\n]+?)\s*\n+"
        r"## Code:\s*\n```(?:python|py)\s*\n(.*?)\n```\s*",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("OpenCUA strict output must contain one Step/Thought/Action/Code shell")
    if not match.group(1).strip() or not match.group(2).strip() or not match.group(3).strip():
        raise ValueError("OpenCUA Thought, Action, and Code sections must be non-empty")
    return match.group(3).strip()


def _require_no_call_coordinates(kwargs: dict[str, Any], call_name: str) -> None:
    if kwargs:
        raise ValueError(f"{call_name} takes no arguments in the canonical action space")


def _extract_strict_pyautogui_calls(code: str) -> list[tuple[str, dict[str, Any]]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"OpenCUA Code section is not valid Python: {exc}") from exc
    calls: list[tuple[str, dict[str, Any]]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError("OpenCUA strict Code section may contain only PyAutoGUI call expressions")
        node = statement.value
        if (
            not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
            or node.func.value.id != "pyautogui"
        ):
            raise ValueError("OpenCUA strict calls must use the pyautogui. prefix")
        calls.append((node.func.attr, _call_kwargs(node)))
    return calls


def _validate_strict_xy_call(kwargs: dict[str, Any], *, call_name: str) -> None:
    if set(kwargs) not in ({"x", "y"}, {"arg0", "arg1"}):
        raise ValueError(f"{call_name} requires exactly x and y")
    x_key, y_key = ("x", "y") if "x" in kwargs else ("arg0", "arg1")
    _validate_integer_scalar(kwargs[x_key], name=f"{call_name} x")
    _validate_integer_scalar(kwargs[y_key], name=f"{call_name} y")
    _xy_from_call_kwargs(kwargs)


def _validate_strict_click_call(
    kwargs: dict[str, Any],
    *,
    coordinate_click_enabled: bool,
) -> None:
    if not kwargs:
        return
    if not coordinate_click_enabled:
        raise ValueError("canonical pyautogui.click() takes no arguments")
    _validate_strict_xy_call(kwargs, call_name="pyautogui.click")


def _validate_strict_drag_call(
    kwargs: dict[str, Any],
    *,
    call_name: str,
) -> tuple[float, float]:
    coordinate_keys = (
        {"x", "y"}
        if "x" in kwargs or "y" in kwargs
        else {"arg0", "arg1"}
    )
    optional_keys = {"duration", "button", "arg2"}
    if not coordinate_keys.issubset(kwargs) or set(kwargs) - coordinate_keys - optional_keys:
        raise ValueError(f"{call_name} requires x and y with optional duration and left button")
    if "arg2" in kwargs and "duration" in kwargs:
        raise ValueError(f"{call_name} specifies duration twice")
    if kwargs.get("button", "left") != "left":
        raise ValueError(f"{call_name} requires the left button")
    duration = kwargs.get("duration", kwargs.get("arg2"))
    if duration is not None:
        _validate_non_negative_finite_number(duration, name=f"{call_name} duration")
    x_value, y_value = _xy_from_call_kwargs(kwargs)
    _validate_integer_scalar(x_value, name=f"{call_name} x")
    _validate_integer_scalar(y_value, name=f"{call_name} y")
    return x_value, y_value


def _current_cursor_relative_point(
    current_cursor_xy: tuple[float, float] | None,
    *,
    image_size_px: tuple[int, int] | None,
) -> tuple[float, float]:
    if current_cursor_xy is None or image_size_px is None:
        raise ValueError("OpenCUA current-cursor drag requires cursor and screenshot sizes")
    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("OpenCUA current-cursor drag requires positive screenshot sizes")
    x_value, y_value = current_cursor_xy
    _require_coordinate_range(x_value, 0, width, name="cursor x")
    _require_coordinate_range(y_value, 0, height, name="cursor y")
    return (
        round(float(x_value) / width * 1000),
        round(float(y_value) / height * 1000),
    )


def _drag_relative_endpoint(
    start: tuple[float, float],
    *,
    offset: tuple[float, float],
    native_image_size_px: tuple[int, int] | None,
) -> tuple[float, float]:
    if (
        native_image_size_px is None
        or native_image_size_px[0] <= 0
        or native_image_size_px[1] <= 0
    ):
        raise ValueError("OpenCUA dragRel requires positive native image sizes")
    width, height = native_image_size_px
    return (
        round(float(start[0]) + float(offset[0]) / width * 1000),
        round(float(start[1]) + float(offset[1]) / height * 1000),
    )


def _extract_pyautogui_calls(raw: str) -> list[tuple[str, dict[str, Any]]]:
    code = _extract_code_text(raw)
    calls: list[tuple[str, dict[str, Any]]] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if call_name is None:
                continue
            kwargs = _call_kwargs(node)
            calls.append((call_name, kwargs))
    return calls


def _extract_code_text(raw: str) -> str:
    fenced = re.findall(r"```(?:python|py|code)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        return "\n".join(fenced)
    if "## Code:" in raw:
        return raw.split("## Code:", 1)[1]
    candidate_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("pyautogui.", "computer.")):
            candidate_lines.append(stripped)
    return "\n".join(candidate_lines) if candidate_lines else raw


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name) and base.id in {"pyautogui", "computer"}:
            return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _call_kwargs(node: ast.Call) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for index, arg in enumerate(node.args):
        kwargs[f"arg{index}"] = _literal(arg)
    for keyword in node.keywords:
        if keyword.arg is not None:
            kwargs[keyword.arg] = _literal(keyword.value)
    return kwargs


def _xy_from_call_kwargs(kwargs: dict[str, Any], *, required: bool = True) -> tuple[float, float] | None:
    x_value = kwargs.get("x", kwargs.get("arg0"))
    y_value = kwargs.get("y", kwargs.get("arg1"))
    if x_value is not None and y_value is not None:
        return (float(x_value), float(y_value))
    coordinate = kwargs.get("coordinate") or kwargs.get("point")
    if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
        return (float(coordinate[0]), float(coordinate[1]))
    if required:
        raise ValueError(f"pyautogui call requires x/y coordinates: {kwargs!r}")
    return None


def _convert_point(
    x_value: float,
    y_value: float,
    *,
    image_size_px: tuple[int, int] | None,
    native_image_size_px: tuple[int, int] | None = None,
    mode: PyAutoGUICoordinateMode,
    strict: bool = False,
) -> tuple[float, float]:
    if strict:
        _require_integer_coordinate(x_value)
        _require_integer_coordinate(y_value)
    if mode == "pixel" and image_size_px is not None:
        width, height = image_size_px
        if width > 0 and height > 0:
            if strict:
                _require_coordinate_range(x_value, 0, width, name="pixel x")
                _require_coordinate_range(y_value, 0, height, name="pixel y")
            return (round(float(x_value) / width * 1000), round(float(y_value) / height * 1000))
    if mode == "qwen25_smart_resize" and native_image_size_px is not None:
        width, height = native_image_size_px
        if width > 0 and height > 0:
            if strict:
                _require_coordinate_range(x_value, 0, width, name="Qwen2.5 x")
                _require_coordinate_range(y_value, 0, height, name="Qwen2.5 y")
            return (
                round(float(x_value) / width * 1000),
                round(float(y_value) / height * 1000),
            )
    if mode == "evocua_0_999":
        return _convert_tool_point(
            (float(x_value), float(y_value)),
            mode=mode,
            strict=strict,
        )
    if strict and mode == "relative_0_1000":
        _require_coordinate_range(x_value, 0, 1000, name="x")
        _require_coordinate_range(y_value, 0, 1000, name="y")
    return (_clean_number(x_value), _clean_number(y_value))


def _convert_ui_venus_action_coordinates(
    action: Action,
    *,
    image_size_px: tuple[int, int] | None,
    native_image_size_px: tuple[int, int] | None,
    strict: bool,
) -> Action:
    if image_size_px is None or native_image_size_px is None:
        return action
    original_width, original_height = image_size_px
    native_width, native_height = native_image_size_px
    if min(original_width, original_height, native_width, native_height) <= 0:
        return action

    def convert_point(x_value: float, y_value: float) -> tuple[int, int]:
        if strict:
            _require_coordinate_range(x_value, 0, native_width, name="UI-Venus native x")
            _require_coordinate_range(y_value, 0, native_height, name="UI-Venus native y")
        pixel_x = int(float(x_value) * original_width / native_width)
        pixel_y = int(float(y_value) * original_height / native_height)
        pixel_x = min(max(pixel_x, 0), original_width)
        pixel_y = min(max(pixel_y, 0), original_height)
        return (
            round(pixel_x / original_width * 1000),
            round(pixel_y / original_height * 1000),
        )

    if isinstance(action, PrimitiveAction) and action.kind == "move_to":
        assert action.x is not None and action.y is not None
        x_value, y_value = convert_point(float(action.x), float(action.y))
        return PrimitiveAction(kind="move_to", x=x_value, y=y_value)
    if isinstance(action, AtomicAction) and action.points:
        return AtomicAction(
            kind=action.kind,
            points=[convert_point(float(x), float(y)) for x, y in action.points],
            answer=action.answer,
        )
    if isinstance(action, WebAction) and action.points:
        return WebAction(
            kind=action.kind,
            points=[convert_point(float(x), float(y)) for x, y in action.points],
            direction=action.direction,
            content=action.content,
            key=action.key,
            app=action.app,
            url=action.url,
        )
    return action


def _convert_action_coordinates(
    action: Action,
    *,
    native_image_size_px: tuple[int, int] | None,
    mode: PyAutoGUICoordinateMode,
    strict: bool = False,
) -> Action:
    if native_image_size_px is None:
        return action
    if isinstance(action, PrimitiveAction) and action.kind == "move_to":
        assert action.x is not None and action.y is not None
        x_value, y_value = _convert_point(
            float(action.x),
            float(action.y),
            image_size_px=None,
            native_image_size_px=native_image_size_px,
            mode=mode,
            strict=strict,
        )
        return PrimitiveAction(kind="move_to", x=x_value, y=y_value)
    if isinstance(action, AtomicAction) and action.points:
        points = [
            _convert_point(
                float(x_value),
                float(y_value),
                image_size_px=None,
                native_image_size_px=native_image_size_px,
                mode=mode,
                strict=strict,
            )
            for x_value, y_value in action.points
        ]
        return AtomicAction(kind=action.kind, points=points, answer=action.answer)
    if isinstance(action, WebAction) and action.points:
        points = [
            _convert_point(
                float(x_value),
                float(y_value),
                image_size_px=None,
                native_image_size_px=native_image_size_px,
                mode=mode,
                strict=strict,
            )
            for x_value, y_value in action.points
        ]
        return WebAction(
            kind=action.kind,
            points=points,
            direction=action.direction,
            content=action.content,
            key=action.key,
            app=action.app,
            url=action.url,
        )
    return action


def _validate_action_coordinates(action: Action, *, strict: bool) -> Action:
    if not strict:
        return action
    if isinstance(action, PrimitiveAction) and action.kind == "move_to":
        assert action.x is not None and action.y is not None
        _require_coordinate_range(action.x, 0, 1000, name="canonical x")
        _require_coordinate_range(action.y, 0, 1000, name="canonical y")
    elif isinstance(action, (AtomicAction, WebAction)):
        for x_value, y_value in action.points:
            _require_coordinate_range(x_value, 0, 1000, name="canonical x")
            _require_coordinate_range(y_value, 0, 1000, name="canonical y")
    return action


def _require_integer_point(point: tuple[float, float]) -> None:
    _require_integer_coordinate(point[0])
    _require_integer_coordinate(point[1])


def _require_integer_coordinate(value: float) -> None:
    if not float(value).is_integer():
        raise ValueError(f"native coordinates must be integers, got {value!r}")


def _require_coordinate_range(
    value: float,
    lower: float,
    upper: float,
    *,
    name: str,
) -> None:
    number = float(value)
    if number < lower or number > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value!r}")


def _clean_number(value: float) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number
