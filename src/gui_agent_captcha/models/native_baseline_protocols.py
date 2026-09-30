from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from ..actions import Action, AtomicAction, PrimitiveAction
from ..core import Observation
from ..prompts.unified_three_action import UNIFIED_THREE_ACTION_SYSTEM_PROMPT
from ..protocol_tracks import canonicalize_task_requirement

REQUIRED_CANONICAL_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
)

QWEN25_FACTOR = 28
QWEN25_MAX_RATIO = 200
UITARS_MIN_PIXELS = 100 * QWEN25_FACTOR * QWEN25_FACTOR
QWEN25_MAX_PIXELS = 16384 * QWEN25_FACTOR * QWEN25_FACTOR
OPENCUA_MIN_PIXELS = 4 * QWEN25_FACTOR * QWEN25_FACTOR
UI_VENUS_MIN_PIXELS = 3136
UI_VENUS_MAX_PIXELS = 12845056
UI_VENUS_NATIVE_COORDINATE_FORMAT = (
    "qwen_smart_resize_pixels_to_repo_0_1000"
)

UI_VENUS_WEB_ACTION_KINDS = (
    "Click",
    "Drag",
    "Scroll",
    "Type",
    "Launch",
    "Wait",
    "Finished",
    "CallUser",
    "LongPress",
    "PressBack",
    "PressHome",
    "PressEnter",
    "PressRecent",
    "Hover",
    "DoubleClick",
    "Hotkey",
)


def native_action_kinds(allowed_kinds: Iterable[str]) -> tuple[str, ...]:
    """Return the benchmark action set while preserving the required five actions."""

    ordered: list[str] = list(REQUIRED_CANONICAL_ACTION_KINDS)
    for value in allowed_kinds:
        kind = str(value).strip()
        if kind and kind not in ordered:
            ordered.append(kind)
    return tuple(ordered)


def native_model_action_kinds(
    style: str,
    allowed_kinds: Iterable[str],
) -> tuple[str, ...]:
    """Return model-facing names for the benchmark action semantics."""

    if style == "ui_venus":
        del allowed_kinds
        return UI_VENUS_WEB_ACTION_KINDS

    canonical_actions = native_action_kinds(allowed_kinds)
    if style == "mai_ui":
        # MAI-UI's native mobile_use contract has one coordinate-bearing click
        # action.  Do not expose a second, conflicting left_click spelling.
        return tuple(
            dict.fromkeys(
                "click" if kind == "left_click" else kind
                for kind in canonical_actions
            )
        )
    if style == "ui_voyager":
        return tuple("swipe" if kind == "drag" else kind for kind in canonical_actions)
    if style not in {"gui_owl", "evocua_s2"}:
        return canonical_actions

    exposed: list[str] = []
    for kind in canonical_actions:
        if kind == "move_to":
            model_kind = "mouse_move"
        elif kind == "click" and style == "evocua_s2":
            # EvoCUA natively expresses coordinate clicks as
            # left_click(coordinate=[x,y]). GUI-Owl uses click.
            model_kind = "left_click"
        else:
            model_kind = kind
        if model_kind not in exposed:
            exposed.append(model_kind)

    # Native left_click_drag starts at the current cursor, so it remains distinct
    # from the benchmark's complete two-point drag(start, end) compatibility action.
    if "drag" in canonical_actions and "left_click_drag" not in exposed:
        exposed.append("left_click_drag")
    native_first = [
        kind
        for kind in ("mouse_move", "click", "left_click", "left_click_drag")
        if kind in exposed
    ]
    return tuple(native_first + [kind for kind in exposed if kind not in native_first])


def native_system_prompt(style: str, *, allowed_kinds: Iterable[str]) -> str:
    del allowed_kinds
    if style == "ui_venus":
        return "You are a helpful assistant."
    return UNIFIED_THREE_ACTION_SYSTEM_PROMPT


def native_six_action_system_prompt(
    style: str,
    *,
    allowed_kinds: Iterable[str],
) -> str:
    """Render the prior eight-agent prompt shell with all six benchmark actions."""

    actions = native_action_kinds(allowed_kinds)
    if style in {"uitars", "dart_gui"}:
        return "You are a helpful assistant."
    if style == "gui_owl":
        return _tool_call_system_prompt(
            native_model_action_kinds(style, actions),
            model="gui_owl",
            coordinate_click_enabled="click" in actions,
        )
    if style == "evocua_s2":
        return _tool_call_system_prompt(
            native_model_action_kinds(style, actions),
            model="evocua",
            coordinate_click_enabled="click" in actions,
        )
    if style == "holo3_tool":
        return _holo3_structured_system_prompt(actions)
    if style == "opencua_pyautogui":
        return _opencua_l2_system_prompt(actions)
    if style == "mai_ui":
        return _mai_ui_system_prompt(native_model_action_kinds(style, actions))
    if style == "ui_voyager":
        return _ui_voyager_system_prompt(native_model_action_kinds(style, actions))
    raise ValueError(f"native six-action prompt is unsupported for style {style!r}")


def native_task_prompt(
    style: str,
    obs: Observation,
    *,
    allowed_kinds: Iterable[str],
    previous_actions: str | None = None,
) -> str:
    instruction = _task_instruction(obs)
    if style == "ui_venus":
        actions = native_model_action_kinds(style, allowed_kinds)
        action_lines = "\n".join(_ui_venus_action_signature(action) for action in actions)
        return _ui_venus_web_prompt(
            instruction=instruction,
            previous_actions=previous_actions or "",
            action_lines=action_lines,
        )
    del allowed_kinds, previous_actions
    return f"Task: {instruction}"


def native_six_action_task_prompt(
    style: str,
    obs: Observation,
    *,
    allowed_kinds: Iterable[str],
    previous_actions: str | None = None,
) -> str:
    """Render the model-family task shell used by the prior six-action runs."""

    actions = native_action_kinds(allowed_kinds)
    instruction = _task_instruction(obs)
    if style in {"uitars", "dart_gui"}:
        return _uitars_task_prompt(instruction, actions)
    if style in {"gui_owl", "evocua_s2"}:
        return (
            "Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous actions:\n"
            f"{previous_actions or 'None'}"
        )
    if style == "opencua_pyautogui":
        return (
            "# Task Instruction:\n"
            f"{instruction}\n\n"
            "Please generate the next move according to the screenshot, task "
            "instruction and previous steps (if provided)."
        )
    if style == "holo3_tool":
        return instruction
    if style == "ui_voyager":
        return f"The user query: {instruction}\n\n"
    if style == "mai_ui":
        return instruction
    raise ValueError(f"native six-action task prompt is unsupported for style {style!r}")


def holo_observation_content(
    obs: Observation,
    *,
    image_path: str | None,
    evicted: bool = False,
) -> list[dict[str, str]]:
    content: list[dict[str, str]] = [
        {"type": "text", "text": f"<observation>\nTask: {_task_instruction(obs)}\n"}
    ]
    if evicted:
        content.append({"type": "text", "text": "[screenshot evicted]"})
    elif image_path is not None:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": "\n</observation>"})
    return content


def native_history_action_summary(style: str, raw: str, action: Action) -> str:
    if style == "ui_venus":
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.DOTALL)
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", raw, re.DOTALL)
        if think_match and action_match:
            return (
                f"<think>{think_match.group(1).strip()}</think>"
                f"<action>{action_match.group(1).strip()}</action>"
            )
    if style == "ui_voyager":
        match = re.search(r"(?:^|\n)\s*Action:\s*(.*?)\s*(?=\n|$)", raw)
        if match and match.group(1).strip():
            return match.group(1).strip()
    if style in {"gui_owl", "evocua_s2"}:
        match = re.search(r"Action\s*:\s*(.*?)(?=<tool_call>|$)", raw, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    if style == "opencua_pyautogui":
        match = re.search(
            r"##\s*Action\s*:\s*(.*?)(?=\n##\s|\n```|$)",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return match.group(1).strip()
    return json.dumps(action.to_dict(), ensure_ascii=False, separators=(",", ":"))


def native_assistant_history_content(style: str, raw: str, action: Action) -> str:
    if style == "holo3_tool":
        return canonical_holo3_response(raw, action)
    if style in {"uitars", "dart_gui"}:
        return _restore_uitars_box_tokens(raw)
    return raw.strip()


def canonical_holo3_response(raw: str, action: Action) -> str:
    payload = _first_json_object(raw)
    if isinstance(payload, dict) and isinstance(payload.get("tool_call"), dict):
        normalized = {
            "thought": str(payload.get("thought") or "Execute the next grounded GUI action."),
            "tool_call": _action_to_holo_tool_call(action),
        }
        if "note" in payload:
            normalized["note"] = payload.get("note")
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    thought_match = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.DOTALL)
    thought = (
        thought_match.group(1).strip()
        if thought_match and thought_match.group(1).strip()
        else "Execute the next grounded GUI action."
    )
    return json.dumps(
        {"thought": thought, "tool_call": _action_to_holo_tool_call(action)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def holo_tool_name(action: Action) -> str:
    return str(action.kind)


def qwen25_native_coordinate_size(
    style: str,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    if style not in {"uitars", "dart_gui", "opencua_pyautogui", "ui_venus"}:
        return image_size
    width, height = image_size
    if style == "ui_venus":
        min_pixels = UI_VENUS_MIN_PIXELS
        max_pixels = UI_VENUS_MAX_PIXELS
    else:
        min_pixels = OPENCUA_MIN_PIXELS if style == "opencua_pyautogui" else UITARS_MIN_PIXELS
        max_pixels = QWEN25_MAX_PIXELS
    resized_height, resized_width = qwen25_smart_resize(
        height,
        width,
        factor=QWEN25_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resized_width, resized_height


def qwen25_smart_resize(
    height: int,
    width: int,
    *,
    factor: int = QWEN25_FACTOR,
    min_pixels: int = UITARS_MIN_PIXELS,
    max_pixels: int = QWEN25_MAX_PIXELS,
) -> tuple[int, int]:
    """Match the factor-28 smart resize used by Qwen2.5-VL OSWorld agents."""

    if min(height, width) <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    if max(height, width) / min(height, width) > QWEN25_MAX_RATIO:
        raise ValueError(f"image aspect ratio exceeds {QWEN25_MAX_RATIO}: {(width, height)}")

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = max(factor, math.ceil(height * beta / factor) * factor)
        resized_width = max(factor, math.ceil(width * beta / factor) * factor)
    return int(resized_height), int(resized_width)


def _task_instruction(obs: Observation) -> str:
    metadata = obs.metadata or {}
    task_type = next(
        (
            value
            for key in ("task_type", "benchmark", "task_id", "episode_id")
            if isinstance((value := metadata.get(key)), str) and value.strip()
        ),
        None,
    )
    return canonicalize_task_requirement(
        obs.instruction,
        task_type=task_type,
        target_label=metadata.get("target_label"),
    )


def _uitars_task_prompt(instruction: str, actions: tuple[str, ...]) -> str:
    action_space = "\n".join(_uitars_action_signature(kind) for kind in actions)
    return f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: a concise plan ending with the next target action
Action: exactly one action call
```

## Action Space
{action_space}

## User Instruction
{instruction}
""".strip()


def _uitars_action_signature(kind: str) -> str:
    signatures = {
        "move_to": (
            "move_to(start_box='<|box_start|>(x,y)<|box_end|>') "
            "# Move the cursor or view without pressing."
        ),
        "mouse_down": (
            "mouse_down() # Press and hold the left mouse button at the current cursor."
        ),
        "mouse_up": "mouse_up() # Release the left mouse button at the current cursor.",
        "left_click": (
            "left_click() # Perform a complete left click at the current cursor; "
            "no coordinates."
        ),
        "drag": (
            "drag(start_box='<|box_start|>(x1,y1)<|box_end|>', "
            "end_box='<|box_start|>(x2,y2)<|box_end|>')"
        ),
        "click": "click(start_box='<|box_start|>(x,y)<|box_end|>')",
    }
    return signatures.get(kind, f"{kind}()")


def _mai_ui_system_prompt(actions: tuple[str, ...]) -> str:
    action_space = "\n".join(_mai_ui_action_signature(kind) for kind in actions)
    return f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{{"name": "mobile_use", "arguments": <args-json-object>}}
</tool_call>
```

## Action Space

{action_space}

## Note
- Coordinates are integer screenshot-relative positions on a 0..999 grid with the origin at the top-left.
- Write a small plan and finally summarize your next action and target in one sentence in <thinking></thinking>.
- Return exactly one action from the Action Space and no text outside the required XML tags.
""".strip()


def _mai_ui_action_signature(kind: str) -> str:
    signatures = {
        "move_to": (
            '{"action": "move_to", "coordinate": [x, y]} '
            "# Move the pointer or view without pressing."
        ),
        "mouse_down": (
            '{"action": "mouse_down"} # Press and hold at the current pointer or reticle.'
        ),
        "mouse_up": (
            '{"action": "mouse_up"} # Release at the current pointer or reticle.'
        ),
        "left_click": (
            '{"action": "left_click"} '
            "# Complete a left click at the current pointer or reticle; no coordinates."
        ),
        "drag": (
            '{"action": "drag", "start_coordinate": [x1, y1], '
            '"end_coordinate": [x2, y2]} # Complete one press-drag-release gesture.'
        ),
        "click": '{"action": "click", "coordinate": [x, y]} # Click the specified point.',
    }
    return signatures.get(kind, f'{{"action": "{kind}"}}')


def _ui_voyager_system_prompt(actions: tuple[str, ...]) -> str:
    coordinate_schema = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 0, "maximum": 999},
    }
    action_descriptions = {
        "move_to": "`move_to`: Move the pointer or view to `coordinate` without pressing.",
        "mouse_down": "`mouse_down`: Press and hold at the current pointer or reticle.",
        "mouse_up": "`mouse_up`: Release at the current pointer or reticle.",
        "left_click": (
            "`left_click`: Complete a left click at the current pointer or reticle; "
            "no coordinates."
        ),
        "swipe": (
            "`swipe`: Perform one continuous pointer gesture from `coordinate` "
            "to `coordinate2`."
        ),
        "click": "`click`: Click the point in `coordinate`.",
    }
    description = (
        "Use a pointer to interact with the current GUI screenshot. Coordinates are integer "
        "screenshot-relative positions on a 0..999 grid with the origin at the top-left. "
        "The available actions are:\n"
        + "\n".join(f"* {action_descriptions[kind]}" for kind in actions)
        + "\nThe benchmark executes `swipe` with its complete press-drag-release implementation."
    )
    tool = {
        "type": "function",
        "function": {
            "name_for_human": "mobile_use",
            "name": "mobile_use",
            "description": description,
            "parameters": {
                "properties": {
                    "action": {
                        "description": "The single action to perform.",
                        "enum": list(actions),
                        "type": "string",
                    },
                    "coordinate": {
                        **coordinate_schema,
                        "description": "Required only by move_to, click, and swipe.",
                    },
                    "coordinate2": {
                        **coordinate_schema,
                        "description": "Required only by swipe.",
                    },
                },
                "required": ["action"],
                "type": "object",
                "additionalProperties": False,
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }
    return "\n".join(
        [
            "You are a helpful assistant.",
            "",
            "# Tools",
            "",
            "You may call one function to assist with the user query.",
            "",
            "You are provided with function signatures within <tools></tools> XML tags:",
            "<tools>",
            json.dumps(tool, ensure_ascii=False, separators=(",", ":")),
            "</tools>",
            "",
            "For the function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>",
            "",
            "# Response format",
            "",
            "Response format for every step:",
            "1. Thought: one concise sentence explaining the next move.",
            "2. Action: a short imperative describing what to do in the GUI.",
            "3. A single <tool_call>...</tool_call> block containing only the JSON.",
            "",
            "Output exactly in the order: Thought, Action, <tool_call>. Do not output anything else.",
        ]
    )


def _ui_venus_action_signature(action: str) -> str:
    signatures = {
        "Click": "- Click(box=(x1,y1))",
        "Drag": "- Drag(start=(x1,y1), end=(x2,y2))",
        "Scroll": "- Scroll(direction='down or up')",
        "Type": "- Type(content='')",
        "Launch": "- Launch(app='' or url='')",
        "Wait": "- Wait()",
        "Finished": "- Finished(content='')",
        "CallUser": "- CallUser(content='')",
        "LongPress": "- LongPress(box=(x1,y1))",
        "PressBack": "- PressBack()",
        "PressHome": "- PressHome()",
        "PressEnter": "- PressEnter()",
        "PressRecent": "- PressRecent()",
        "Hover": "- Hover(box=(x1,y1))",
        "DoubleClick": "- DoubleClick(box=(x1,y1))",
        "Hotkey": (
            "- Hotkey(keys=['ctrl', 'c']) # Split keys with comma and wrap each key in "
            "single quotes. Do not use more than 3 keys in one Hotkey action."
        ),
    }
    try:
        return signatures[action]
    except KeyError as exc:
        raise ValueError(f"unsupported UI-Venus native action: {action!r}") from exc


def _ui_venus_web_prompt(
    *,
    instruction: str,
    previous_actions: str,
    action_lines: str,
) -> str:
    return f"""**You are a GUI Agent.**
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

### Available Actions
You may execute one of the following functions:
{action_lines}

### User Task
{instruction}

### Previous Actions
{previous_actions}

### Output Format
<think> your thinking process </think>
<action> the next action </action>
<conclusion> the conclusion about the next action </conclusion>

### Instruction
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- For complex information-retrieval tasks, use `CallUser(content='...')` to reply only at the very end, after gathering all required info. Combine web evidence with your reasoning.
- To input text: first `Click(box=...)` on the textbox, then `Type(content='...')`. The system automatically presses `ENTER` afterward. If search filters are needed, click the search button after typing.
- Try to use simple language when searching.
- Distinguish textbox from button: never `Type` into a button. If no textbox is visible, try clicking the search icon first — the input field may appear afterward.
- Execute only one action per step.
- Strictly avoid repeating the same action when the webpage remains unchanged — you may have executed the wrong action. Continuous use of `Wait()` is also NOT allowed.
"""


def _tool_call_system_prompt(
    actions: tuple[str, ...],
    *,
    model: str,
    coordinate_click_enabled: bool,
) -> str:
    action_enum = list(actions)
    coordinate_max = 999 if model == "evocua" else 1000
    coordinate_frame = (
        "the EvoCUA 0..999 integer grid (1000 positions per axis)"
        if model == "evocua"
        else "integer [0,1000] screenshot-relative coordinates"
    )
    coordinate_schema = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 0, "maximum": coordinate_max},
    }
    coordinate_click_name = "click" if model == "gui_owl" else "left_click"
    click_description = (
        f"The model-native {coordinate_click_name} action uses exactly one "
        "coordinate=[x,y] to click an arbitrary point. "
        + (
            "left_click without coordinates clicks at the current cursor. "
            if model == "gui_owl"
            else "Omit coordinate only when clicking at the current cursor. "
        )
        if coordinate_click_enabled
        else "left_click performs a coordinate-free click at the current cursor; "
        "arbitrary-point clicking is disabled. "
    )
    drag_duration_description = (
        " EvoCUA may include an optional non-negative duration in seconds; its official "
        "executor defaults to 0.5."
        if model == "evocua"
        else ""
    )
    description = (
        "Use a mouse to interact with a desktop GUI and take screenshots. Coordinates use "
        f"{coordinate_frame} with origin at the top-left. "
        "Use the model-native mouse_move action with exactly one coordinate=[x,y] to move "
        "the cursor or view without pressing. Use the model-native left_click_drag action "
        "when the cursor is already at the drag start; it takes one destination coordinate "
        "and performs a complete drag from the current cursor."
        f"{drag_duration_description} The benchmark also requires compatibility actions: "
        "mouse_down presses and holds at the current cursor; mouse_up releases there; "
        f"{click_description}"
        "drag performs one complete press-drag-release gesture and requires exactly "
        "points=[[x1,y1],[x2,y2]]. mouse_down and mouse_up take no coordinate fields. "
        "Do not mix coordinate encodings."
    )
    if model == "gui_owl":
        description += " Keep the action description brief and visually grounded."
    else:
        description += (
            " If a previous action did not work, adjust the next action using the new screenshot."
        )
    properties: dict[str, Any] = {
        "action": {"type": "string", "enum": action_enum},
        "coordinate": {
            **coordinate_schema,
            "description": (
                "One native [x,y] point for mouse_move, left_click_drag, or "
                f"{coordinate_click_name}."
                if coordinate_click_enabled
                else "One native [x,y] point for mouse_move or left_click_drag."
            ),
        },
        "points": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": coordinate_schema,
            "description": (
                "Exactly [[x1,y1],[x2,y2]] for the benchmark compatibility drag action."
            ),
        },
        "x": {"type": "integer", "minimum": 0, "maximum": coordinate_max},
        "y": {"type": "integer", "minimum": 0, "maximum": coordinate_max},
    }
    if model == "evocua":
        properties["duration"] = {
            "type": "number",
            "minimum": 0,
            "description": (
                "Optional left_click_drag duration in seconds; defaults to 0.5."
            ),
        }
    tool = {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["action"],
                "additionalProperties": False,
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }
    return "\n".join(
        [
            "# Tools",
            "",
            "You normally call one function to perform the next GUI action. For a native "
            "drag only, you may call exactly two functions in order: mouse_move(start), "
            "then left_click_drag(destination).",
            "",
            "You are provided with function signatures within <tools></tools> XML tags:",
            "<tools>",
            json.dumps(tool, ensure_ascii=False, separators=(",", ":")),
            "</tools>",
            "",
            "For the function call, return a JSON object within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name":"computer_use","arguments":{"action":"mouse_move","coordinate":[500,500]}}',
            "</tool_call>",
            "",
            "# Response format",
            "1) Action: one short imperative sentence.",
            "2) Exactly one <tool_call> block, except for the exact two-block native drag "
            "macro: mouse_move followed by left_click_drag.",
            "Do not output anything else.",
        ]
    )


def _holo3_structured_system_prompt(actions: tuple[str, ...]) -> str:
    schema = holo3_structured_output_schema(actions)
    return (
        "You are Holo, a GUI agent operating in a multi-step screenshot-and-action loop. "
        "Reason before each step. Return exactly one JSON object in assistant content; "
        "do not use XML tool calls. Coordinates use integer [0,1000] bins relative to "
        "the screenshot with top-left origin. move_to: move the cursor or view without "
        "pressing; mouse_down and mouse_up act at the current cursor; left_click is a "
        "coordinate-free complete click; drag is a complete press-drag-release macro with "
        "two points. Use note for durable information that future turns must remember.\n\n"
        "<output_format>\n```json\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        + "\n```\n</output_format>"
    )


def holo3_structured_output_schema(actions: Iterable[str]) -> dict[str, Any]:
    """Return the exact Holo3 assistant-content schema used by prompt and decoder.

    ``anyOf`` is deliberately used for the nullable note instead of the shorthand
    ``type: [string, null]``.  They are equivalent JSON Schema, while the Outlines
    version available in the evaluation environment only compiles the former.
    """

    variants: list[dict[str, Any]] = []
    for kind in actions:
        properties: dict[str, Any] = {"tool_name": {"const": kind}}
        required = ["tool_name"]
        if kind == "move_to":
            properties.update(
                {
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                }
            )
            required.extend(["x", "y"])
        elif kind == "drag":
            properties["points"] = {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                },
            }
            required.append("points")
        elif kind == "click":
            properties.update(
                {
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                }
            )
            required.extend(["x", "y"])
        variants.append(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        )
    schema = {
        "type": "object",
        "properties": {
            "note": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Durable task-relevant memory from the current observation.",
            },
            "thought": {"type": "string", "description": "One-line plan for the next action."},
            "tool_call": {"oneOf": variants},
        },
        "required": ["thought", "tool_call"],
        "additionalProperties": False,
    }
    return schema


def _opencua_l2_system_prompt(actions: tuple[str, ...]) -> str:
    action_lines = "\n".join(_opencua_action_line(kind) for kind in actions)
    return f"""You are a GUI agent. You are given a task, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. Do not terminate the task unless you are sure it is finished. If the task cannot be completed exactly as instructed, report failure instead of claiming partial progress as success.

For each step, provide your response in this format:
# Step {{step number}}:
## Thought:
{{thought}}
## Action:
{{action}}
## Code:
```python
{{code}}
```

For the Thought section, include all of the following:
- Reflection on the previous action when one exists:
  - Consider whether the previous action was correct and what outcome the new screenshot shows.
  - If it was correct, describe the state change and why it advances the task.
  - If it was incorrect, diagnose what went wrong and choose a logical recovery.
- Step-by-step progress assessment:
  - Use the history screenshots, former actions, and current screenshot.
  - Identify completed parts and how they contribute to the overall goal.
  - Make a plan for completing the remaining work.
- Next-action prediction:
  - Propose the most likely next action and explain why it is the best choice.
- Use first-person perspective in the reasoning.

For the Action section, write one clear, concise, actionable imperative sentence. Describe the target by its visible name, appearance, and relative location without giving coordinates.

For the Code section, output exactly one of these canonical actions as PyAutoGUI code:
{action_lines}

Coordinate convention for OpenCUA-72B:
- moveTo, dragTo, and coordinate-bearing click use absolute positions on the Qwen2.5-VL factor-28 smart-resized image.
- dragRel dx and dy use relative pixel deltas in that same smart-resized image coordinate frame.
- left_click, mouse_down, and mouse_up act at the current cursor and therefore take no coordinates.

Return one step only. Do not emit screenshot calls, element-locator calls, variables shared across turns, or more than one canonical action. A drag may start at the current cursor with dragTo/dragRel, or use moveTo followed by dragTo/dragRel; these forms each represent one complete drag macro.
""".strip()


def _opencua_action_line(kind: str) -> str:
    lines = {
        "move_to": (
            "- move_to: pyautogui.moveTo(x=<absolute_x>, y=<absolute_y>) "
            "# Move the cursor or view without pressing."
        ),
        "mouse_down": "- mouse_down: pyautogui.mouseDown()",
        "mouse_up": "- mouse_up: pyautogui.mouseUp()",
        "left_click": "- left_click: pyautogui.click()",
        "drag": (
            "- drag: pyautogui.dragTo(x=<absolute_x2>, y=<absolute_y2>) or "
            "pyautogui.dragRel(<dx>, <dy>) from the current cursor; optionally precede "
            "either form with pyautogui.moveTo(x=<absolute_x1>, y=<absolute_y1>)"
        ),
        "click": "- click: pyautogui.click(x=<absolute_x>, y=<absolute_y>)",
    }
    return lines.get(kind, f"- {kind}: pyautogui.{kind}()")


def _first_json_object(raw: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _restore_uitars_box_tokens(raw: str) -> str:
    text = raw.strip()
    action_header = re.search(r"(?m)^Action:", text)
    if action_header is None:
        return text
    prefix = text[:action_header.start()]
    action_text = text[action_header.end():]
    pattern = re.compile(
        r"(start_box|end_box)\s*=\s*'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*\)'"
    )

    def replacement(match: re.Match[str]) -> str:
        return (
            f"{match.group(1)}='<|box_start|>({match.group(2)},{match.group(3)})"
            "<|box_end|>'"
        )

    return prefix + "Action:" + pattern.sub(replacement, action_text)


def _action_to_holo_tool_call(action: Action) -> dict[str, Any]:
    if isinstance(action, PrimitiveAction):
        if action.kind == "move_to":
            return {
                "tool_name": "move_to",
                "x": int(round(float(action.x or 0))),
                "y": int(round(float(action.y or 0))),
            }
        return {"tool_name": action.kind}
    if isinstance(action, AtomicAction):
        if action.kind == "drag":
            return {
                "tool_name": "drag",
                "points": [[int(round(x)), int(round(y))] for x, y in action.points[:2]],
            }
        if action.kind == "click" and action.points:
            x, y = action.points[0]
            return {"tool_name": "click", "x": int(round(x)), "y": int(round(y))}
        return {"tool_name": action.kind}
    return {"tool_name": str(action.kind)}
