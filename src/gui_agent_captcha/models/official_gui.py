"""Official-first adapters for the six local GUI baselines.

The existing local backend owns model loading and generation. This adapter owns
the upstream prompts, message layout and native calls, independently of the
benchmark's six internal mouse primitives.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..core import Observation, StepResult
from ..train.qwen3_vl_sft import ImageCoordinateTransform
from .local_baseline import LocalBaselineVisionBackend, _extract_thought_text
from .native_baseline_protocols import _restore_uitars_box_tokens, qwen25_smart_resize
from .official_gui_actions import (
    BUTTON_EXTENSION_STYLES,
    EXECUTOR_ALIASES,
    HOLO_ACTIONS,
    OFFICIAL_TOOL_ACTIONS,
    PY_AUTO_GUI_PARAMETERS,
    TOOL_ACTIONS,
    UITARS_ACTIONS,
    UITARS_OFFICIAL_ACTIONS,
    OfficialGuiAction,
    OfficialGuiProtocolError,
    current_position_click_action,
    missing_mouse_actions,
    parse_official_response,
    without_thinking,
)
from .official_gui_button_prompts import guidance as button_prompt_guidance
from .official_gui_prompts import PROMPTS
from .ui_venus2_official import SCHEMAS as VENUS2_SCHEMAS

OFFICIAL_GUI_CONTRACT = "official_first_gui_v2"
BUTTON_EXTENSION_CONTRACT = "official_first_gui_button_extension_v3"
GUI_OWL_BUTTON_EXTENSION_CONTRACT = "official_first_gui_button_extension_v4_owl_current_click"
BUTTON_DESCRIPTION = (
    "Benchmark extension: mouse_down presses and holds the left mouse button at "
    "the current cursor or fixed reticle; mouse_up releases it there. Neither "
    "takes coordinates or moves the cursor. The held button state persists across "
    "decisions, including cursor movements, until mouse_up. When using either new "
    "button action, return it as the only call in that response; a fresh screenshot "
    "is returned before the next decision. "
    "A click presses and releases immediately, and a complete drag also releases; "
    "neither keeps the button held for a later decision."
)


def extend_button_prompt(prompt: str, style: str) -> str:
    """Extend the existing computer_use declaration inside the text prompt."""

    match = re.search(r"<tools>\s*(\{.*?)\s*</tools>", prompt, re.S)
    if not match:
        raise ValueError("official computer_use prompt has no tools block")
    # Some upstream fixtures contain literal newlines inside JSON descriptions.
    tool = json.loads(match[1], strict=False)
    action = tool["function"]["parameters"]["properties"]["action"]
    action["enum"].extend(missing_mouse_actions(style))
    action["description"] += "\n" + BUTTON_DESCRIPTION
    click_action = current_position_click_action(style)
    if style == "gui_owl":
        action["description"] += (
            "\n* left_click_current: Press and release the left mouse button once at the current cursor without moving. Takes no arguments."
            "\n* left_click requires coordinate=[x,y] and moves there before clicking."
        )
        tool["function"]["parameters"]["properties"]["coordinate"]["description"] = (
            "(x, y): The coordinates to move the mouse to. Required for mouse_move, left_click_drag and left_click."
            " Omit for left_click_current, mouse_down and mouse_up."
        )
    tool["function"]["parameters"]["properties"]["button"] = {
        "type": "string", "enum": ["left"],
        "description": "Optional for mouse_down/mouse_up only; defaults to left. No coordinates.",
    }
    extended = prompt[:match.start(1)] + json.dumps(tool, ensure_ascii=False) + prompt[match.end(1):]
    click_instruction = (
        "Use left_click_current without arguments for a click at the current cursor. left_click requires coordinate=[x,y]. "
        if style == "gui_owl" else "Omit coordinate for a click at the current cursor. "
    )
    return extended + (
        "\n# Benchmark button output examples\n"
        'To press and hold without moving: <tool_call>{"name":"computer_use",'
        '"arguments":{"action":"mouse_down"}}</tool_call>\n'
        'To release without moving: <tool_call>{"name":"computer_use",'
        '"arguments":{"action":"mouse_up"}}</tool_call>\n'
        'To click in place: <tool_call>{"name":"computer_use",'
        f'"arguments":{{"action":"{click_action}"}}}}</tool_call>\n'
        f"{click_instruction}"
        "Use the existing response format for all other actions.\n"
    )
HISTORY_STRATEGIES = {
    "ui_venus2_computer": "official_venus2_current_plus2_images_all_accepted_responses",
    "uitars": "official_uitars_latest5_images_all_responses",
    "dart_gui": "official_dart_latest5_images_all_responses",
    "gui_owl": "official_owl_latest5_images_older_action_summary",
    "evocua_s2": "official_evocua_current_plus4_history_older_action_summary",
    "opencua_pyautogui": "official_opencua_l2_image3_action_history10",
    "holo3_tool": "official_holo31_function_calls_latest3_images",
}
GENERATION_LIMITS = {
    "ui_venus2_computer": 8192,
    "uitars": 1000,
    "dart_gui": 500,
    "gui_owl": 32768,
    "evocua_s2": 32768,
    "opencua_pyautogui": 2048,
    "holo3_tool": 8192,
}


def model_action_names(style: str, *, button_extension: bool = False) -> tuple[str, ...]:
    if button_extension:
        if style not in BUTTON_EXTENSION_STYLES:
            raise ValueError(f"button extension is not defined for {style}")
        return (*model_action_names(style), *missing_mouse_actions(style))
    if style == "ui_venus2_computer":
        return (*VENUS2_SCHEMAS, "Sequence")
    if style in OFFICIAL_TOOL_ACTIONS:
        return OFFICIAL_TOOL_ACTIONS[style]
    if style in {"uitars", "dart_gui"}:
        return UITARS_ACTIONS
    if style == "holo3_tool":
        return HOLO_ACTIONS
    if style == "opencua_pyautogui":
        return tuple(f"pyautogui.{name}" for name in PY_AUTO_GUI_PARAMETERS) + (
            "pyautogui.hotkey",
            "computer.terminate",
            "computer.triple_click",
            "computer.wait",
        )
    raise ValueError(f"unsupported official GUI style: {style}")


def official_action_names(style: str) -> tuple[str, ...]:
    """Return upstream action names before benchmark extensions and aliases."""

    if style in {"uitars", "dart_gui"}:
        return UITARS_OFFICIAL_ACTIONS
    if style == "holo3_tool":
        # Holo3.1 has no universal published desktop enum; its local function
        # declarations are recorded separately as harness-defined tools.
        return ()
    return model_action_names(style)


def advertised_action_names(style: str) -> tuple[str, ...]:
    """Return action names visible to the model in the final prompt."""

    return model_action_names(style)


def protocol_metadata(style: str, *, button_extension: bool = False,
                      button_prompt_reminder: bool = False) -> dict[str, Any]:
    if button_prompt_reminder and not button_extension:
        raise ValueError("button prompt reminder requires the button extension")
    if button_extension:
        names = list(model_action_names(style, button_extension=True))
        metadata = protocol_metadata(style)
        contract = GUI_OWL_BUTTON_EXTENSION_CONTRACT if style == "gui_owl" else BUTTON_EXTENSION_CONTRACT
        metadata.update(
            contract=contract,
            advertised_action_names=names,
            model_action_names=names,
            added_actions=[*metadata["added_actions"], *missing_mouse_actions(style)],
            current_position_click=f"{current_position_click_action(style)} without coordinates",
            button_extension_description=BUTTON_DESCRIPTION,
            optional_button_argument=("left" if style in TOOL_ACTIONS else None),
            environment_contract="native_mouse_mapping_stateful_button_extension_v1",
        )
        if style == "gui_owl":
            metadata["native_left_click_requires_coordinate"] = True
        if button_prompt_reminder:
            metadata.update(
                contract=contract + "_reminder",
                button_prompt_variant="reminder",
            )
        return metadata
    return {
        "contract": OFFICIAL_GUI_CONTRACT,
        "source": (
            {"url": "https://github.com/inclusionAI/UI-Venus/blob/"
             "de1f35d17e29cb53ccabe3489b0421c6eca3d26e/models/computer/computer_example.py",
             "retrieved_at": "2026-09-10", "modifications": []}
            if style == "ui_venus2_computer" else PROMPTS["sources"][style]
        ),
        **({"temperature": 1.0, "top_p": 0.7, "enable_thinking": True,
            "parse_retries": 0, "transport": "local_vllm_openai_api",
            "generation_source": "model card agentic temperature; Computer example top_p"}
           if style == "ui_venus2_computer" else {}),
        **({"computer_use_parser_contract": "official_aliases_ordered_calls_v2"}
           if style in TOOL_ACTIONS else {}),
        **({"holo_parser_contract": "ordered_drag_points_v2"}
           if style == "holo3_tool" else {}),
        "history_strategy": HISTORY_STRATEGIES[style],
        "official_action_names": list(official_action_names(style)),
        "advertised_action_names": list(advertised_action_names(style)),
        "executor_aliases": list(EXECUTOR_ALIASES.get(style, ())),
        # Kept as the model-facing compatibility field used by run manifests.
        "model_action_names": list(advertised_action_names(style)),
        "added_actions": ["move_to"] if style in {"uitars", "dart_gui"} else [],
        "harness_defined_tools": list(HOLO_ACTIONS) if style == "holo3_tool" else [],
        "max_new_tokens": GENERATION_LIMITS[style],
        "image_processing": (
            "official_evocua_factor32_resize_then_checkpoint_processor"
            if style == "evocua_s2"
            else "checkpoint_processor_without_benchmark_pre_resize"
        ),
        "environment_contract": "native_mouse_mapping_nonmouse_no_effect_v1",
        "scoring_policy": "first_real_release_attempt_success_only_no_correction",
    }


def holo31_tools() -> list[dict[str, Any]]:
    coordinate = {"type": "integer", "minimum": 0, "maximum": 1000}
    definitions = {
        "click": (
            "Click at (x, y) coordinates",
            {
                "element": {
                    "type": "string",
                    "description": "Detailed description of the target UI element",
                },
                "x": coordinate,
                "y": coordinate,
            },
        ),
        "write": (
            "Type text into the currently focused element without clicking first",
            {
                "content": {"type": "string"},
                "press_enter": {"type": "boolean", "default": False},
            },
        ),
        "answer": ("Provide a final answer", {"content": {"type": "string"}}),
        "move_to": (
            "Move the cursor or view without pressing; coordinates are in [0,1000]",
            {
                "x": coordinate,
                "y": coordinate,
            },
        ),
        "mouse_down": ("Press and hold the left mouse button at the current cursor", {}),
        "mouse_up": ("Release the left mouse button at the current cursor", {}),
        "left_click": ("Click the left button at the current cursor without moving", {}),
        "drag": (
            "Press at the first point, drag through the points and release at the last",
            {
                "points": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": coordinate,
                    },
                },
            },
        ),
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [key for key in properties if key != "press_enter"],
                },
            },
        }
        for name, (description, properties) in definitions.items()
    ]


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _image(path: Path) -> dict[str, str]:
    return {"type": "image", "image": str(path)}


def _summary(raw: str) -> str:
    match = re.search(
        r"(?:^|\n)(?:##\s*)?Action:\s*(.*?)(?=\n##|\n```|<tool_call>|$)",
        without_thinking(raw),
        re.S,
    )
    return match[1].strip() if match else without_thinking(raw)


@dataclass
class OfficialGuiBackend(LocalBaselineVisionBackend):
    prompt_contract: str = OFFICIAL_GUI_CONTRACT
    image_max_pixels: int | None = None
    max_new_tokens: int = 0
    button_extension: bool = False
    button_prompt_reminder: bool = False

    def __post_init__(self) -> None:
        if self.button_prompt_reminder and not self.button_extension:
            raise ValueError("button prompt reminder requires the button extension")
        if self.button_extension:
            model_action_names(self.prompt_style, button_extension=True)
            self.prompt_contract = protocol_metadata(
                self.prompt_style, button_extension=True,
                button_prompt_reminder=self.button_prompt_reminder,
            )["contract"]
        if self.max_new_tokens == 0:
            self.max_new_tokens = GENERATION_LIMITS[self.prompt_style]

    def _load_image_with_transform(
        self, image_path: Path
    ) -> tuple[Image.Image, ImageCoordinateTransform]:
        if self.prompt_style != "evocua_s2":
            return super()._load_image_with_transform(image_path)
        # mm_agents/evocua/utils.py::process_image pre-resizes using PIL before
        # the checkpoint processor. Preserve that resampling operation too.
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original = image.size
        height, width = qwen25_smart_resize(
            original[1],
            original[0],
            factor=32,
            min_pixels=4 * 28 * 28,
            max_pixels=16 * 16 * 4 * 12800,
        )
        image = image.resize((width, height))
        return image, ImageCoordinateTransform(original, image.size)

    def _native_system_prompt(self, allowed_kinds: Iterable[str]) -> str:
        if self.prompt_style in {"uitars", "dart_gui"}:
            return "You are a helpful assistant."
        if self.prompt_style == "holo3_tool":
            # The checkpoint chat template supplies the function-call grammar.
            return "You are Holo, a GUI agent operating a computer through screenshots and tools."
        prompt = PROMPTS[self.prompt_style]
        if self.button_extension and self.prompt_style in {"gui_owl", "evocua_s2"}:
            prompt = extend_button_prompt(prompt, self.prompt_style)
        return prompt

    def _native_task_prompt(
        self, obs: Observation, *, allowed_kinds: Iterable[str], previous_actions: str | None = None
    ) -> str:
        prompt = self._native_task_prompt_base(
            obs, allowed_kinds=allowed_kinds, previous_actions=previous_actions
        )
        if self.button_prompt_reminder:
            prompt += button_prompt_guidance(self.prompt_style, "reminder")
        return prompt

    def _native_task_prompt_base(
        self, obs: Observation, *, allowed_kinds: Iterable[str], previous_actions: str | None = None
    ) -> str:
        # Keep the task instruction supplied by the environment verbatim.
        instruction = obs.instruction
        if self.prompt_style in {"uitars", "dart_gui"}:
            prompt = PROMPTS[self.prompt_style]
            addition = (
                "move_to(start_box='<|box_start|>(x,y)<|box_end|>') "
                "# Benchmark extension: move the cursor or view without pressing.\n"
            )
            if self.button_extension:
                addition += (
                    "mouse_down() # Press and hold the left mouse button at the current cursor.\n"
                    "mouse_up() # Release the left mouse button at the current cursor.\n"
                    "left_click() # Press and release once at the current cursor, without moving.\n"
                    "# " + BUTTON_DESCRIPTION + "\n"
                    "# Button output format: Action: mouse_down() to press and hold; "
                    "Action: mouse_up() to release; Action: left_click() for a click in place. No arguments.\n"
                )
            prompt = prompt.replace("## Action Space\n", "## Action Space\n\n" + addition, 1)
            return prompt.format(instruction=instruction, language="English")
        if self.prompt_style in {"gui_owl", "evocua_s2"}:
            previous = (None if previous_actions == "None" else previous_actions) or (
                "No previous action." if self.prompt_style == "gui_owl" else "None"
            )
            wrapper = "\n" if self.prompt_style == "evocua_s2" else ""
            return wrapper + (
                "Please generate the next move according to the UI screenshot, instruction "
                "and previous actions.\n\n"
                f"Instruction: {instruction}\n\nPrevious actions:\n{previous}"
            )
        return (
            f"\n# Task Instruction:\n{instruction}\n\nPlease generate the next move according "
            "to the screenshot, task instruction and previous steps (if provided).\n"
        )

    def _build_prompt_messages(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
        condition: str | None,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        style = self.prompt_style
        system = self._native_system_prompt(allowed_kinds)
        current = Path(obs.screenshot_path)
        if style in {"uitars", "dart_gui"}:
            return self._build_uitars_family_messages(
                system,
                self._native_task_prompt(obs, allowed_kinds=()),
                current,
                historical_image_max=4,
            )
        if style in {"gui_owl", "evocua_s2"}:
            messages, images = self._build_tool_call_family_messages(
                obs,
                system,
                current,
                allowed_kinds=(),
                historical_image_max=4,
                image_first=style == "evocua_s2",
            )
            # The official Owl cookbook uses Step1, whereas EvoCUA uses Step 1.
            if style == "gui_owl":
                for part in messages[1]["content"]:
                    if part.get("type") == "text":
                        before, previous = part["text"].rsplit("Previous actions:\n", 1)
                        previous = re.sub(r"(?m)^Step (\d+):", r"Step\1:", previous)
                        part["text"] = before + "Previous actions:\n" + previous
            return messages, images
        if style == "opencua_pyautogui":
            count = self._history_record_count()
            messages = [{"role": "system", "content": system}]
            images = []
            for index in range(max(0, count - 10), count):
                if index >= count - 2:
                    path = self._history_image_paths[index]
                    messages.append({"role": "user", "content": [_image(path)]})
                    images.append(path)
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"# Step {index + 1}:\n## Action:\n{self._history_action_summaries[index]}\n"
                        ),
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _image(current),
                        _text(self._native_task_prompt(obs, allowed_kinds=())),
                    ],
                }
            )
            return messages, images + [current]
        if style != "holo3_tool":
            raise ValueError(f"unsupported official GUI style: {style}")
        count = self._history_record_count()
        messages = [{"role": "system", "content": system}]
        images = []
        for index in range(count + 1):
            path = current if index == count else self._history_image_paths[index]
            content = [_text("<observation>\n")]
            if index == 0:
                content.append(_text(obs.instruction))
            if index >= count - 2:
                content.append(_image(path))
                images.append(path)
            else:
                content.append(_text("[screenshot evicted]"))
            content.append(_text("\n</observation>"))
            messages.append({"role": "user", "content": content})
            if index < count:
                action = self._history_actions[index]
                calls = [
                    {
                        "id": f"call_{index}_{n}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for n, call in enumerate(action.calls)
                ]
                body = without_thinking(self._history_responses[index])
                durable = body.split("<tool_call>", 1)[0].strip()
                messages.append({"role": "assistant", "content": durable, "tool_calls": calls})
                outcomes = (
                    history[index].info.get("native_call_results", [])
                    if index < len(history)
                    else []
                )
                for n, call in enumerate(calls):
                    outcome = outcomes[n] if n < len(outcomes) else {"status": "not_executed"}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(outcome, ensure_ascii=False),
                        }
                    )
        return messages, images

    def _build_prompt_text(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
        condition: str | None,
    ) -> tuple[str, list[Path], list[dict[str, Any]]]:
        messages, images = self._build_prompt_messages(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
            condition=condition,
        )
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.prompt_style == "holo3_tool":
            kwargs.update(tools=holo31_tools(), enable_thinking=True)
        elif self.prompt_style in {"gui_owl", "evocua_s2"}:
            kwargs["enable_thinking"] = True
        return self.processor.apply_chat_template(messages, **kwargs), images, messages

    def _generation_kwargs(self) -> dict[str, Any]:
        settings = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        if self.prompt_style == "holo3_tool":
            settings.update(do_sample=True, temperature=0.8)
        return settings

    def _generate_holo3_structured(self, batch: Any, *, allowed_kinds: Iterable[str]) -> Any:
        # Parent dispatches Holo here; 3.1 generates its own native XML grammar.
        return self.model.generate(**batch, **self._generation_kwargs())

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> OfficialGuiAction:
        self._ensure_loaded()
        self._reset_history_if_needed(history)
        self._call_index += 1
        self.last_prediction_index = self._call_index
        self.last_native_image_size_px = None
        self.last_raw_prediction = None
        self.last_think_text = None
        self.last_prompt_messages = None
        self.last_prompt_text = None
        self.last_prompt_image_paths = ()
        raw, action, error = "", None, None
        try:
            raw, _transform = self._generate_raw(
                obs,
                history,
                allowed_kinds=(),
                budget=budget,
                condition=condition,
            )
            self.last_raw_prediction = raw
            self.last_think_text = _extract_thought_text(raw)
            if self.prompt_style in {"uitars", "dart_gui", "opencua_pyautogui"}:
                if self.last_native_image_size_px is None:
                    raise ValueError(
                        "processor image_grid_thw required for native pixel coordinates"
                    )
                size = self.last_native_image_size_px
            else:
                size = (999, 999) if self.prompt_style in {
                    "evocua_s2", "ui_venus2_computer"
                } else (1000, 1000)
            try:
                action = parse_official_response(
                    raw, self.prompt_style, coordinate_size=size,
                    button_extension=self.button_extension,
                )
            except (ValueError, SyntaxError, TypeError) as exc:
                raise OfficialGuiProtocolError(str(exc)) from exc
            self._history_image_paths.append(Path(obs.screenshot_path))
            self._history_responses.append(raw)
            self._history_assistant_contents.append(
                _restore_uitars_box_tokens(raw)
                if self.prompt_style in {"uitars", "dart_gui"}
                else raw
            )
            self._history_action_summaries.append(_summary(raw))
            self._history_actions.append(action)
            self._history_observations.append(obs)
            return action
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._write_call_log(
                obs=obs,
                history=history,
                allowed_kinds=model_action_names(
                    self.prompt_style, button_extension=self.button_extension
                ),
                raw_prediction=raw,
                parsed_action=action,
                parse_error=error,
                budget=budget,
                condition=condition,
            )
            if self.call_log_dir is not None:
                path = self.call_log_dir / f"call_{self._call_index:06d}.json"
                payload = json.loads(path.read_text())
                payload["protocol"] = protocol_metadata(
                    self.prompt_style, button_extension=self.button_extension,
                    button_prompt_reminder=self.button_prompt_reminder,
                )
                if self.prompt_style == "holo3_tool":
                    payload["tools"] = holo31_tools()
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
