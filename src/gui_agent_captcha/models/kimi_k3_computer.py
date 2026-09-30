from __future__ import annotations

import base64
import copy
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..actions import Action
from ..core import Observation, StepResult
from ..protocol_tracks import canonicalize_task_requirement
from .openai_responses_six_action import (
    OPENAI_RESPONSES_SIX_ACTION_KINDS,
    six_action_arguments_to_action,
)

JsonRequester = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]

KIMI_K3_CANONICAL_ACTION_KINDS = OPENAI_RESPONSES_SIX_ACTION_KINDS
KIMI_K3_MODEL_TOOL_NAMES = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "click_current_position",
    "click",
    "drag",
)
KIMI_K3_PROMPT_VERSION = "kimi_k3_kimicu_native_pixel_image_only_six_action_v2"
KIMI_K3_COMPUTER_USE_CONTRACT = "benchmark_custom_kimicu_native_screenshot_pixels"
KIMI_K3_PROTOCOL_TRACK = (
    "kimi_k3_chat_preserved_thinking_kimicu_native_pixel_six_action_v3"
)
KIMI_K3_HISTORY_STRATEGY = "full_messages_preserved_thinking_all_screenshots"
KIMI_K3_NATIVE_COORDINATE_FORMAT = "kimicu_screenshot_pixels"
KIMI_K3_DEFAULT_SCREEN_SIZE_PX = (1280, 720)
KIMI_K3_SYSTEM_PROMPT = "\n\n".join(
    (
        "You are controlling a computer from screenshots. This benchmark exposes an "
        "image-only six-action subset whose tool names and coordinate arguments are "
        "aligned with Kimi Computer Use where equivalents exist. Inspect the latest "
        "screenshot and call exactly one provided tool. If a target is uncertain, choose "
        "the most likely location.",
        "After the harness executes your tool call, it will send a new screenshot. Treat "
        "that newest screenshot as authoritative and verify the result before choosing "
        "the next action. UI changes can make earlier coordinates stale. Accessibility "
        "indexes, get_app_state, typing, scrolling, key presses, and other Kimi Computer "
        "Use tools are unavailable in this benchmark.",
        "Coordinates use Kimi Computer Use screenshot pixels. Each screenshot message "
        "states the current image width and height. Use integer pixel coordinates from "
        "that exact image: x=0 is the left edge and y=0 is the top edge. Do not normalize "
        "coordinates to 0-1000 or 0-1.",
        "Available tools: move_to(x, y) moves without pressing; mouse_down() presses and "
        "holds at the current cursor; mouse_up() releases at the current cursor; "
        "click_current_position() clicks at the current cursor or reticle; click(x, y) "
        "clicks a screenshot-relative point; drag(from_x, from_y, to_x, to_y) performs "
        "one continuous press-drag-release gesture. click and drag follow Kimi Computer "
        "Use naming; the other tools preserve benchmark compatibility.",
        "Return only the next action through exactly one function call. Do not return "
        "prose, XML, a JSON wrapper, multiple tool calls, or an unavailable action.",
    )
)
DEFAULT_MODEL = "kimi-k3"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "max"

_DATA_URL_RE = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+")


class KimiK3ApiError(RuntimeError):
    """The Kimi API request failed before returning a response."""


class KimiK3OutputError(ValueError):
    """Kimi returned output outside the six-action benchmark contract."""


@dataclass(frozen=True)
class KimiK3ComputerConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    tool_choice: str = "auto"
    unauthenticated: bool = False

    def __post_init__(self) -> None:
        if self.unauthenticated and self.api_key:
            raise ValueError("Unauthenticated Kimi requests must not include an API key")
        if not self.unauthenticated and not self.api_key.strip():
            raise ValueError("Kimi API key must not be empty")
        if not self.model.strip():
            raise ValueError("Kimi model must not be empty")
        if not self.base_url.strip():
            raise ValueError("Kimi base_url must not be empty")
        if self.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("Kimi reasoning_effort must be low, high, or max")
        if self.tool_choice not in {"auto", "required"}:
            raise ValueError("Kimi tool_choice must be auto or required")


@dataclass(frozen=True)
class _ParsedToolCall:
    assistant_message: dict[str, Any]
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    action: Action


def _screen_size_px(size_px: tuple[int, int]) -> tuple[int, int]:
    width, height = (int(value) for value in size_px)
    if width <= 0 or height <= 0:
        raise KimiK3OutputError(f"invalid screenshot size: {size_px!r}")
    return width, height


def _coordinate_schema(axis: str, extent: int, description: str) -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 0,
        "maximum": extent - 1,
        "description": (
            f"{description} Use the original screenshot pixel {axis} coordinate "
            f"in [0, {extent - 1}]."
        ),
    }


def _kimi_function_tool(
    name: str,
    description: str,
    *,
    properties: Mapping[str, Any] | None = None,
    required: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties or {}),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def kimi_k3_six_action_tools(
    screenshot_size_px: tuple[int, int] = KIMI_K3_DEFAULT_SCREEN_SIZE_PX,
) -> list[dict[str, Any]]:
    """Return six separately named tools following Kimi's function-call shape."""

    width, height = _screen_size_px(screenshot_size_px)
    x = _coordinate_schema("x", width, "Screenshot horizontal coordinate.")
    y = _coordinate_schema("y", height, "Screenshot vertical coordinate.")
    return [
        _kimi_function_tool(
            "move_to",
            "Move the cursor to a screenshot-relative coordinate without pressing.",
            properties={"x": x, "y": y},
            required=("x", "y"),
        ),
        _kimi_function_tool(
            "mouse_down",
            "Press and hold the primary mouse button at the current cursor position.",
        ),
        _kimi_function_tool(
            "mouse_up",
            "Release the primary mouse button at the current cursor position.",
        ),
        _kimi_function_tool(
            "click_current_position",
            "Click once at the current cursor or reticle position.",
        ),
        _kimi_function_tool(
            "click",
            "Click once at a screenshot-relative coordinate.",
            properties={"x": x, "y": y},
            required=("x", "y"),
        ),
        _kimi_function_tool(
            "drag",
            "Drag continuously from one screenshot-relative coordinate to another.",
            properties={
                "from_x": _coordinate_schema(
                    "x", width, "Drag start horizontal coordinate."
                ),
                "from_y": _coordinate_schema(
                    "y", height, "Drag start vertical coordinate."
                ),
                "to_x": _coordinate_schema("x", width, "Drag end horizontal coordinate."),
                "to_y": _coordinate_schema("y", height, "Drag end vertical coordinate."),
            },
            required=("from_x", "from_y", "to_x", "to_y"),
        ),
    ]


def kimi_k3_tool_arguments_to_action(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_kinds: Iterable[str],
    screenshot_size_px: tuple[int, int] = KIMI_K3_DEFAULT_SCREEN_SIZE_PX,
) -> Action:
    """Map Kimi-facing tool names to the benchmark's canonical six actions."""

    if tool_name not in KIMI_K3_MODEL_TOOL_NAMES:
        raise KimiK3OutputError(f"unsupported Kimi tool name: {tool_name!r}")
    expected = {
        "move_to": {"x", "y"},
        "mouse_down": set(),
        "mouse_up": set(),
        "click_current_position": set(),
        "click": {"x", "y"},
        "drag": {"from_x", "from_y", "to_x", "to_y"},
    }[tool_name]
    actual = set(arguments)
    if actual != expected:
        raise KimiK3OutputError(
            f"{tool_name} requires fields {sorted(expected)!r}, got {sorted(actual)!r}"
        )

    width, height = _screen_size_px(screenshot_size_px)

    def relative(value: Any, *, extent: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise KimiK3OutputError(
                f"{label} must be an integer screenshot pixel coordinate"
            )
        if not 0 <= value < extent:
            raise KimiK3OutputError(f"{label} must be in [0, {extent - 1}]")
        return int(round(float(value) / float(extent) * 1000.0))

    if tool_name == "move_to":
        canonical = {
            "kind": "move_to",
            "x": relative(arguments.get("x"), extent=width, label="move_to.x"),
            "y": relative(arguments.get("y"), extent=height, label="move_to.y"),
        }
    elif tool_name in {"mouse_down", "mouse_up"}:
        canonical = {"kind": tool_name}
    elif tool_name == "click_current_position":
        canonical = {"kind": "left_click"}
    elif tool_name == "click":
        canonical = {
            "kind": "click",
            "points": [
                [
                    relative(arguments.get("x"), extent=width, label="click.points[0].x"),
                    relative(
                        arguments.get("y"), extent=height, label="click.points[0].y"
                    ),
                ]
            ],
        }
    else:
        canonical = {
            "kind": "drag",
            "points": [
                [
                    relative(
                        arguments.get("from_x"),
                        extent=width,
                        label="drag.points[0].x",
                    ),
                    relative(
                        arguments.get("from_y"),
                        extent=height,
                        label="drag.points[0].y",
                    ),
                ],
                [
                    relative(
                        arguments.get("to_x"), extent=width, label="drag.points[1].x"
                    ),
                    relative(
                        arguments.get("to_y"), extent=height, label="drag.points[1].y"
                    ),
                ],
            ],
        }
    try:
        return six_action_arguments_to_action(canonical, allowed_kinds=allowed_kinds)
    except ValueError as exc:
        raise KimiK3OutputError(str(exc)) from exc


def _chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid Kimi base URL: {base_url!r}")
    path = f"{parsed.path.rstrip('/')}/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _safe_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _default_json_requester(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body), ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if body.get("stream") is True:
                payload = _consume_chat_completion_stream(response)
            else:
                payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {_safe_base_url(url)}: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Kimi API returned a non-object JSON payload")
    return payload


def _consume_chat_completion_stream(response: Any) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    saw_done = False
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            saw_done = True
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Kimi stream event is invalid JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise RuntimeError("Kimi stream event must be a JSON object")
        error = event.get("error")
        if error is not None:
            raise RuntimeError(f"Kimi stream returned an error: {error}")
        events.append(event)
    if not saw_done:
        raise RuntimeError("Kimi stream ended without [DONE]")
    return _assemble_chat_completion_stream(events)


def _assemble_chat_completion_stream(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    response_id: str | None = None
    model: str | None = None
    created: Any = None
    usage: Any = None
    role = "assistant"
    content_parts: list[str] = []
    reasoning_content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    event_count = 0

    for raw_event in events:
        event = dict(raw_event)
        event_count += 1
        if response_id is None and isinstance(event.get("id"), str):
            response_id = str(event["id"])
        if model is None and isinstance(event.get("model"), str):
            model = str(event["model"])
        if created is None and event.get("created") is not None:
            created = event.get("created")
        if event.get("usage") is not None:
            usage = event.get("usage")
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            choice_index = choice.get("index", 0)
            if choice_index != 0:
                raise RuntimeError(f"Kimi stream returned unexpected choice index {choice_index!r}")
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            if isinstance(delta.get("role"), str):
                role = str(delta["role"])
            if isinstance(delta.get("content"), str):
                content_parts.append(str(delta["content"]))
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_content_parts.append(str(delta["reasoning_content"]))
            if isinstance(delta.get("reasoning"), str):
                reasoning_parts.append(str(delta["reasoning"]))
            raw_tool_calls = delta.get("tool_calls")
            if not isinstance(raw_tool_calls, list):
                continue
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, Mapping):
                    continue
                tool_index = raw_tool_call.get("index", 0)
                if isinstance(tool_index, bool) or not isinstance(tool_index, int):
                    raise RuntimeError("Kimi stream tool-call index must be an integer")
                tool_call = tool_calls.setdefault(
                    tool_index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if isinstance(raw_tool_call.get("id"), str):
                    tool_call["id"] += str(raw_tool_call["id"])
                if isinstance(raw_tool_call.get("type"), str):
                    tool_call["type"] = str(raw_tool_call["type"])
                function = raw_tool_call.get("function")
                if isinstance(function, Mapping):
                    if isinstance(function.get("name"), str):
                        tool_call["function"]["name"] += str(function["name"])
                    if isinstance(function.get("arguments"), str):
                        tool_call["function"]["arguments"] += str(function["arguments"])

    if event_count == 0:
        raise RuntimeError("Kimi stream contained no JSON events")
    if finish_reason is None:
        raise RuntimeError("Kimi stream ended without finish_reason")
    message: dict[str, Any] = {
        "role": role,
        "content": "".join(content_parts),
    }
    if reasoning_content_parts:
        message["reasoning_content"] = "".join(reasoning_content_parts)
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    assembled: dict[str, Any] = {
        "id": response_id,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
        "stream_event_count": event_count,
        "stream_done": True,
    }
    if created is not None:
        assembled["created"] = created
    if usage is not None:
        assembled["usage"] = usage
    return assembled


def _retryable_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "temporarily",
            "connection",
            "invalid model name passed",
            "internal server error",
            "cuda out of memory",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _sanitize_for_log(value: Any, *, api_key: str, field_name: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_for_log(item, api_key=api_key, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_log(item, api_key=api_key) for item in value]
    if not isinstance(value, str):
        return value
    redacted = value.replace(api_key, "<redacted-api-key>") if api_key else value
    if field_name == "url" and redacted.startswith("data:"):
        return f"<redacted-data-url length={len(redacted)}>"
    return _DATA_URL_RE.sub("<redacted-data-url>", redacted)


class KimiK3ComputerBackend:
    """Kimi-K3 multimodal tool loop with preserved thinking and every screenshot."""

    def __init__(
        self,
        *,
        config: KimiK3ComputerConfig,
        call_log_dir: Path | None = None,
        prompt_extension: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        request_timeout_s: float = 180.0,
        request_retries: int = 6,
        requester: JsonRequester | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if request_retries < 1:
            raise ValueError("request_retries must be >= 1")
        self.config = config
        self.model = config.model
        self.call_log_dir = call_log_dir
        self.prompt_extension = prompt_extension.strip() if prompt_extension else None
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.request_retries = request_retries
        self.requester = requester or _default_json_requester
        self.protocol_track = KIMI_K3_PROTOCOL_TRACK
        self.history_strategy = KIMI_K3_HISTORY_STRATEGY
        self.prompt_version = KIMI_K3_PROMPT_VERSION
        self.computer_use_contract = KIMI_K3_COMPUTER_USE_CONTRACT
        self.provider_native_computer_use = False
        self.sampling_controls_unsupported = True
        self.canonical_action_kinds = KIMI_K3_CANONICAL_ACTION_KINDS
        self.model_tool_names = KIMI_K3_MODEL_TOOL_NAMES

        self._url = _chat_completions_url(config.base_url)
        self._messages: list[dict[str, Any]] = []
        self._pending_tool_call_id: str | None = None
        self._pending_tool_name: str | None = None
        self._pending_action_json: dict[str, Any] | None = None
        self._episode_instruction: str | None = None
        self._episode_key: tuple[str, str] | None = None
        self._last_history_length: int | None = None
        self._call_index = 0

        self.last_prediction_index: int | None = None
        self.last_raw_prediction: str | None = None
        self.last_think_text: str | None = None
        self.last_native_action_json: dict[str, Any] | None = None
        self.last_native_response: dict[str, Any] | None = None
        self.last_call_log_path: str | None = None
        self.last_error: str | None = None

    @property
    def native_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._messages)

    def reset_episode(self) -> None:
        self._messages.clear()
        self._pending_tool_call_id = None
        self._pending_tool_name = None
        self._pending_action_json = None
        self._episode_instruction = None
        self._episode_key = None
        self._last_history_length = None
        self.last_prediction_index = None
        self.last_raw_prediction = None
        self.last_think_text = None
        self.last_native_action_json = None
        self.last_native_response = None
        self.last_call_log_path = None
        self.last_error = None

    def reset(self) -> None:
        self.reset_episode()

    def close(self) -> None:
        self.reset_episode()

    @staticmethod
    def _episode_key_for(obs: Observation) -> tuple[str, str] | None:
        metadata = obs.metadata or {}
        for name in ("episode_id", "task_id"):
            value = metadata.get(name)
            if value is not None and str(value).strip():
                return name, str(value)
        return None

    def _maybe_reset_episode(self, obs: Observation, history: list[StepResult]) -> None:
        key = self._episode_key_for(obs)
        changed_key = key is not None and self._episode_key is not None and key != self._episode_key
        restarted_history = (
            bool(self._messages)
            and self._last_history_length is not None
            and len(history) <= self._last_history_length
        )
        if changed_key or restarted_history:
            self.reset_episode()
        if key is not None:
            self._episode_key = key

    @staticmethod
    def _screenshot_data_url(obs: Observation) -> str:
        path = Path(obs.screenshot_path)
        if not path.is_file():
            raise FileNotFoundError(f"screenshot is missing: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _task_type(obs: Observation) -> str | None:
        metadata = obs.metadata or {}
        for key in ("task_type", "benchmark", "task_id", "episode_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _screen_message(self, obs: Observation, *, step_index: int) -> dict[str, Any]:
        task = canonicalize_task_requirement(
            obs.instruction,
            task_type=self._task_type(obs),
            target_label=(obs.metadata or {}).get("target_label"),
        ).strip()
        if not task:
            raise KimiK3OutputError("observation instruction must not be empty")
        width, height = _screen_size_px(obs.size_px)
        coordinate_instruction = (
            f"The attached screenshot is {width}x{height} pixels. Use original screenshot "
            f"pixels: x must be 0-{width - 1} and y must be 0-{height - 1}."
        )
        if step_index == 1:
            text = (
                f"Task: {task}\n{coordinate_instruction}\n"
                "Use exactly one provided UI tool."
            )
        else:
            text = (
                "The attached image is the updated state after executing the previous "
                "action and is the current observation. Use previous tool calls as context "
                f"only. {coordinate_instruction} Return exactly one tool call for step "
                f"{step_index}."
            )
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": self._screenshot_data_url(obs), "detail": "high"},
                },
            ],
        }

    def _append_current_turn(self, obs: Observation, *, step_index: int) -> None:
        if not self._messages:
            self._episode_instruction = obs.instruction
            self._messages.append(self._screen_message(obs, step_index=1))
            return
        if obs.instruction != self._episode_instruction:
            raise ValueError("observation instruction changed; call reset_episode() first")
        if (
            self._pending_tool_call_id is None
            or self._pending_tool_name is None
            or self._pending_action_json is None
        ):
            raise KimiK3OutputError("cannot continue without the prior Kimi tool call")
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": self._pending_tool_call_id,
                "name": self._pending_tool_name,
                "content": json.dumps(
                    {"ok": True, "executed_action": self._pending_action_json},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        self._messages.append(self._screen_message(obs, step_index=step_index))
        self._pending_tool_call_id = None
        self._pending_tool_name = None
        self._pending_action_json = None

    def _request_body(
        self,
        screenshot_size_px: tuple[int, int] = KIMI_K3_DEFAULT_SCREEN_SIZE_PX,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "\n\n".join(p for p in (KIMI_K3_SYSTEM_PROMPT, self.prompt_extension) if p)},
                *copy.deepcopy(self._messages),
            ],
            "tools": kimi_k3_six_action_tools(screenshot_size_px),
            "tool_choice": self.config.tool_choice,
            "reasoning_effort": self.config.reasoning_effort,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

    @staticmethod
    def _parse_response(
        response: Mapping[str, Any],
        *,
        allowed_kinds: tuple[str, ...],
        screenshot_size_px: tuple[int, int] = KIMI_K3_DEFAULT_SCREEN_SIZE_PX,
    ) -> _ParsedToolCall:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise KimiK3OutputError("Kimi response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise KimiK3OutputError("Kimi choice must be an object")
        if choice.get("finish_reason") != "tool_calls":
            raise KimiK3OutputError("Kimi finish_reason must be 'tool_calls'")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise KimiK3OutputError("Kimi choice is missing its assistant message")
        assistant_message = copy.deepcopy(dict(message))
        if assistant_message.get("role") != "assistant":
            raise KimiK3OutputError("Kimi message role must be assistant")
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise KimiK3OutputError("Kimi must return exactly one tool_call")
        tool_call = tool_calls[0]
        if not isinstance(tool_call, Mapping):
            raise KimiK3OutputError("Kimi tool_call must be an object")
        tool_call_id = tool_call.get("id")
        function = tool_call.get("function")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise KimiK3OutputError("Kimi tool_call is missing id")
        if not isinstance(function, Mapping):
            raise KimiK3OutputError("Kimi tool_call is missing its function")
        tool_name = function.get("name")
        if not isinstance(tool_name, str) or tool_name not in KIMI_K3_MODEL_TOOL_NAMES:
            raise KimiK3OutputError("Kimi returned an unexpected tool name")
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise KimiK3OutputError(f"Kimi tool arguments are invalid JSON: {exc}") from exc
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise KimiK3OutputError("Kimi tool arguments must be an object")
        action = kimi_k3_tool_arguments_to_action(
            tool_name,
            arguments,
            allowed_kinds=allowed_kinds,
            screenshot_size_px=screenshot_size_px,
        )
        return _ParsedToolCall(
            assistant_message=assistant_message,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            action=action,
        )

    def _write_call_log(self, payload: Mapping[str, Any]) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.call_log_dir / f"call_{self._call_index:03d}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        sanitized = _sanitize_for_log(dict(payload), api_key=self.config.api_key)
        temporary.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self.last_call_log_path = str(path)

    def _safe_error_text(self, exc: BaseException) -> str:
        text = str(exc)
        if self.config.api_key:
            text = text.replace(self.config.api_key, "<redacted-api-key>")
        return _DATA_URL_RE.sub("<redacted-data-url>", text)

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action:
        del budget, condition
        allowed = tuple(dict.fromkeys(str(kind) for kind in allowed_kinds))
        if allowed != self.canonical_action_kinds:
            raise ValueError(
                "Kimi-K3 evaluation requires the exact six-action benchmark contract "
                f"{self.canonical_action_kinds!r}, got {allowed!r}"
            )
        self._maybe_reset_episode(obs, history)
        step_index = len(history) + 1
        self._append_current_turn(obs, step_index=step_index)
        body = self._request_body(obs.size_px)
        self._call_index += 1
        self.last_prediction_index = step_index
        self.last_call_log_path = None
        started_at = time.time()
        attempts: list[dict[str, Any]] = []
        response: Mapping[str, Any] | None = None
        last_error: BaseException | None = None
        headers = {"Content-Type": "application/json"}
        if not self.config.unauthenticated:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for attempt in range(1, self.request_retries + 1):
            attempt_started = time.time()
            try:
                candidate = self.requester(
                    self._url,
                    headers,
                    body,
                    self.request_timeout_s,
                )
                if not isinstance(candidate, Mapping):
                    raise TypeError("Kimi requester returned a non-mapping response")
                response = candidate
                attempts.append(
                    {"attempt": attempt, "latency_s": time.time() - attempt_started, "error": None}
                )
                break
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "latency_s": time.time() - attempt_started,
                        "error": {
                            "type": type(exc).__name__,
                            "message": self._safe_error_text(exc),
                        },
                    }
                )
                if attempt >= self.request_retries or not _retryable_error(exc):
                    break
                time.sleep(min(2.0 * (2 ** (attempt - 1)), 30.0))

        common_log = {
            "provider": "kimi",
            "model": self.model,
            "protocol_track": self.protocol_track,
            "history_strategy": self.history_strategy,
            "prompt_version": self.prompt_version,
            "computer_use_contract": self.computer_use_contract,
            "provider_native_computer_use": self.provider_native_computer_use,
            "action_kinds": list(self.canonical_action_kinds),
            "model_tool_names": list(self.model_tool_names),
            "base_url": _safe_base_url(self.config.base_url),
            "request": body,
            "request_attempts": attempts,
            "api_key_persisted": False,
        }
        if response is None:
            cause = last_error or RuntimeError("Kimi returned no response")
            self.last_error = f"{type(cause).__name__}: {self._safe_error_text(cause)}"
            self._write_call_log(
                {
                    **common_log,
                    "response": None,
                    "error": self.last_error,
                    "latency_s": time.time() - started_at,
                }
            )
            raise KimiK3ApiError(
                f"Kimi returned no response after {len(attempts)} attempt(s)"
            ) from last_error

        try:
            parsed = self._parse_response(
                response,
                allowed_kinds=allowed,
                screenshot_size_px=obs.size_px,
            )
        except Exception as exc:
            error = exc if isinstance(exc, KimiK3OutputError) else KimiK3OutputError(str(exc))
            self.last_error = f"{type(error).__name__}: {error}"
            self._write_call_log(
                {
                    **common_log,
                    "response": response,
                    "error": self.last_error,
                    "latency_s": time.time() - started_at,
                }
            )
            raise error from exc

        self._messages.append(parsed.assistant_message)
        self._pending_tool_call_id = parsed.tool_call_id
        self._pending_tool_name = parsed.tool_name
        self._pending_action_json = {
            "name": parsed.tool_name,
            "arguments": dict(parsed.arguments),
        }
        self._last_history_length = len(history)
        self.last_native_response = copy.deepcopy(dict(response))
        self.last_native_action_json = {
            "name": parsed.tool_name,
            "arguments": dict(parsed.arguments),
        }
        self.last_raw_prediction = json.dumps(
            self.last_native_action_json,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        reasoning = parsed.assistant_message.get("reasoning_content")
        if reasoning is None:
            reasoning = parsed.assistant_message.get("reasoning")
        self.last_think_text = str(reasoning) if reasoning is not None else None
        self.last_error = None
        self._write_call_log(
            {
                **common_log,
                "response": response,
                "tool_call_id": parsed.tool_call_id,
                "action": {
                    "name": parsed.tool_name,
                    "arguments": parsed.arguments,
                },
                "canonical_action": parsed.action.to_dict(),
                "error": None,
                "latency_s": time.time() - started_at,
            }
        )
        return parsed.action


__all__ = [
    "KIMI_K3_CANONICAL_ACTION_KINDS",
    "KIMI_K3_COMPUTER_USE_CONTRACT",
    "KIMI_K3_HISTORY_STRATEGY",
    "KIMI_K3_MODEL_TOOL_NAMES",
    "KIMI_K3_NATIVE_COORDINATE_FORMAT",
    "KIMI_K3_PROMPT_VERSION",
    "KIMI_K3_PROTOCOL_TRACK",
    "KIMI_K3_SYSTEM_PROMPT",
    "KimiK3ApiError",
    "KimiK3ComputerBackend",
    "KimiK3ComputerConfig",
    "KimiK3OutputError",
    "kimi_k3_six_action_tools",
    "kimi_k3_tool_arguments_to_action",
]
