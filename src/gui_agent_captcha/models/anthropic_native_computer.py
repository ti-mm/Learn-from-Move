from __future__ import annotations

import base64
import copy
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from ..actions import Action
from ..core import Observation, StepResult

JsonRequester = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]
ObservationSupplier = Callable[[], Observation]

ANTHROPIC_VERSION = "2023-06-01"
COMPUTER_TOOLSET_TYPE = "computer_toolset_20260801"
COMPUTER_TOOLSET_NAME = "computer"
COMPUTER_TOOLSET_MEMBERS = (
    "key",
    "hold_key",
    "type",
    "cursor_position",
    "mouse_move",
    "left_mouse_down",
    "left_mouse_up",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "wait",
    "screenshot",
    "zoom",
)
NOT_EXECUTED_ERROR = "Not executed: an earlier computer action in this turn failed."

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_REQUEST_RETRIES = 3
DEFAULT_MAX_INTERNAL_TOOL_ROUNDS = 8
DEFAULT_SCREENSHOTS_TO_KEEP: int | None = None
DEFAULT_SCREENSHOT_PRUNE_INTERVAL = 25
NATIVE_PROTOCOL_TRACK = "anthropic_computer_toolset_20260801"

_DATA_URL_RE = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+")
_IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class AnthropicNativeApiError(RuntimeError):
    """The native Anthropic request failed."""


class AnthropicResponseBudgetExhausted(RuntimeError):
    """All allowed responses have been received and their tool batches drained."""


class AnthropicNativeOutputError(ValueError):
    """Claude returned a response outside the supported computer toolset contract."""


@dataclass(frozen=True)
class AnthropicComputerToolAction:
    """One unmodified member call from Anthropic's built-in computer toolset."""

    name: str
    input: dict[str, Any]
    toolset_name: str = COMPUTER_TOOLSET_NAME

    @property
    def kind(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "toolset_name": self.toolset_name,
            "name": self.name,
            "input": copy.deepcopy(self.input),
        }


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
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Anthropic returned a non-object JSON response from {url}")
    return payload


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
            "502",
            "503",
            "504",
        )
    )


def _sanitize_for_log(value: Any, *, field_name: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_for_log(item, field_name=str(key)) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if (
        isinstance(value, str)
        and field_name
        and field_name.lower()
        in {
            "api_key",
            "authorization",
            "x-api-key",
        }
    ):
        return "<redacted-secret>"
    if isinstance(value, str) and field_name == "data":
        return f"<redacted-image-data length={len(value)}>"
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        return f"<redacted-data-url length={len(value)}>"
    if isinstance(value, str) and _DATA_URL_RE.search(value):
        return _DATA_URL_RE.sub("<redacted-data-url>", value)
    if (
        isinstance(value, str)
        and len(value) > 256
        and all(character.isalnum() or character in "+/=_-" for character in value)
    ):
        return f"<redacted-long-payload length={len(value)}>"
    return value


def _sanitize_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _media_type(path: Path) -> str:
    return _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "image/png")


def _screenshot_block(obs: Observation) -> dict[str, Any]:
    path = Path(obs.screenshot_path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _media_type(path),
            "data": encoded,
        },
    }


def _success_tool_result(
    tool_use_id: str,
    content: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "toolset_name": COMPUTER_TOOLSET_NAME,
        "content": content,
    }


def _error_tool_result(tool_use_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "toolset_name": COMPUTER_TOOLSET_NAME,
        "is_error": True,
        "content": text,
    }


def _prune_old_screenshot_blocks(
    messages: list[dict[str, Any]],
    *,
    screenshots_to_keep: int,
) -> None:
    screenshot_blocks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            result_content = block.get("content")
            if not isinstance(result_content, list):
                continue
            for item in result_content:
                if isinstance(item, dict) and item.get("type") == "image":
                    screenshot_blocks.append((block, item))

    remove_count = max(0, len(screenshot_blocks) - screenshots_to_keep)
    for tool_result, image in screenshot_blocks[:remove_count]:
        content = tool_result.get("content")
        if isinstance(content, list):
            tool_result["content"] = [
                {"type": "text", "text": "[Image Omitted]"} if item is image else item
                for item in content
            ]


def _episode_key(obs: Observation) -> tuple[str, str] | None:
    metadata = obs.metadata or {}
    for name in ("episode_id", "task_id"):
        value = metadata.get(name)
        if value is not None and str(value).strip():
            return name, str(value)
    return None


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnthropicNativeOutputError(f"{label} must be numeric")
    if not (float("-inf") < float(value) < float("inf")):
        raise AnthropicNativeOutputError(f"{label} must be finite")
    return float(value)


def _pixel_coordinate(
    value: Any,
    *,
    size_px: tuple[int, int],
    label: str,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnthropicNativeOutputError(f"{label} must be [x, y]")
    width, height = size_px
    if width <= 0 or height <= 0:
        raise AnthropicNativeOutputError(f"observation size must be positive, got {size_px!r}")
    x = _number(value[0], label=f"{label}[0]")
    y = _number(value[1], label=f"{label}[1]")
    if not 0 <= x < width or not 0 <= y < height:
        raise AnthropicNativeOutputError(
            f"{label} {value!r} is outside screenshot {width}x{height}"
        )
    return x, y


def _reject_unexpected_inputs(
    member_name: str,
    native_input: Mapping[str, Any],
    allowed_keys: set[str],
) -> None:
    unexpected = sorted(str(key) for key in native_input if key not in allowed_keys)
    if unexpected:
        raise AnthropicNativeOutputError(
            f"computer member {member_name!r} received unexpected input keys {unexpected!r}"
        )


def _modifier_keys(value: Any, *, label: str = "text") -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str) or not value.strip():
        raise AnthropicNativeOutputError(f"{label} must be a non-empty modifier string")
    modifiers = tuple(part.casefold() for part in value.split("+"))
    allowed = {"shift", "ctrl", "alt", "super"}
    if any(part not in allowed for part in modifiers) or len(set(modifiers)) != len(modifiers):
        raise AnthropicNativeOutputError(
            f"{label} must contain unique '+'-joined modifiers from {sorted(allowed)!r}"
        )
    return modifiers


def _positive_integer(value: Any, *, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AnthropicNativeOutputError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise AnthropicNativeOutputError(f"{label} must be <= {maximum}")
    return value


def _computer_action(
    member_name: str,
    native_input: Mapping[str, Any],
    *,
    size_px: tuple[int, int],
) -> AnthropicComputerToolAction:
    empty_members = {"cursor_position", "left_mouse_down", "left_mouse_up", "screenshot"}
    click_members = {
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
    }
    if member_name in empty_members:
        _reject_unexpected_inputs(member_name, native_input, set())
    elif member_name in click_members:
        _reject_unexpected_inputs(member_name, native_input, {"coordinate", "text"})
        coordinate = native_input.get("coordinate")
        if coordinate is not None:
            _pixel_coordinate(coordinate, size_px=size_px, label="coordinate")
        _modifier_keys(native_input.get("text"))
    if member_name == "mouse_move":
        _reject_unexpected_inputs(member_name, native_input, {"coordinate"})
        _pixel_coordinate(native_input.get("coordinate"), size_px=size_px, label="coordinate")
    elif member_name == "left_click_drag":
        _reject_unexpected_inputs(
            member_name,
            native_input,
            {"start_coordinate", "coordinate", "text"},
        )
        _pixel_coordinate(
            native_input.get("start_coordinate"),
            size_px=size_px,
            label="start_coordinate",
        )
        _pixel_coordinate(
            native_input.get("coordinate"), size_px=size_px, label="coordinate"
        )
        _modifier_keys(native_input.get("text"))
    elif member_name == "scroll":
        _reject_unexpected_inputs(
            member_name,
            native_input,
            {"scroll_direction", "scroll_amount", "coordinate", "text"},
        )
        if native_input.get("scroll_direction") not in {"up", "down", "left", "right"}:
            raise AnthropicNativeOutputError(
                "scroll_direction must be one of 'up', 'down', 'left', or 'right'"
            )
        _positive_integer(native_input.get("scroll_amount"), label="scroll_amount")
        coordinate = native_input.get("coordinate")
        if coordinate is not None:
            _pixel_coordinate(coordinate, size_px=size_px, label="coordinate")
        _modifier_keys(native_input.get("text"))
    elif member_name in {"type", "key", "hold_key"}:
        allowed_keys = {"text"}
        if member_name == "key":
            allowed_keys.add("repeat")
        elif member_name == "hold_key":
            allowed_keys.add("duration")
        _reject_unexpected_inputs(member_name, native_input, allowed_keys)
        text = native_input.get("text")
        if not isinstance(text, str) or not text:
            raise AnthropicNativeOutputError(f"computer member {member_name!r} requires text")
        if member_name == "key":
            _positive_integer(native_input.get("repeat", 1), label="repeat", maximum=100)
        elif member_name == "hold_key":
            duration = _number(native_input.get("duration"), label="duration")
            if not 0 <= duration <= 300:
                raise AnthropicNativeOutputError(
                    "hold_key duration must be between 0 and 300 seconds"
                )
    elif member_name == "wait":
        _reject_unexpected_inputs(member_name, native_input, {"duration"})
        duration = _number(native_input.get("duration"), label="duration")
        if not 0 <= duration <= 300:
            raise AnthropicNativeOutputError("wait duration must be between 0 and 300 seconds")
    elif member_name == "zoom":
        _reject_unexpected_inputs(member_name, native_input, {"region"})
        region = native_input.get("region")
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise AnthropicNativeOutputError("region must be [x0, y0, x1, y1]")
        x0 = _number(region[0], label="region[0]")
        y0 = _number(region[1], label="region[1]")
        x1 = _number(region[2], label="region[2]")
        y1 = _number(region[3], label="region[3]")
        width, height = size_px
        if not (0 <= x0 < width and 0 <= y0 < height and x1 <= width and y1 <= height):
            raise AnthropicNativeOutputError(
                f"zoom region {region!r} is outside screenshot {width}x{height}"
            )
        if x1 <= x0 or y1 <= y0:
            raise AnthropicNativeOutputError("zoom region must have x1 > x0 and y1 > y0")
    elif member_name not in empty_members and member_name not in click_members:
        raise AnthropicNativeOutputError(f"unsupported computer toolset member {member_name!r}")
    return AnthropicComputerToolAction(name=member_name, input=copy.deepcopy(dict(native_input)))


def _zoom_block(obs: Observation, region: Any) -> dict[str, Any]:
    action = _computer_action("zoom", {"region": region}, size_px=obs.size_px)
    x0, y0, x1, y1 = (int(round(value)) for value in action.input["region"])
    with Image.open(obs.screenshot_path) as image:
        cropped = image.crop((x0, y0, x1, y1))
        encoded = BytesIO()
        cropped.save(encoded, format="PNG")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(encoded.getvalue()).decode("ascii"),
        },
    }


class AnthropicNativeComputerBackend:
    """Stateful adapter for Anthropic's GA ``computer_toolset_20260801``."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        system_prompt: str | None = None,
        prompt_extension: str | None = None,
        call_log_dir: Path | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
        max_internal_tool_rounds: int = DEFAULT_MAX_INTERNAL_TOOL_ROUNDS,
        max_responses: int | None = None,
        screenshots_to_keep: int | None = DEFAULT_SCREENSHOTS_TO_KEEP,
        screenshot_prune_interval: int = DEFAULT_SCREENSHOT_PRUNE_INTERVAL,
        observation_supplier: ObservationSupplier | None = None,
        requester: JsonRequester | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if request_retries < 1:
            raise ValueError("request_retries must be >= 1")
        if max_internal_tool_rounds < 1:
            raise ValueError("max_internal_tool_rounds must be >= 1")
        if max_responses is not None and max_responses < 1:
            raise ValueError("max_responses must be >= 1")
        if screenshots_to_keep is not None and screenshots_to_keep < 1:
            raise ValueError("screenshots_to_keep must be >= 1")
        if screenshot_prune_interval < 1:
            raise ValueError("screenshot_prune_interval must be >= 1")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.prompt_extension = prompt_extension.strip() if prompt_extension else None
        self.call_log_dir = call_log_dir
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.request_retries = request_retries
        self.max_internal_tool_rounds = max_internal_tool_rounds
        self.max_responses = max_responses
        self.response_count = 0
        self.screenshots_to_keep = screenshots_to_keep
        self.screenshot_prune_interval = screenshot_prune_interval
        self.observation_supplier = observation_supplier
        self.requester = requester or _default_json_requester

        self.protocol_track = NATIVE_PROTOCOL_TRACK
        if screenshots_to_keep is None:
            self.history_strategy = "native_full_structured_messages_all_screenshots"
        else:
            self.history_strategy = (
                f"native_full_structured_messages_recent_{screenshots_to_keep}_screenshots_"
                f"prune_every_{screenshot_prune_interval}_turns"
            )
        self.sampling_controls_unsupported = True
        self._call_index = 0
        self._messages: list[dict[str, Any]] = []
        self._episode_instruction: str | None = None
        self._episode_key: tuple[str, str] | None = None
        self._last_history_length: int | None = None
        self._pending_external_tool_use: dict[str, Any] | None = None
        self._pending_tool_uses: list[dict[str, Any]] = []
        self._pending_tool_results: list[dict[str, Any]] = []
        self._screenshot_result_count = 0

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
        self.response_count = 0
        self._messages.clear()
        self._episode_instruction = None
        self._episode_key = None
        self._last_history_length = None
        self._pending_external_tool_use = None
        self._pending_tool_uses.clear()
        self._pending_tool_results.clear()
        self._screenshot_result_count = 0
        self.last_raw_prediction = None
        self.last_think_text = None
        self.last_native_action_json = None
        self.last_native_response = None
        self.last_prediction_index = None
        self.last_call_log_path = None
        self.last_error = None

    def reset(self) -> None:
        self.reset_episode()

    def close(self) -> None:
        self.reset_episode()

    def _safe_error_text(self, exc: BaseException) -> str:
        text = str(exc).replace(self.api_key, "<redacted-api-key>")
        return _DATA_URL_RE.sub("<redacted-data-url>", text)

    def _write_call_log(self, payload: Mapping[str, Any]) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.call_log_dir / f"call_{self._call_index:03d}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_sanitize_for_log(dict(payload)), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self.last_call_log_path = str(path)

    def _maybe_reset_episode(self, obs: Observation, history: list[StepResult]) -> None:
        key = _episode_key(obs)
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

    def _start_or_continue_episode(self, obs: Observation) -> None:
        if not self._messages:
            self._episode_instruction = obs.instruction
            self._messages.append({"role": "user", "content": obs.instruction})
            return
        if obs.instruction != self._episode_instruction:
            raise ValueError("observation instruction changed; call reset_episode() first")
        if self._pending_external_tool_use is not None:
            content: list[dict[str, Any]] = [{"type": "text", "text": "OK"}]
            if not self._pending_tool_uses:
                content.append(_screenshot_block(obs))
            self._pending_tool_results.append(
                _success_tool_result(self._pending_external_tool_use["id"], content)
            )
            self._pending_external_tool_use = None

    def _append_pending_tool_results(self) -> None:
        if not self._pending_tool_results:
            return
        self._messages.append(
            {"role": "user", "content": copy.deepcopy(self._pending_tool_results)}
        )
        self._screenshot_result_count += sum(
            item.get("type") == "image"
            for result in self._pending_tool_results
            if isinstance(result.get("content"), list)
            for item in result["content"]
            if isinstance(item, dict)
        )
        self._pending_tool_results.clear()
        if (
            self.screenshots_to_keep is not None
            and self._screenshot_result_count
            and self._screenshot_result_count % self.screenshot_prune_interval == 0
        ):
            _prune_old_screenshot_blocks(
                self._messages,
                screenshots_to_keep=self.screenshots_to_keep,
            )

    def _fresh_observation(self, prior: Observation, *, member_name: str) -> Observation:
        if self.observation_supplier is None:
            raise AnthropicNativeOutputError(
                f"computer member {member_name!r} requires an observation_supplier that "
                "captures a fresh screenshot"
            )
        fresh = self.observation_supplier()
        if not isinstance(fresh, Observation):
            raise AnthropicNativeOutputError(
                f"observation_supplier must return an Observation after {member_name!r}"
            )
        if fresh.instruction != prior.instruction:
            raise AnthropicNativeOutputError(
                "observation_supplier returned an observation from a different task"
            )
        return fresh

    def _request_parts(
        self,
        obs: Observation,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        width, height = obs.size_px
        if width <= 0 or height <= 0:
            raise ValueError(f"observation size must be positive, got {obs.size_px!r}")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": copy.deepcopy(self._messages),
            "tools": [{"type": COMPUTER_TOOLSET_TYPE}],
        }
        system = "\n\n".join(p for p in (self.system_prompt, self.prompt_extension) if p)
        if system:
            body["system"] = system
        return (
            f"{self.base_url}/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            body,
        )

    def _request(
        self,
        obs: Observation,
        *,
        internal_round: int,
        history_length: int,
    ) -> Mapping[str, Any]:
        if self.max_responses is not None and self.response_count >= self.max_responses:
            raise AnthropicResponseBudgetExhausted("model response budget exhausted")
        url, headers, body = self._request_parts(obs)
        self._call_index += 1
        self.last_prediction_index = self._call_index
        self.last_call_log_path = None
        started_at = time.time()
        response: Mapping[str, Any] | None = None
        last_error: BaseException | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.request_retries + 1):
            attempt_started = time.time()
            try:
                candidate = self.requester(url, headers, body, self.request_timeout_s)
                if not isinstance(candidate, Mapping):
                    raise TypeError("requester returned a non-mapping response")
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
        if response is None:
            exc = last_error or RuntimeError("Anthropic returned no response")
            self.last_error = f"{type(exc).__name__}: {self._safe_error_text(exc)}"
            self._write_call_log(
                {
                    "protocol_track": self.protocol_track,
                    "model": self.model,
                    "base_url": _sanitize_base_url(self.base_url),
                    "request": body,
                    "internal_round": internal_round,
                    "runner_history_length": history_length,
                    "request_attempts": attempts,
                    "error": self.last_error,
                    "api_key_persisted": False,
                    "latency_s": time.time() - started_at,
                }
            )
            raise AnthropicNativeApiError("Anthropic native computer request failed") from exc
        self.response_count += 1
        self._write_call_log(
            {
                "protocol_track": self.protocol_track,
                "model": self.model,
                "base_url": _sanitize_base_url(self.base_url),
                "request": body,
                "response": response,
                "internal_round": internal_round,
                "runner_history_length": history_length,
                "request_attempts": attempts,
                "history_strategy": self.history_strategy,
                "api_key_persisted": False,
                "latency_s": time.time() - started_at,
            }
        )
        return response

    def _tool_uses(
        self,
        response: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        stop_reason = response.get("stop_reason")
        content = response.get("content")
        if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
            raise AnthropicNativeOutputError("Anthropic response content must be a block list")
        tool_uses = [block for block in content if block.get("type") == "tool_use"]
        if stop_reason == "end_turn":
            if tool_uses:
                raise AnthropicNativeOutputError(
                    "Anthropic end_turn response must not contain a tool_use block"
                )
            return copy.deepcopy(content), None
        if stop_reason != "tool_use":
            raise AnthropicNativeOutputError(
                f"Anthropic response stop_reason is {stop_reason!r}, expected 'tool_use' or 'end_turn'"
            )
        if not tool_uses:
            raise AnthropicNativeOutputError(
                "Anthropic tool_use response does not contain a computer member call"
            )
        for tool_use in tool_uses:
            if tool_use.get("toolset_name") != COMPUTER_TOOLSET_NAME:
                raise AnthropicNativeOutputError(
                    "unexpected computer toolset_name "
                    f"{tool_use.get('toolset_name')!r}; expected {COMPUTER_TOOLSET_NAME!r}"
                )
            member_name = tool_use.get("name")
            if member_name not in COMPUTER_TOOLSET_MEMBERS:
                raise AnthropicNativeOutputError(
                    f"unexpected computer toolset member {member_name!r}"
                )
            if not isinstance(tool_use.get("id"), str) or not tool_use["id"]:
                raise AnthropicNativeOutputError("computer member tool_use is missing an id")
            if not isinstance(tool_use.get("input"), dict):
                raise AnthropicNativeOutputError("computer member input must be an object")
        return copy.deepcopy(content), copy.deepcopy(tool_uses)

    def _set_last_native_action(self, tool_use: Mapping[str, Any]) -> None:
        native_action = {
            "name": tool_use["name"],
            "toolset_name": tool_use["toolset_name"],
            "input": copy.deepcopy(tool_use["input"]),
        }
        self.last_native_action_json = native_action
        self.last_raw_prediction = json.dumps(
            native_action,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _fail_current_batch(self, tool_use_id: str, error: AnthropicNativeOutputError) -> None:
        self.last_error = f"{type(error).__name__}: {error}"
        self._pending_tool_results.append(_error_tool_result(tool_use_id, str(error)))
        for skipped in self._pending_tool_uses:
            self._pending_tool_results.append(_error_tool_result(skipped["id"], NOT_EXECUTED_ERROR))
        self._pending_tool_uses.clear()
        self._append_pending_tool_results()

    def report_environment_error(self, error: BaseException) -> None:
        """Return one failed external member and halt the rest of its batch."""

        tool_use = self._pending_external_tool_use
        if tool_use is None:
            raise RuntimeError("there is no pending computer action to fail")
        self._pending_external_tool_use = None
        safe_text = self._safe_error_text(error)
        output_error = AnthropicNativeOutputError(
            f"environment failed to execute computer member {tool_use['name']!r}: {safe_text}"
        )
        self._fail_current_batch(tool_use["id"], output_error)

    def _drain_pending_tool_uses(
        self,
        obs: Observation,
        *,
        history_length: int,
    ) -> tuple[AnthropicComputerToolAction | None, Observation]:
        current_obs = obs
        while self._pending_tool_uses:
            tool_use = self._pending_tool_uses.pop(0)
            self._set_last_native_action(tool_use)
            member_name = str(tool_use["name"])
            native_input = tool_use["input"]
            try:
                if member_name == "screenshot":
                    _computer_action(member_name, native_input, size_px=current_obs.size_px)
                    current_obs = self._fresh_observation(
                        current_obs,
                        member_name=member_name,
                    )
                    self._pending_tool_results.append(
                        _success_tool_result(tool_use["id"], [_screenshot_block(current_obs)])
                    )
                    continue
                if member_name == "zoom":
                    _computer_action(member_name, native_input, size_px=current_obs.size_px)
                    current_obs = self._fresh_observation(
                        current_obs,
                        member_name=member_name,
                    )
                    self._pending_tool_results.append(
                        _success_tool_result(
                            tool_use["id"],
                            [_zoom_block(current_obs, native_input.get("region"))],
                        )
                    )
                    continue
                if member_name == "wait":
                    _computer_action(member_name, native_input, size_px=current_obs.size_px)
                    content: list[dict[str, Any]] = [{"type": "text", "text": "OK"}]
                    if not self._pending_tool_uses:
                        content.append(_screenshot_block(current_obs))
                    self._pending_tool_results.append(_success_tool_result(tool_use["id"], content))
                    continue
                if member_name == "cursor_position":
                    _computer_action(member_name, native_input, size_px=current_obs.size_px)
                    if current_obs.cursor_xy is None:
                        raise AnthropicNativeOutputError(
                            "cursor_position is unavailable because the environment did not "
                            "report cursor_xy"
                        )
                    x, y = current_obs.cursor_xy
                    content = [{"type": "text", "text": f"X={round(x)},Y={round(y)}"}]
                    if not self._pending_tool_uses:
                        content.append(_screenshot_block(current_obs))
                    self._pending_tool_results.append(_success_tool_result(tool_use["id"], content))
                    continue

                action = _computer_action(
                    member_name,
                    native_input,
                    size_px=current_obs.size_px,
                )
                self._pending_external_tool_use = tool_use
                self._last_history_length = history_length
                self.last_error = None
                return action, current_obs
            except AnthropicNativeOutputError as error:
                self._fail_current_batch(tool_use["id"], error)
                return None, current_obs

        self._append_pending_tool_results()
        return None, current_obs

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action | AnthropicComputerToolAction | None:
        del budget, condition
        self._maybe_reset_episode(obs, history)
        continuing_batch = bool(
            self._pending_external_tool_use is not None or self._pending_tool_uses
        )
        self._start_or_continue_episode(obs)
        self.last_raw_prediction = None
        self.last_native_action_json = None
        self.last_error = None
        if not continuing_batch:
            self.last_think_text = None
            self.last_native_response = None
        current_obs = obs
        request_round = 0

        while True:
            action, current_obs = self._drain_pending_tool_uses(
                current_obs,
                history_length=len(history),
            )
            if action is not None:
                return action
            if request_round >= self.max_internal_tool_rounds:
                raise AnthropicNativeOutputError(
                    "computer toolset loop exhausted internal rounds without an environment action"
                )

            request_round += 1
            response = self._request(
                current_obs,
                internal_round=request_round,
                history_length=len(history),
            )
            self.last_native_response = copy.deepcopy(dict(response))
            try:
                assistant_content, tool_uses = self._tool_uses(response)
                thinking = [
                    str(block.get("thinking"))
                    for block in assistant_content
                    if block.get("type") == "thinking" and block.get("thinking") is not None
                ]
                self.last_think_text = "\n".join(thinking) or None
                self._messages.append({"role": "assistant", "content": assistant_content})
                if tool_uses is None:
                    native_action = {"type": "end_turn"}
                    self.last_native_action_json = native_action
                    self.last_raw_prediction = json.dumps(
                        native_action,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    self._last_history_length = len(history)
                    return None
                self._pending_tool_uses = tool_uses
                self._pending_tool_results = []
            except Exception as exc:
                error = (
                    exc
                    if isinstance(exc, AnthropicNativeOutputError)
                    else AnthropicNativeOutputError(str(exc))
                )
                self.last_error = f"{type(error).__name__}: {error}"
                self._write_call_log(
                    {
                        "protocol_track": self.protocol_track,
                        "model": self.model,
                        "base_url": _sanitize_base_url(self.base_url),
                        "response": response,
                        "internal_round": request_round,
                        "error": self.last_error,
                        "api_key_persisted": False,
                    }
                )
                raise error from exc


__all__ = [
    "ANTHROPIC_VERSION",
    "COMPUTER_TOOLSET_MEMBERS",
    "COMPUTER_TOOLSET_NAME",
    "COMPUTER_TOOLSET_TYPE",
    "NATIVE_PROTOCOL_TRACK",
    "NOT_EXECUTED_ERROR",
    "AnthropicNativeApiError",
    "AnthropicComputerToolAction",
    "AnthropicNativeComputerBackend",
    "AnthropicNativeOutputError",
]
