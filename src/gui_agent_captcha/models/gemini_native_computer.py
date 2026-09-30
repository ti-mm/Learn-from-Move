from __future__ import annotations

import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..actions import Action, AtomicAction, PrimitiveAction, WebAction
from ..core import Observation, StepResult

JsonRequester = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]
ObservationSupplier = Callable[[], Observation]

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_REQUEST_RETRIES = 3
DEFAULT_MAX_INTERNAL_SCREENSHOTS = 8
GEMINI_NATIVE_PROTOCOL_TRACK = "gemini_native_computer_use_browser"

# Google documents these as the Gemini 3.x browser environment actions. The
# model always receives this native browser tool unchanged; this adapter only
# normalizes task-relevant calls after Gemini has selected an action.
BROWSER_NATIVE_ACTIONS = (
    "click",
    "double_click",
    "triple_click",
    "middle_click",
    "right_click",
    "mouse_down",
    "mouse_up",
    "move",
    "type",
    "drag_and_drop",
    "wait",
    "press_key",
    "key_down",
    "key_up",
    "hotkey",
    "take_screenshot",
    "scroll",
    "go_back",
    "navigate",
    "go_forward",
)


class GeminiNativeApiError(RuntimeError):
    """The Gemini Interactions request failed before returning a response."""


class GeminiNativeOutputError(ValueError):
    """Gemini returned a response that cannot be executed by this benchmark."""


class GeminiNativeSafetyDecisionError(GeminiNativeOutputError):
    """Gemini requires confirmation or blocked a native computer action."""


@dataclass(frozen=True)
class GeminiNativeComputerConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = field(default="GEMINI_API_KEY", repr=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Gemini API key must not be empty")
        if not self.model.strip():
            raise ValueError("Gemini model must not be empty")
        if not self.base_url.strip():
            raise ValueError("Gemini base_url must not be empty")

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> GeminiNativeComputerConfig:
        source = os.environ if env is None else env
        api_key = str(source.get(api_key_env, "")).strip()
        if not api_key:
            raise ValueError(f"{api_key_env} is required for {model}")
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env,
        )


@dataclass(frozen=True)
class _NativeFunctionCall:
    interaction_id: str
    call_id: str
    name: str
    arguments: dict[str, Any]
    native_step: dict[str, Any]


@dataclass(frozen=True)
class _QueuedAction:
    action: Action
    call: _NativeFunctionCall


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
        raise RuntimeError(f"HTTP {exc.code} from Gemini Interactions API: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini Interactions API returned a non-object JSON response")
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
            str(key): _sanitize_for_log(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if isinstance(value, str) and field_name == "data":
        return f"<redacted-image-data length={len(value)}>"
    if isinstance(value, str) and field_name and any(
        marker in field_name.lower() for marker in ("api_key", "authorization")
    ):
        return "<redacted>"
    return value


def _sanitize_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")


def _image_part(obs: Observation) -> dict[str, str]:
    path = Path(obs.screenshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"observation screenshot does not exist: {path}")
    return {
        "type": "image",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "mime_type": _mime_type(path),
    }


def _episode_key(obs: Observation) -> tuple[str, str] | None:
    metadata = obs.metadata or {}
    for name in ("episode_id", "task_id"):
        value = metadata.get(name)
        if value is not None and str(value).strip():
            return name, str(value)
    return None


def _native_coordinate(arguments: Mapping[str, Any], key: str) -> float:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeminiNativeOutputError(f"{key} must be a numeric Gemini coordinate")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 999.0:
        raise GeminiNativeOutputError(f"{key} must be in Gemini's inclusive range [0,999]")
    # Gemini's official executor maps coordinates with ``value / 1000 * extent``.
    # The benchmark uses that same 0..1000 denominator, so no endpoint stretching
    # is needed even though Gemini currently emits integer values in 0..999.
    return number


def _point(arguments: Mapping[str, Any], x_key: str, y_key: str) -> tuple[float, float]:
    return _native_coordinate(arguments, x_key), _native_coordinate(arguments, y_key)


def _can_emit_native_action(name: str, allowed: set[str]) -> bool:
    primitives = {"move_to", "mouse_down", "mouse_up"}.issubset(allowed)
    if name == "move":
        return "move_to" in allowed
    if name == "click":
        return "click" in allowed or primitives
    if name == "drag_and_drop":
        return "drag" in allowed or primitives
    if name in {"mouse_down", "mouse_up"}:
        return "move_to" in allowed and name in allowed
    if name == "wait":
        return "wait" in allowed
    return name == "take_screenshot"


def _computer_use_tool(allowed_kinds: Iterable[str]) -> dict[str, Any]:
    allowed = {str(kind) for kind in allowed_kinds}
    executable = {
        name
        for name in (
            "move",
            "click",
            "drag_and_drop",
            "mouse_down",
            "mouse_up",
            "wait",
            "take_screenshot",
        )
        if _can_emit_native_action(name, allowed)
    }
    if executable == {"take_screenshot"}:
        raise ValueError("allowed_kinds contains no executable Gemini Computer Use action")
    return {"type": "computer_use", "environment": "browser"}


def _actions_from_native_call(
    call: _NativeFunctionCall,
    *,
    allowed_kinds: Iterable[str],
) -> list[Action]:
    allowed = {str(kind) for kind in allowed_kinds}
    arguments = call.arguments
    if call.name == "take_screenshot":
        return []
    if call.name == "wait":
        if "wait" not in allowed:
            raise GeminiNativeOutputError("Gemini wait is unavailable in this action condition")
        seconds = arguments.get("seconds", 1)
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or seconds < 0
        ):
            raise GeminiNativeOutputError("Gemini wait seconds must be a finite non-negative number")
        return [WebAction(kind="wait", duration_s=float(seconds))]
    if call.name == "move":
        if "move_to" not in allowed:
            raise GeminiNativeOutputError("Gemini move is unavailable in this action condition")
        x_value, y_value = _point(arguments, "x", "y")
        return [PrimitiveAction(kind="move_to", x=x_value, y=y_value)]
    if call.name == "click":
        point = _point(arguments, "x", "y")
        if "click" in allowed:
            return [AtomicAction(kind="click", points=[point])]
        if {"move_to", "mouse_down", "mouse_up"}.issubset(allowed):
            return [
                PrimitiveAction(kind="move_to", x=point[0], y=point[1]),
                PrimitiveAction(kind="mouse_down"),
                PrimitiveAction(kind="mouse_up"),
            ]
        raise GeminiNativeOutputError("Gemini click cannot be represented by allowed_kinds")
    if call.name == "drag_and_drop":
        start = _point(arguments, "start_x", "start_y")
        end = _point(arguments, "end_x", "end_y")
        if "drag" in allowed:
            return [AtomicAction(kind="drag", points=[start, end])]
        if {"move_to", "mouse_down", "mouse_up"}.issubset(allowed):
            return [
                PrimitiveAction(kind="move_to", x=start[0], y=start[1]),
                PrimitiveAction(kind="mouse_down"),
                PrimitiveAction(kind="move_to", x=end[0], y=end[1]),
                PrimitiveAction(kind="mouse_up"),
            ]
        raise GeminiNativeOutputError(
            "Gemini drag_and_drop cannot be represented by allowed_kinds"
        )
    if call.name in {"mouse_down", "mouse_up"}:
        if "move_to" not in allowed or call.name not in allowed:
            raise GeminiNativeOutputError(
                f"Gemini {call.name} requires move_to and {call.name} in allowed_kinds"
            )
        x_value, y_value = _point(arguments, "x", "y")
        return [
            PrimitiveAction(kind="move_to", x=x_value, y=y_value),
            PrimitiveAction(kind=call.name),
        ]
    raise GeminiNativeOutputError(f"unsupported Gemini browser action: {call.name!r}")


def _validate_safety_decision(call: _NativeFunctionCall) -> None:
    value = call.arguments.get("safety_decision")
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise GeminiNativeOutputError("Gemini safety_decision must be an object")
    decision = value.get("decision")
    if not isinstance(decision, str) or not decision:
        raise GeminiNativeOutputError(
            "Gemini safety_decision must contain a non-empty decision"
        )
    normalized = decision.strip().lower()
    if normalized in {"regular", "allowed"}:
        return
    explanation = value.get("explanation")
    detail = f": {explanation}" if isinstance(explanation, str) and explanation else ""
    if normalized == "require_confirmation":
        raise GeminiNativeSafetyDecisionError(
            "Gemini native action requires user confirmation; automated benchmark "
            f"execution is disabled{detail}"
        )
    if normalized == "blocked":
        raise GeminiNativeSafetyDecisionError(
            f"Gemini native action was blocked by the provider safety decision{detail}"
        )
    raise GeminiNativeSafetyDecisionError(
        f"unsupported Gemini safety decision {decision!r}; action was not executed"
    )


def _parse_interaction(
    response: Mapping[str, Any],
) -> tuple[str, list[_NativeFunctionCall]]:
    interaction_id = response.get("id")
    if not isinstance(interaction_id, str) or not interaction_id:
        raise GeminiNativeOutputError("Gemini interaction response is missing its id")
    steps = response.get("steps")
    if not isinstance(steps, list):
        raise GeminiNativeOutputError("Gemini interaction response must contain steps")
    raw_calls = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("type") == "function_call"
    ]
    calls: list[_NativeFunctionCall] = []
    seen_call_ids: set[str] = set()
    for step in raw_calls:
        name = step.get("name")
        call_id = step.get("id", step.get("call_id"))
        arguments = step.get("arguments")
        if not isinstance(name, str) or not name:
            raise GeminiNativeOutputError("Gemini function_call is missing its name")
        if not isinstance(call_id, str) or not call_id:
            raise GeminiNativeOutputError("Gemini function_call is missing its call id")
        if call_id in seen_call_ids:
            raise GeminiNativeOutputError(f"duplicate Gemini function call id {call_id!r}")
        if not isinstance(arguments, dict):
            raise GeminiNativeOutputError("Gemini function_call arguments must be an object")
        seen_call_ids.add(call_id)
        calls.append(
            _NativeFunctionCall(
                interaction_id=interaction_id,
                call_id=call_id,
                name=name,
                arguments=deepcopy(arguments),
                native_step=deepcopy(step),
            )
        )
    return interaction_id, calls


class GeminiNativeComputerUseBackend:
    """Gemini 3.6 Flash's native browser Computer Use interaction loop."""

    def __init__(
        self,
        *,
        config: GeminiNativeComputerConfig,
        call_log_dir: Path | None = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
        max_internal_screenshots: int = DEFAULT_MAX_INTERNAL_SCREENSHOTS,
        observation_supplier: ObservationSupplier | None = None,
        requester: JsonRequester | None = None,
    ) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if request_retries < 1:
            raise ValueError("request_retries must be >= 1")
        if max_internal_screenshots < 1:
            raise ValueError("max_internal_screenshots must be >= 1")
        self.config = config
        self.model = config.model
        self.call_log_dir = call_log_dir
        self.request_timeout_s = request_timeout_s
        self.request_retries = request_retries
        self.max_internal_screenshots = max_internal_screenshots
        self.observation_supplier = observation_supplier
        self.requester = requester or _default_json_requester
        self.sampling_controls_unsupported = True
        self.protocol_track = GEMINI_NATIVE_PROTOCOL_TRACK
        self.history_strategy = "previous_interaction_id_full_server_history"

        self._call_index = 0
        self._prediction_index = 0
        self._previous_interaction_id: str | None = None
        self._pending_calls: list[_NativeFunctionCall] = []
        self._pending_actions: list[_QueuedAction] = []
        self._episode_key: tuple[str, str] | None = None
        self._last_history_length = 0

        self.last_prediction_index: int | None = None
        self.last_raw_prediction: str | None = None
        self.last_think_text: str | None = None
        self.last_intent: str | None = None
        self.last_native_call: dict[str, Any] | None = None
        self.last_native_response: dict[str, Any] | None = None
        self.native_call_history: list[dict[str, Any]] = []
        self.last_call_log_path: str | None = None
        self.last_error: str | None = None

    @property
    def previous_interaction_id(self) -> str | None:
        return self._previous_interaction_id

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "protocol_track": GEMINI_NATIVE_PROTOCOL_TRACK,
            "native_tool": "computer_use",
            "environment": "browser",
            "coordinate_format": "gemini_0_999_official_1000_denominator",
            "history_strategy": self.history_strategy,
            "native_action_kinds": list(BROWSER_NATIVE_ACTIONS),
        }

    def reset_episode(self) -> None:
        self._previous_interaction_id = None
        self._pending_calls.clear()
        self._pending_actions.clear()
        self._episode_key = None
        self._last_history_length = 0
        self._prediction_index = 0
        self.last_prediction_index = None
        self.last_raw_prediction = None
        self.last_think_text = None
        self.last_intent = None
        self.last_native_call = None
        self.last_native_response = None
        self.native_call_history.clear()
        self.last_error = None

    def reset(self) -> None:
        self.reset_episode()

    def _maybe_reset_episode(self, obs: Observation, history: list[StepResult]) -> None:
        key = _episode_key(obs)
        changed_key = key is not None and self._episode_key is not None and key != self._episode_key
        history_restarted = not history and self._last_history_length > 0
        if changed_key or history_restarted:
            self.reset_episode()
        if key is not None:
            self._episode_key = key

    def _safe_error_text(self, exc: BaseException) -> str:
        text = str(exc).replace(self.config.api_key, "<redacted-api-key>")
        return re.sub(
            r'("data"\s*:\s*")[^"]+("\s*[,}])',
            r"\1<redacted-image-data>\2",
            text,
        )

    def _fresh_observation_for_screenshot(self, prior: Observation) -> Observation:
        if self.observation_supplier is None:
            raise GeminiNativeOutputError(
                "native take_screenshot requires an observation_supplier that captures a fresh screenshot"
            )
        fresh = self.observation_supplier()
        if not isinstance(fresh, Observation):
            raise GeminiNativeOutputError(
                "observation_supplier must return an Observation after native take_screenshot"
            )
        if fresh.instruction != prior.instruction:
            raise GeminiNativeOutputError(
                "observation_supplier returned an observation from a different task"
            )
        return fresh

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

    def _request(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self._call_index += 1
        url = f"{self.config.base_url.rstrip('/')}/interactions"
        headers = {
            "x-goog-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }
        attempts: list[dict[str, Any]] = []
        response: Mapping[str, Any] | None = None
        last_error: BaseException | None = None
        started_at = time.time()
        for attempt in range(1, self.request_retries + 1):
            attempt_started = time.time()
            try:
                candidate = self.requester(url, headers, body, self.request_timeout_s)
                if not isinstance(candidate, Mapping):
                    raise TypeError("Gemini requester returned a non-mapping response")
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
            cause = last_error or RuntimeError("Gemini returned no response")
            self.last_error = f"{type(cause).__name__}: {self._safe_error_text(cause)}"
            self._write_call_log(
                {
                    "provider": "gemini",
                    "model": self.model,
                    "base_url": _sanitize_base_url(self.config.base_url),
                    "request": body,
                    "request_attempts": attempts,
                    "error": self.last_error,
                    "api_key_persisted": False,
                    "latency_s": time.time() - started_at,
                }
            )
            raise GeminiNativeApiError(
                f"Gemini returned no response after {len(attempts)} attempt(s)"
            ) from last_error
        self._write_call_log(
            {
                "provider": "gemini",
                "model": self.model,
                "base_url": _sanitize_base_url(self.config.base_url),
                "history_strategy": "previous_interaction_id_full_server_history",
                "request": body,
                "response": response,
                "request_attempts": attempts,
                "api_key_persisted": False,
                "latency_s": time.time() - started_at,
            }
        )
        return response

    def _initial_request(
        self,
        obs: Observation,
        *,
        tool: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {"type": "text", "text": obs.instruction},
                _image_part(obs),
            ],
            "tools": [dict(tool)],
        }

    def _function_result(
        self,
        call: _NativeFunctionCall,
        obs: Observation,
    ) -> dict[str, Any]:
        result: dict[str, str] = {"status": "success"}
        for key in ("url", "page_url", "current_url"):
            value = (obs.metadata or {}).get(key)
            if isinstance(value, str) and value:
                result["url"] = value
                break
        return {
            "type": "function_result",
            "name": call.name,
            "call_id": call.call_id,
            "result": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                },
                _image_part(obs),
            ],
        }

    def _continuation_request(
        self,
        calls: list[_NativeFunctionCall],
        obs: Observation,
        *,
        tool: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not calls:
            raise GeminiNativeOutputError(
                "cannot continue Gemini interaction without pending function calls"
            )
        interaction_ids = {call.interaction_id for call in calls}
        if len(interaction_ids) != 1:
            raise GeminiNativeOutputError(
                "pending Gemini function calls span multiple interactions"
            )
        return {
            "model": self.model,
            "previous_interaction_id": calls[0].interaction_id,
            "input": [self._function_result(call, obs) for call in calls],
            "tools": [dict(tool)],
        }

    def _remember_response(
        self,
        response: Mapping[str, Any],
        calls: list[_NativeFunctionCall],
    ) -> None:
        self.last_native_response = deepcopy(dict(response))
        if not calls:
            self.last_native_call = None
            self.last_raw_prediction = json.dumps(
                dict(response), ensure_ascii=False, separators=(",", ":")
            )
            self.last_think_text = None
            self.last_intent = None
            return
        call = calls[0]
        self.last_native_call = deepcopy(call.native_step)
        self.native_call_history.extend(deepcopy(item.native_step) for item in calls)
        self.last_raw_prediction = json.dumps(
            call.native_step, ensure_ascii=False, separators=(",", ":")
        )
        intent = call.arguments.get("intent")
        self.last_intent = str(intent) if isinstance(intent, str) else None
        self.last_think_text = self.last_intent

    def _pop_pending_action(self) -> Action:
        queued = self._pending_actions.pop(0)
        call = queued.call
        self.last_native_call = deepcopy(call.native_step)
        self.last_raw_prediction = json.dumps(
            call.native_step, ensure_ascii=False, separators=(",", ":")
        )
        intent = call.arguments.get("intent")
        self.last_intent = str(intent) if isinstance(intent, str) else None
        self.last_think_text = self.last_intent
        self._prediction_index += 1
        self.last_prediction_index = self._prediction_index
        return queued.action

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
        tool = _computer_use_tool(allowed)
        self._maybe_reset_episode(obs, history)
        self.last_error = None

        if self._pending_actions:
            self._last_history_length = len(history)
            return self._pop_pending_action()

        internal_screenshots = 0
        current_obs = obs
        while True:
            if not self._pending_calls:
                if self._previous_interaction_id is not None:
                    raise GeminiNativeOutputError(
                        "Gemini interaction has no pending function call; reset the episode"
                    )
                body = self._initial_request(current_obs, tool=tool)
            else:
                if any(call.name == "take_screenshot" for call in self._pending_calls):
                    current_obs = self._fresh_observation_for_screenshot(current_obs)
                body = self._continuation_request(self._pending_calls, current_obs, tool=tool)

            response = self._request(body)
            self._pending_calls.clear()
            interaction_id, calls = _parse_interaction(response)
            self._previous_interaction_id = interaction_id
            self._remember_response(response, calls)

            if not calls:
                self._last_history_length = len(history)
                if "done" in allowed:
                    self._prediction_index += 1
                    self.last_prediction_index = self._prediction_index
                    return PrimitiveAction(kind="done")
                raise GeminiNativeOutputError(
                    "Gemini returned model output without a Computer Use function_call"
                )

            for call in calls:
                _validate_safety_decision(call)
            queued_actions = [
                _QueuedAction(action=action, call=call)
                for call in calls
                for action in _actions_from_native_call(call, allowed_kinds=allowed)
            ]
            self._pending_calls = calls
            if not queued_actions:
                internal_screenshots += 1
                if internal_screenshots > self.max_internal_screenshots:
                    raise GeminiNativeOutputError(
                        "Gemini exceeded the internal take_screenshot limit"
                    )
                continue

            self._pending_actions = queued_actions
            self._last_history_length = len(history)
            return self._pop_pending_action()
