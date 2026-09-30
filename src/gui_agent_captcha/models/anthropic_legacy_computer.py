from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from ..core import Observation
from .anthropic_native_computer import (
    ANTHROPIC_VERSION,
    COMPUTER_TOOLSET_NAME,
    AnthropicNativeComputerBackend,
    AnthropicNativeOutputError,
)

COMPUTER_BETA = "computer-use-2025-11-24"
COMPUTER_TOOL_TYPE = "computer_20251124"
NATIVE_PROTOCOL_TRACK = "anthropic_native_computer_20251124_noop_non_six"
COMPUTER_ACTION_KINDS = (
    "screenshot",
    "cursor_position",
    "left_click",
    "type",
    "key",
    "mouse_move",
    "scroll",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_mouse_down",
    "left_mouse_up",
    "hold_key",
    "wait",
    "zoom",
)


def computer_tool_declaration(size_px: tuple[int, int]) -> dict[str, Any]:
    width, height = size_px
    if width <= 0 or height <= 0:
        raise ValueError(f"observation size must be positive, got {size_px!r}")
    return {
        "type": COMPUTER_TOOL_TYPE,
        "name": COMPUTER_TOOLSET_NAME,
        "display_width_px": width,
        "display_height_px": height,
        "enable_zoom": True,
    }


def _legacy_wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wire_messages = copy.deepcopy(messages)
    for message in wire_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block.pop("toolset_name", None)
    return wire_messages


class AnthropicLegacyComputerBackend(AnthropicNativeComputerBackend):
    """Run the official beta ``computer_20251124`` wire contract.

    Member calls are normalized internally to the current executor shape so the
    environment boundary can keep only the six pointer-compatible mutations.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.protocol_track = NATIVE_PROTOCOL_TRACK

    def _request_parts(
        self,
        obs: Observation,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _legacy_wire_messages(self._messages),
            "tools": [computer_tool_declaration(obs.size_px)],
        }
        system = "\n\n".join(p for p in (self.system_prompt, self.prompt_extension) if p)
        if system:
            body["system"] = system
        return (
            f"{self.base_url}/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": COMPUTER_BETA,
                "Content-Type": "application/json",
            },
            body,
        )

    def _tool_uses(
        self,
        response: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        stop_reason = response.get("stop_reason")
        content = response.get("content")
        if not isinstance(content, list) or not all(
            isinstance(block, dict) for block in content
        ):
            raise AnthropicNativeOutputError(
                "Anthropic response content must be a block list"
            )
        raw_tool_uses = [block for block in content if block.get("type") == "tool_use"]
        if stop_reason == "end_turn":
            if raw_tool_uses:
                raise AnthropicNativeOutputError(
                    "Anthropic end_turn response must not contain a tool_use block"
                )
            return copy.deepcopy(content), None
        if stop_reason != "tool_use":
            raise AnthropicNativeOutputError(
                f"Anthropic response stop_reason is {stop_reason!r}, "
                "expected 'tool_use' or 'end_turn'"
            )
        if not raw_tool_uses:
            raise AnthropicNativeOutputError(
                "Anthropic tool_use response does not contain a computer action"
            )

        normalized: list[dict[str, Any]] = []
        for tool_use in raw_tool_uses:
            if tool_use.get("name") != COMPUTER_TOOLSET_NAME:
                raise AnthropicNativeOutputError(
                    f"unexpected legacy tool name {tool_use.get('name')!r}; "
                    f"expected {COMPUTER_TOOLSET_NAME!r}"
                )
            if not isinstance(tool_use.get("id"), str) or not tool_use["id"]:
                raise AnthropicNativeOutputError("computer tool_use is missing an id")
            native_input = tool_use.get("input")
            if not isinstance(native_input, dict):
                raise AnthropicNativeOutputError("computer tool input must be an object")
            member_name = native_input.get("action")
            if member_name not in COMPUTER_ACTION_KINDS:
                raise AnthropicNativeOutputError(
                    f"unexpected computer action {member_name!r}"
                )
            normalized.append(
                {
                    "type": "tool_use",
                    "id": tool_use["id"],
                    "name": member_name,
                    "toolset_name": COMPUTER_TOOLSET_NAME,
                    "input": {
                        str(key): copy.deepcopy(value)
                        for key, value in native_input.items()
                        if key != "action"
                    },
                }
            )
        return copy.deepcopy(content), normalized

    def _set_last_native_action(self, tool_use: Mapping[str, Any]) -> None:
        native_action = {
            "name": COMPUTER_TOOLSET_NAME,
            "input": {
                "action": tool_use["name"],
                **copy.deepcopy(dict(tool_use["input"])),
            },
        }
        self.last_native_action_json = native_action
        self.last_raw_prediction = json.dumps(
            native_action,
            ensure_ascii=False,
            separators=(",", ":"),
        )


__all__ = [
    "COMPUTER_BETA",
    "COMPUTER_ACTION_KINDS",
    "COMPUTER_TOOL_TYPE",
    "NATIVE_PROTOCOL_TRACK",
    "AnthropicLegacyComputerBackend",
    "computer_tool_declaration",
]
