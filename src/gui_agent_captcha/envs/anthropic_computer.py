from __future__ import annotations

import copy
from typing import Any

from ..actions import AtomicAction, PrimitiveAction
from ..core import Observation, StepResult
from ..models.anthropic_native_computer import (
    COMPUTER_TOOLSET_MEMBERS,
    AnthropicComputerToolAction,
)

SIX_ACTION_ENVIRONMENT_MEMBERS = (
    "mouse_move",
    "left_mouse_down",
    "left_mouse_up",
    "left_click",
    "left_click_drag",
)


def _normalized_coordinate(
    coordinate: Any,
    *,
    observation: Observation,
    label: str,
) -> tuple[float, float]:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        raise ValueError(f"{label} must be [x, y]")
    width, height = observation.size_px
    if width <= 0 or height <= 0:
        raise ValueError(f"observation size must be positive, got {observation.size_px!r}")
    x, y = float(coordinate[0]), float(coordinate[1])
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"{label} {coordinate!r} is outside screenshot {width}x{height}")
    return x * 1000.0 / width, y * 1000.0 / height


def _coordinate_or_cursor(
    action: AnthropicComputerToolAction,
    observation: Observation,
) -> tuple[float, float]:
    coordinate = action.input.get("coordinate")
    if coordinate is None:
        if observation.cursor_xy is None:
            raise ValueError(f"{action.name} omitted coordinate but cursor position is unavailable")
        coordinate = observation.cursor_xy
    return _normalized_coordinate(
        coordinate,
        observation=observation,
        label="coordinate",
    )


def _no_environment_effect_result(
    action: AnthropicComputerToolAction,
    observation: Observation,
) -> StepResult:
    return StepResult(
        observation=observation,
        reward=None,
        done=False,
        info={
            "executed_kind": action.name,
            "native_toolset": action.toolset_name,
            "native_input": copy.deepcopy(action.input),
            "environment_effect": "none",
            "environment_actions": [],
            "state_delta": {},
        },
        action=action,  # type: ignore[arg-type]
    )


def _environment_action(
    action: AnthropicComputerToolAction,
    observation: Observation,
) -> PrimitiveAction | AtomicAction | None:
    name = action.name
    if name == "mouse_move":
        x, y = _normalized_coordinate(
            action.input["coordinate"],
            observation=observation,
            label="coordinate",
        )
        return PrimitiveAction(kind="move_to", x=x, y=y)
    if name == "left_mouse_down":
        return PrimitiveAction(kind="mouse_down")
    if name == "left_mouse_up":
        return PrimitiveAction(kind="mouse_up")
    if name == "left_click":
        if "coordinate" not in action.input:
            return PrimitiveAction(kind="left_click")
        return AtomicAction(
            kind="click",
            points=[_coordinate_or_cursor(action, observation)],
        )
    if name == "left_click_drag":
        return AtomicAction(
            kind="drag",
            points=[
                _normalized_coordinate(
                    action.input["start_coordinate"],
                    observation=observation,
                    label="start_coordinate",
                ),
                _normalized_coordinate(
                    action.input["coordinate"],
                    observation=observation,
                    label="coordinate",
                ),
            ],
        )
    return None


def execute_anthropic_computer_action(
    environment: object,
    action: AnthropicComputerToolAction,
    observation: Observation,
) -> StepResult:
    """Execute an official member call without exposing the repo's six-action contract."""

    if action.toolset_name != "computer":
        raise ValueError(f"unexpected toolset_name {action.toolset_name!r}")
    if action.name not in COMPUTER_TOOLSET_MEMBERS:
        raise ValueError(f"unexpected computer member {action.name!r}")

    if action.name not in SIX_ACTION_ENVIRONMENT_MEMBERS:
        return _no_environment_effect_result(action, observation)
    internal_action = _environment_action(action, observation)
    if internal_action is None:
        raise RuntimeError(f"missing six-action adaptation for {action.name!r}")

    step = getattr(environment, "step", None)
    if not callable(step):
        raise RuntimeError("environment does not expose step()")
    result = step(internal_action)
    if not isinstance(result, StepResult):
        raise RuntimeError("environment step() returned an invalid value")
    info = dict(result.info)
    info.update(
        {
            "executed_kind": action.name,
            "native_toolset": action.toolset_name,
            "native_input": copy.deepcopy(action.input),
            "environment_effect": "six_action_adapter",
            "environment_actions": [internal_action.to_dict()],
        }
    )
    result.info = info
    result.action = action  # type: ignore[assignment]
    return result


__all__ = ["SIX_ACTION_ENVIRONMENT_MEMBERS", "execute_anthropic_computer_action"]
