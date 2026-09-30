"""Execute native local-agent decisions in the isolated exploration canvases."""

from __future__ import annotations

import math
import time
from typing import Callable

from ..actions import AtomicAction, PrimitiveAction, WebAction
from ..benchmarks.exploration_depth.runner import ExplorationTraceSession
from ..core import Observation, StepResult
from ..models.official_gui_actions import (
    OfficialGuiAction,
    OfficialGuiProtocolError,
    environment_actions,
)


def execute_official_gui_action(
    session: ExplorationTraceSession,
    action: OfficialGuiAction,
    observation: Observation,
    execute_mouse: Callable[[PrimitiveAction | AtomicAction], StepResult],
) -> StepResult:
    outcomes = []
    result = None
    for call in action.calls:
        try:
            commands, effect = environment_actions(call, action, observation)
        except (ValueError, TypeError, KeyError) as exc:
            raise OfficialGuiProtocolError(str(exc)) from exc
        executed = []
        if commands:
            for command in commands:
                result = execute_mouse(command)
                observation = result.observation
                executed.append(command.to_dict())
                get_audit = getattr(session.env, "get_evaluator_audit", None)
                audit = get_audit() if callable(get_audit) else {}
                rotation = audit.get("rotation_state")
                attempt = rotation.get("attempt") if isinstance(rotation, dict) else None
                if isinstance(attempt, dict):
                    # Rotation's replay environment permits correction after a
                    # failed release. The evaluation stops at its first actual
                    # attempt, including when raw result.done is still False.
                    result.done = True
                    result.reward = float(attempt.get("success") is True)
                    result.info["strict_first_release"] = True
                # Never execute a later call after the environment's first terminal release.
                if result.done:
                    break
        else:
            if effect == "wait":
                seconds = call.arguments.get("time", call.arguments.get("seconds", 5.0))
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds)
                    or seconds < 0
                ):
                    raise ValueError("wait duration must be finite and non-negative")
                time.sleep(seconds)
                # All six environments implement wait as a fresh observation.
                # Sleep above also covers the first-person canvas whose wait is
                # observation-only. Zero here avoids sleeping twice elsewhere.
                result = session.step(WebAction("wait", duration_s=0.0))
                observation = result.observation
            else:
                result = StepResult(
                    observation=observation,
                    reward=None,
                    done=effect in {"model_terminated", "model_requested_user"},
                    info={"state_delta": {}, "environment_effect": effect},
                    action=action,
                )
                # Legal no-effect actions still consume a decision and appear in the trace.
                session._record(observation=observation, result=result, action=action)
        outcomes.append({"name": call.name, "effect": effect, "executed_actions": executed})
        if result.done:
            break
    assert result is not None
    result.info["native_call_results"] = outcomes
    result.info["native_action"] = action.to_dict()
    result.action = action
    return result
