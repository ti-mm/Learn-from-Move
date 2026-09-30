from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from ..actions import Action
from latentguiworld.suite import load_evaluation_manifest as load_manifest
from ..benchmarks.exploration_depth.service import RotationReplayServer
from ..custom_envs.catalog import EXPLORATION_DEPTH_VARIANTS, build_benchmark_variant
from ..models.openai_responses_six_action import (
    OPENAI_RESPONSES_SIX_ACTION_KINDS,
    SIX_ACTION_INSTRUCTIONS,
    six_action_arguments_to_action,
)

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_MAX_TURNS = 12
DEFAULT_LIMIT_PER_VARIANT = 10
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_RETRIES = 3
DEFAULT_VIEWPORT = (1280, 720)
VARIANTS = tuple(spec.key for spec in EXPLORATION_DEPTH_VARIANTS)

_write_lock = threading.Lock()
_print_lock = threading.Lock()


class CodexCliError(RuntimeError):
    """Codex CLI did not produce a usable model response."""


class CodexCliProtocolError(ValueError):
    """Codex output did not satisfy the isolated six-action contract."""


@dataclass(frozen=True)
class EvaluationTask:
    episode: dict[str, Any]

    @property
    def variant(self) -> str:
        return str(self.episode["variant"])

    @property
    def episode_id(self) -> str:
        return str(self.episode["episode_id"])


def _json_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _log(tag: str, message: str) -> None:
    with _print_lock:
        print(f"[{tag}] {message}", flush=True)


def _schema_payload() -> dict[str, Any]:
    nullable_coordinate = {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 1000,
    }
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
            },
            "x": dict(nullable_coordinate),
            "y": dict(nullable_coordinate),
            "points": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 1000,
                            },
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    {"type": "null"},
                ]
            },
        },
        # OpenAI structured outputs require every property to be required. Fields
        # not used by a particular action are emitted as null and removed below.
        "required": ["kind", "x", "y", "points"],
        "additionalProperties": False,
    }


def _canonical_arguments(raw: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    kind = raw.get("kind")
    if kind not in OPENAI_RESPONSES_SIX_ACTION_KINDS:
        raise CodexCliProtocolError(f"unsupported action kind: {kind!r}")
    x = raw.get("x")
    y = raw.get("y")
    points = raw.get("points")
    normalized_from: str | None = None

    if kind == "move_to":
        canonical = {"kind": kind, "x": x, "y": y}
    elif kind in {"mouse_down", "mouse_up", "left_click"}:
        if x is not None or y is not None or points is not None:
            raise CodexCliProtocolError(f"{kind} does not accept coordinates")
        canonical = {"kind": kind}
    elif kind == "click":
        if points is None and x is not None and y is not None:
            points = [[x, y]]
            normalized_from = "click_xy_to_single_point"
        canonical = {"kind": kind, "points": points}
    else:
        canonical = {"kind": kind, "points": points}

    return canonical, normalized_from


def _parse_events(stdout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", ""))
        if item_type and item_type not in {
            "agent_message",
            "reasoning",
            "analysis",
            "error",
        }:
            tool_events.append(
                {
                    "event_type": event.get("type"),
                    "item_type": item_type,
                    "item_id": item.get("id"),
                }
            )
    return events, tool_events


def _prompt(
    *,
    instruction: str,
    history: list[dict[str, Any]],
    turn_index: int,
    max_turns: int,
) -> str:
    history_text = (
        json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        if history
        else "[]"
    )
    return "\n\n".join(
        (
            SIX_ACTION_INSTRUCTIONS,
            "CLI adapter: no computer_use function is exposed by this process. Emit the "
            "one computer_use argument object directly as the schema-conforming final JSON. "
            "Use null for fields that do not apply: move_to uses x/y; click uses one point; "
            "drag uses exactly two points; mouse_down, mouse_up, and left_click use no "
            "coordinates. Do not call shell, MCP, web, file, or any other tool.",
            f"Task instruction: {instruction}",
            f"Current turn: {turn_index + 1} of {max_turns}.",
            "Previously executed canonical actions, oldest first: " + history_text,
            "The attached images are the latest observation history in oldest-to-newest "
            "order (at most three); the final image is the current observation. Return "
            "exactly one next action.",
        )
    )


class CodexCliSixActionBackend:
    def __init__(
        self,
        *,
        model: str,
        schema_path: Path,
        call_log_dir: Path,
        policy_workdir: Path,
        max_turns: int,
        timeout_s: float,
        retries: int,
        executable: str | None = None,
    ) -> None:
        self.model = model
        self.schema_path = schema_path.resolve()
        self.call_log_dir = call_log_dir.resolve()
        self.policy_workdir = policy_workdir.resolve()
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.retries = retries
        self.executable = executable or shutil.which("codex") or ""
        if not self.executable:
            raise FileNotFoundError("codex CLI is not available on PATH")
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        self.policy_workdir.mkdir(parents=True, exist_ok=True)
        self._call_index = 0
        self.last_raw_prediction: str | None = None
        self.last_native_action_json: dict[str, Any] | None = None
        self.last_call_log_path: str | None = None
        self.last_normalization: str | None = None

    def predict_action(
        self,
        *,
        screenshot_paths: list[str],
        instruction: str,
        action_history: list[dict[str, Any]],
    ) -> Action:
        if not screenshot_paths or len(screenshot_paths) > 3:
            raise ValueError("screenshot_paths must contain the latest 1 to 3 observations")
        resolved_screenshot_paths = [
            str(Path(path).resolve()) for path in screenshot_paths
        ]
        self._call_index += 1
        call_index = self._call_index
        prompt = _prompt(
            instruction=instruction,
            history=action_history,
            turn_index=call_index - 1,
            max_turns=self.max_turns,
        )
        output_path = self.call_log_dir / f"call_{call_index:03d}_last_message.json"
        attempts: list[dict[str, Any]] = []
        started_at = time.monotonic()
        final_error: BaseException | None = None

        for attempt_index in range(1, self.retries + 1):
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--disable",
                "shell_tool",
                "--disable",
                "apps",
                "--disable",
                "plugins",
                "--disable",
                "browser_use",
                "--disable",
                "computer_use",
                "--disable",
                "image_generation",
                "--disable",
                "workspace_dependencies",
                "-C",
                str(self.policy_workdir),
                "-s",
                "read-only",
                "-m",
                self.model,
                "--json",
                "--output-schema",
                str(self.schema_path),
                "-o",
                str(output_path),
                prompt,
                "-i",
                *resolved_screenshot_paths,
            ]
            attempt_started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.policy_workdir,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                final_error = error
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "latency_s": time.monotonic() - attempt_started,
                        "returncode": None,
                        "error": _json_error(error),
                    }
                )
                continue

            event_path = self.call_log_dir / (
                f"call_{call_index:03d}_attempt_{attempt_index:02d}_events.jsonl"
            )
            event_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path = self.call_log_dir / (
                f"call_{call_index:03d}_attempt_{attempt_index:02d}_stderr.txt"
            )
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            events, tool_events = _parse_events(completed.stdout)
            attempt = {
                "attempt": attempt_index,
                "latency_s": time.monotonic() - attempt_started,
                "returncode": completed.returncode,
                "event_path": str(event_path),
                "stderr_path": str(stderr_path),
                "event_types": [event.get("type") for event in events],
                "unexpected_tool_events": tool_events,
            }
            attempts.append(attempt)
            if tool_events:
                final_error = CodexCliProtocolError(
                    f"Codex used disallowed non-GUI tools: {tool_events!r}"
                )
                break
            if completed.returncode != 0:
                final_error = CodexCliError(
                    f"codex exited {completed.returncode}: {completed.stderr.strip() or completed.stdout[-1000:]}"
                )
                if attempt_index < self.retries:
                    time.sleep(min(float(attempt_index), 3.0))
                continue
            try:
                raw_text = output_path.read_text(encoding="utf-8").strip()
                raw = json.loads(raw_text)
                if not isinstance(raw, dict):
                    raise CodexCliProtocolError("structured action output is not an object")
                canonical, normalization = _canonical_arguments(raw)
                action = six_action_arguments_to_action(
                    canonical,
                    allowed_kinds=OPENAI_RESPONSES_SIX_ACTION_KINDS,
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                final_error = error
                break

            self.last_raw_prediction = raw_text
            self.last_native_action_json = canonical
            self.last_normalization = normalization
            call_log = {
                "call_index": call_index,
                "model": self.model,
                "transport": "codex_cli_ephemeral",
                "sandbox": "read-only",
                "user_config_loaded": False,
                "policy_workdir": str(self.policy_workdir),
                "screenshot_paths": resolved_screenshot_paths,
                "instruction": instruction,
                "action_history": action_history,
                "allowed_action_kinds": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
                "raw_action": raw,
                "canonical_action": canonical,
                "normalization": normalization,
                "attempts": attempts,
                "unexpected_tool_event_count": 0,
                "latency_s": time.monotonic() - started_at,
            }
            call_log_path = self.call_log_dir / f"call_{call_index:03d}.json"
            _write_json(call_log_path, call_log)
            self.last_call_log_path = str(call_log_path)
            return action

        error = final_error or CodexCliError("codex returned no usable action")
        call_log_path = self.call_log_dir / f"call_{call_index:03d}.json"
        _write_json(
            call_log_path,
            {
                "call_index": call_index,
                "model": self.model,
                "transport": "codex_cli_ephemeral",
                "sandbox": "read-only",
                "user_config_loaded": False,
                "policy_workdir": str(self.policy_workdir),
                "screenshot_paths": resolved_screenshot_paths,
                "instruction": instruction,
                "action_history": action_history,
                "allowed_action_kinds": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
                "attempts": attempts,
                "error": _json_error(error),
                "latency_s": time.monotonic() - started_at,
            },
        )
        self.last_call_log_path = str(call_log_path)
        raise error

def select_tasks(
    manifest: Mapping[str, Any],
    *,
    limit_per_variant: int,
) -> tuple[list[EvaluationTask], dict[str, list[str]]]:
    if limit_per_variant < 1:
        raise ValueError("limit_per_variant must be >= 1")
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("manifest episodes must be a list")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for episode in raw_episodes:
        if isinstance(episode, dict):
            by_family.setdefault(str(episode["family"]), []).append(episode)

    selected_pair_ids: dict[str, list[str]] = {}
    tasks: list[EvaluationTask] = []
    for family, episodes in sorted(by_family.items()):
        pair_ids = sorted({str(episode["pair_id"]) for episode in episodes})
        chosen = pair_ids[:limit_per_variant]
        if len(chosen) != limit_per_variant:
            raise ValueError(
                f"family {family!r} has {len(chosen)} pairs, expected {limit_per_variant}"
            )
        selected_pair_ids[family] = chosen
        chosen_set = set(chosen)
        selected = [
            episode for episode in episodes if str(episode["pair_id"]) in chosen_set
        ]
        variants = sorted({str(episode["variant"]) for episode in selected})
        for variant in variants:
            variant_episodes = sorted(
                (
                    episode
                    for episode in selected
                    if str(episode["variant"]) == variant
                ),
                key=lambda episode: str(episode["pair_id"]),
            )
            if len(variant_episodes) != limit_per_variant:
                raise ValueError(
                    f"variant {variant!r} selected {len(variant_episodes)} episodes, "
                    f"expected {limit_per_variant}"
                )
            tasks.extend(EvaluationTask(episode=episode) for episode in variant_episodes)

    selected_variants = {task.variant for task in tasks}
    if selected_variants != set(VARIANTS):
        raise ValueError(
            f"manifest variants {sorted(selected_variants)!r} do not match {sorted(VARIANTS)!r}"
        )
    expected = len(VARIANTS) * limit_per_variant
    if len(tasks) != expected:
        raise ValueError(f"selected {len(tasks)} tasks, expected {expected}")
    return sorted(tasks, key=lambda task: (task.variant, task.episode_id)), selected_pair_ids


def _validate_observation(screenshot_path: str, size_px: tuple[int, int]) -> None:
    if tuple(size_px) != DEFAULT_VIEWPORT:
        raise RuntimeError(f"unexpected observation viewport: {size_px!r}")
    with Image.open(screenshot_path) as image:
        if tuple(image.size) != DEFAULT_VIEWPORT or image.format != "PNG":
            raise RuntimeError(
                f"observation must be 1280x720 PNG, got {image.size!r} {image.format!r}"
            )


def _meaningful_delta(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(float(value)) > 1e-9
    if isinstance(value, Mapping):
        return any(_meaningful_delta(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_meaningful_delta(item) for item in value)
    return False


def _action_dict(action: Action) -> dict[str, Any]:
    return action.to_dict() if hasattr(action, "to_dict") else {"kind": action.kind}


def _case_dir(output_root: Path, task: EvaluationTask) -> Path:
    return output_root / "cases" / task.variant / task.episode_id


def _record_path(output_root: Path, task: EvaluationTask) -> Path:
    return output_root / "records" / task.variant / f"{task.episode_id}.json"


def run_task(
    task: EvaluationTask,
    *,
    manifest_path: Path,
    rotation_base_url: str,
    output_root: Path,
    schema_path: Path,
    model: str,
    max_turns: int,
    timeout_s: float,
    retries: int,
) -> dict[str, Any]:
    started_at = time.monotonic()
    episode = task.episode
    case_dir = _case_dir(output_root, task)
    kwargs: dict[str, Any] = {
        "manifest_path": manifest_path,
        "artifact_dir": case_dir / "frames",
    }
    if task.variant.startswith("rotation_"):
        kwargs["base_url"] = rotation_base_url
    environment = build_benchmark_variant(task.variant, **kwargs)
    backend = CodexCliSixActionBackend(
        model=model,
        schema_path=schema_path,
        call_log_dir=case_dir / "model_calls",
        policy_workdir=output_root / "policy_workdirs" / task.episode_id,
        max_turns=max_turns,
        timeout_s=timeout_s,
        retries=retries,
    )
    trace_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    action_history: list[dict[str, Any]] = []
    observation_history: list[str] = []
    terminal_reason = "max_turns"
    semantic_interactions = 0
    infra_error: dict[str, str] | None = None

    try:
        observation = environment.reset(task_id=task.episode_id)
        _validate_observation(observation.screenshot_path, observation.size_px)
        observation_history.append(observation.screenshot_path)
        get_audit = getattr(environment, "get_evaluator_audit", None)
        audit_rows.append(
            {
                "frame_index": 0,
                "semantic_interaction_index": 0,
                "evaluator_only": get_audit() if callable(get_audit) else {},
            }
        )
        for turn_index in range(max_turns):
            before_path = observation.screenshot_path
            try:
                action = backend.predict_action(
                    screenshot_paths=observation_history[-3:],
                    instruction=str(observation.instruction),
                    action_history=action_history,
                )
            except CodexCliError as error:
                terminal_reason = "infra_error"
                infra_error = _json_error(error)
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": None,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "predict_error": _json_error(error),
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break
            except Exception as error:
                terminal_reason = "predict_error"
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": None,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "predict_error": _json_error(error),
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break

            canonical_action = _action_dict(action)
            try:
                result = environment.step(action)
            except Exception as error:
                terminal_reason = "step_error"
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": canonical_action,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "step_error": _json_error(error),
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break

            state_delta = dict(result.info.get("state_delta", {}))
            if not result.done and _meaningful_delta(state_delta):
                semantic_interactions += 1
            success = bool(result.done and result.reward == 1.0)
            trace_rows.append(
                {
                    "frame_index": turn_index + 1,
                    "semantic_interaction_index": semantic_interactions,
                    "action": canonical_action,
                    "observation_before_path": before_path,
                    "observation_after_path": result.observation.screenshot_path,
                    "environment_state_delta": state_delta,
                    "terminal_success": success,
                    "interactions_to_success": semantic_interactions if success else None,
                    "corrective_interaction_count": max(semantic_interactions - 1, 0),
                    "model_raw_prediction": backend.last_raw_prediction,
                    "model_normalization": backend.last_normalization,
                    "model_call_log_path": backend.last_call_log_path,
                }
            )
            action_history.append(canonical_action)
            observation = result.observation
            observation_history.append(observation.screenshot_path)
            get_audit = getattr(environment, "get_evaluator_audit", None)
            audit_rows.append(
                {
                    "frame_index": turn_index + 1,
                    "semantic_interaction_index": semantic_interactions,
                    "evaluator_only": get_audit() if callable(get_audit) else {},
                }
            )
            if result.done:
                terminal_reason = "success" if success else "failure"
                break
    except Exception as error:
        terminal_reason = "infra_error"
        infra_error = _json_error(error)
        infra_error["traceback"] = traceback.format_exc()
    finally:
        environment.close()

    trace_path = case_dir / "trace.jsonl"
    audit_path = case_dir / "audit.jsonl"
    _write_jsonl(trace_path, trace_rows)
    _write_jsonl(audit_path, audit_rows)
    action_counts = Counter(
        str(row["action"]["kind"])
        for row in trace_rows
        if isinstance(row.get("action"), dict)
    )
    record = {
        "record_status": "complete",
        "suite_id": episode["suite_id"],
        "pair_id": episode["pair_id"],
        "family": episode["family"],
        "variant": task.variant,
        "exploration_level": episode["exploration_level"],
        "episode_id": task.episode_id,
        "case_seed": episode["case_seed"],
        "split": episode["split"],
        "model": model,
        "transport": "codex_cli_ephemeral",
        "success": terminal_reason == "success",
        "terminal_reason": terminal_reason,
        "max_turns": max_turns,
        "executed_turns": len(action_history),
        "vlm_calls": backend._call_index,
        "semantic_interactions": semantic_interactions,
        "interactions_to_success": (
            semantic_interactions if terminal_reason == "success" else None
        ),
        "corrective_interaction_count": max(semantic_interactions - 1, 0),
        "action_counts": dict(sorted(action_counts.items())),
        "allowed_action_kinds": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
        "history_strategy": "latest_3_observation_images_and_all_canonical_action_text",
        "unexpected_tool_event_count": sum(
            len(attempt.get("unexpected_tool_events", []))
            for log_path in sorted((case_dir / "model_calls").glob("call_*.json"))
            for attempt in json.loads(log_path.read_text(encoding="utf-8")).get(
                "attempts", []
            )
        ),
        "trace_path": str(trace_path),
        "audit_path": str(audit_path),
        "infra_error": infra_error,
        "latency_s": time.monotonic() - started_at,
    }
    _write_json(_record_path(output_root, task), record)
    return record


def load_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "records").glob("*/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("record_status") == "complete":
            records.append(record)
    return records


def summarize(
    records: list[dict[str, Any]],
    *,
    selected_counts: Mapping[str, int],
    output_root: Path,
    model: str,
    max_turns: int,
) -> dict[str, Any]:
    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [row for row in records if row.get("variant") == variant]
        successes = sum(row.get("success") is True for row in subset)
        turns = [int(row.get("executed_turns", 0)) for row in subset]
        per_variant[variant] = {
            "expected_count": int(selected_counts.get(variant, 0)),
            "completed_count": len(subset),
            "success_count": successes,
            "success_rate": successes / len(subset) if subset else 0.0,
            "mean_executed_turns": statistics.fmean(turns) if turns else None,
            "terminal_counts": dict(
                sorted(Counter(str(row.get("terminal_reason")) for row in subset).items())
            ),
            "unexpected_tool_event_count": sum(
                int(row.get("unexpected_tool_event_count", 0)) for row in subset
            ),
            "episode_ids": sorted(str(row["episode_id"]) for row in subset),
        }
    expected_total = sum(int(value) for value in selected_counts.values())
    return {
        "evaluation": "exploration_depth_6x10_codex_cli",
        "model": model,
        "transport": "codex_cli_ephemeral",
        "output_root": str(output_root),
        "allowed_action_kinds": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
        "max_turns": max_turns,
        "history_strategy": "latest_3_observation_images_and_all_canonical_action_text",
        "expected_count": expected_total,
        "completed_count": len(records),
        "missing_count": max(0, expected_total - len(records)),
        "success_count": sum(row.get("success") is True for row in records),
        "unexpected_tool_event_count": sum(
            int(row.get("unexpected_tool_event_count", 0)) for row in records
        ),
        "variants": per_variant,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all six exploration-depth variants with Codex CLI."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--limit-per-variant", type=int, default=DEFAULT_LIMIT_PER_VARIANT
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model != DEFAULT_MODEL:
        raise SystemExit(f"this evaluation requires --model {DEFAULT_MODEL}")
    if args.max_turns != DEFAULT_MAX_TURNS:
        raise SystemExit(f"this evaluation requires --max-turns {DEFAULT_MAX_TURNS}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = load_manifest(manifest_path)
    tasks, selected_pair_ids = select_tasks(
        manifest,
        limit_per_variant=args.limit_per_variant,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    schema_path = output_root / "six_action_output_schema.json"
    _write_json(schema_path, _schema_payload())
    selected_counts = Counter(task.variant for task in tasks)
    run_manifest = {
        "suite_manifest": str(manifest_path),
        "suite_id": manifest["suite_id"],
        "model": args.model,
        "transport": "codex_cli_ephemeral",
        "selected_pair_ids": selected_pair_ids,
        "selected_counts": dict(sorted(selected_counts.items())),
        "expected_count": len(tasks),
        "allowed_action_kinds": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
        "prompt": SIX_ACTION_INSTRUCTIONS,
        "prompt_adapter": (
            "schema-conforming final JSON stands in for one computer_use call"
        ),
        "history_strategy": "latest_3_observation_images_and_all_canonical_action_text",
        "policy_input_fields": [
            "instruction",
            "latest_3_observation_images",
            "action_history",
        ],
        "policy_workdir_contract": "empty isolated directory",
        "max_turns": args.max_turns,
        "workers": args.workers,
        "timeout_s": args.timeout_s,
        "retries": args.retries,
        "codex_executable": shutil.which("codex"),
        "openai_api_key_required": False,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _log("main", f"tasks={len(tasks)} workers={args.workers} output={output_root}")

    with RotationReplayServer(public_root=manifest_path.parent) as replay:
        def submit(task: EvaluationTask) -> dict[str, Any]:
            tag = f"{task.variant}/{task.episode_id}"
            record_path = _record_path(output_root, task)
            if record_path.is_file():
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                if existing.get("record_status") == "complete":
                    _log(tag, "skip")
                    return existing
            _log(tag, "start")
            record = run_task(
                task,
                manifest_path=manifest_path,
                rotation_base_url=replay.base_url,
                output_root=output_root,
                schema_path=schema_path,
                model=args.model,
                max_turns=args.max_turns,
                timeout_s=args.timeout_s,
                retries=args.retries,
            )
            _log(
                tag,
                f"{'PASS' if record['success'] else 'FAIL'} "
                f"terminal={record['terminal_reason']} turns={record['executed_turns']} "
                f"latency={record['latency_s']:.1f}s",
            )
            return record

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(submit, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                with _write_lock:
                    records = load_records(output_root)
                    summary = summarize(
                        records,
                        selected_counts=selected_counts,
                        output_root=output_root,
                        model=args.model,
                        max_turns=args.max_turns,
                    )
                    _write_json(output_root / "summary.json", summary)
                    _write_jsonl(
                        output_root / "records.jsonl",
                        sorted(
                            records,
                            key=lambda row: (str(row["variant"]), str(row["episode_id"])),
                        ),
                    )

    records = load_records(output_root)
    summary = summarize(
        records,
        selected_counts=selected_counts,
        output_root=output_root,
        model=args.model,
        max_turns=args.max_turns,
    )
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["missing_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
