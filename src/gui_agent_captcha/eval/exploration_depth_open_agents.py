from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..actions import Action, AtomicAction, PrimitiveAction
from ..benchmarks.exploration_depth.contracts import EXPLORATION_BENCHMARK_VARIANTS, FORMAL_SUITE_ID
from latentguiworld.suite import load_evaluation_manifest as load_manifest
from ..benchmarks.exploration_depth.runner import ExplorationTraceSession
from ..benchmarks.exploration_depth.service import RotationReplayServer
from ..benchmarks.exploration_depth.training_no_think import SIX_ACTION_KINDS
from ..core import Observation, StepResult
from ..custom_envs.catalog import build_benchmark_variant
from ..envs.browser_base import BrowserBenchmarkEnv
from ..envs.official_gui import execute_official_gui_action
from ..models.baseline_adapters import BaselinePromptStyle
from ..models.local_baseline import BaselineHistoryMode, LocalBaselineVisionBackend
from ..models.official_gui import (
    GENERATION_LIMITS,
    HISTORY_STRATEGIES,
    OFFICIAL_GUI_CONTRACT,
    OfficialGuiBackend,
    model_action_names,
    protocol_metadata,
)
from ..models.official_gui_actions import OfficialGuiAction, OfficialGuiProtocolError
from ..policies.move_observe_act import ClosedLoopPrimitivePolicy

DEFAULT_MAX_TURNS = 12
DEFAULT_HISTORY_IMAGE_MAX = 5
PUBLIC_CHECKPOINT_ROOT = Path(
    str(Path(__file__).resolve().parents[3] / 'models/')
)
VARIANTS = tuple(contract.key for contract in EXPLORATION_BENCHMARK_VARIANTS)
FIRST_PERSON_VARIANTS = frozenset({"ten_choice_first_person", "drag_first_person"})
ERROR_TERMINAL_REASONS = frozenset({"infra_error", "predict_error", "step_error"})
ACTION_EXECUTION_CONTRACT = "first_person_sensitivity_aware_click_drag_v1"
PROMPT_CONTRACT = OFFICIAL_GUI_CONTRACT
FIRST_PERSON_ACTION_SEQUENCES = {
    "click": ["sensitivity_move_to", "left_click_center"],
    "drag": [
        "sensitivity_move_to_start",
        "mouse_down_center",
        "sensitivity_drag_path",
        "mouse_up",
    ],
}


@dataclass(frozen=True)
class OpenGuiAgentSpec:
    key: str
    label: str
    prompt_style: BaselinePromptStyle
    checkpoint_name: str
    history_mode: BaselineHistoryMode
    model_coordinate_schema: str
    max_new_tokens: int
    button_extension: bool = False
    button_prompt_reminder: bool = False

    @property
    def prompt_contract(self) -> str:
        return protocol_metadata(
            self.prompt_style, button_extension=self.button_extension,
            button_prompt_reminder=self.button_prompt_reminder
        )["contract"]

    def checkpoint(self, checkpoint_root: Path = PUBLIC_CHECKPOINT_ROOT) -> Path:
        return checkpoint_root / self.checkpoint_name


OPEN_GUI_AGENT_SPECS: tuple[OpenGuiAgentSpec, ...] = (
    OpenGuiAgentSpec(
        "dart_gui",
        "DART-GUI",
        "dart_gui",
        "dart-gui-7b",
        "dart_osworld",
        "qwen25_smart_resize_absolute_pixels",
        500,
    ),
    OpenGuiAgentSpec(
        "uitars",
        "UI-TARS",
        "uitars",
        "UI-TARS-1.5-7B",
        "uitars_osworld",
        "qwen25_smart_resize_absolute_pixels",
        1000,
    ),
    OpenGuiAgentSpec(
        "gui_owl",
        "GUI-Owl",
        "gui_owl",
        "GUI-Owl-1.5-32B-Think",
        "gui_owl15",
        "normalized_0_1000_absolute_position",
        32768,
    ),
    OpenGuiAgentSpec(
        "evocua",
        "EvoCUA",
        "evocua_s2",
        "EvoCUA-32B-20260105",
        "evocua_s2",
        "normalized_0_999_absolute_position",
        32768,
    ),
    OpenGuiAgentSpec(
        "holo31",
        "Holo3.1",
        "holo3_tool",
        "Holo-3.1-35B-A3B",
        "holo3_tool",
        "normalized_0_1000_absolute_position",
        8192,
    ),
    OpenGuiAgentSpec(
        "opencua",
        "OpenCUA",
        "opencua_pyautogui",
        "OpenCUA-72B",
        "opencua_l2",
        "qwen25_smart_resize_absolute_pixels_and_relative_deltas",
        2048,
    ),
)
OPEN_GUI_AGENT_BY_KEY = {spec.key: spec for spec in OPEN_GUI_AGENT_SPECS}


class OpenGuiAgentEvalError(RuntimeError):
    """A six-agent exploration-depth evaluation contract was violated."""


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


def validate_suite(manifest: Mapping[str, Any]) -> dict[str, int]:
    if manifest.get("schema") != "gui_captcha_exploration_depth_manifest_v1":
        raise OpenGuiAgentEvalError("unsupported exploration-depth manifest schema")
    counts = Counter(str(episode.get("variant")) for episode in manifest.get("episodes", []))
    expected = set(VARIANTS)
    if set(counts) != expected:
        raise OpenGuiAgentEvalError(f"six-environment manifest variants mismatch: {sorted(counts)}")
    if any(count < 1 for count in counts.values()):
        raise OpenGuiAgentEvalError("every exploration-depth variant must be non-empty")
    if manifest.get("suite_id") == FORMAL_SUITE_ID:
        if manifest.get("frozen") is not True or set(counts.values()) != {150}:
            raise OpenGuiAgentEvalError(
                f"formal suite {FORMAL_SUITE_ID} must be frozen with 150 cases per variant"
            )
    return {variant: counts[variant] for variant in VARIANTS}


def select_episodes(
    manifest: Mapping[str, Any],
    *,
    variants: Iterable[str] = VARIANTS,
    limit_per_variant: int | None = None,
    episode_shard_index: int = 0,
    episode_shard_count: int = 1,
) -> list[dict[str, Any]]:
    validate_suite(manifest)
    selected_variants = tuple(dict.fromkeys(str(value) for value in variants))
    unknown = sorted(set(selected_variants) - set(VARIANTS))
    if unknown:
        raise OpenGuiAgentEvalError(f"unknown exploration variants: {unknown}")
    if not selected_variants:
        raise OpenGuiAgentEvalError("at least one exploration variant is required")
    if limit_per_variant is not None and limit_per_variant < 1:
        raise OpenGuiAgentEvalError("limit_per_variant must be positive")
    if episode_shard_count < 1:
        raise OpenGuiAgentEvalError("episode_shard_count must be positive")
    if not 0 <= episode_shard_index < episode_shard_count:
        raise OpenGuiAgentEvalError(
            "episode_shard_index must be in [0, episode_shard_count)"
        )

    selected: list[dict[str, Any]] = []
    for variant in selected_variants:
        episodes = sorted(
            (
                dict(episode)
                for episode in manifest.get("episodes", [])
                if episode.get("variant") == variant
            ),
            key=lambda episode: (
                str(episode.get("pair_id", "")),
                str(episode.get("episode_id", "")),
            ),
        )
        if limit_per_variant is not None:
            episodes = episodes[:limit_per_variant]
        if episode_shard_count > len(episodes):
            raise OpenGuiAgentEvalError(
                f"episode_shard_count={episode_shard_count} exceeds "
                f"{variant} episode count={len(episodes)}"
            )
        episodes = episodes[episode_shard_index::episode_shard_count]
        selected.extend(episodes)
    return selected


def build_backend(
    spec: OpenGuiAgentSpec,
    *,
    checkpoint: Path | None = None,
) -> LocalBaselineVisionBackend:
    checkpoint_path = spec.checkpoint() if checkpoint is None else Path(checkpoint)
    if spec.prompt_style == "ui_venus2_computer":
        from ..models.ui_venus2_backend import Venus2Backend

        return Venus2Backend(checkpoint_path=checkpoint_path)
    return OfficialGuiBackend(
        checkpoint_path=checkpoint_path,
        prompt_style=spec.prompt_style,
        max_new_tokens=GENERATION_LIMITS[spec.prompt_style],
        image_max_pixels=None,
        strict_native_output=True,
        prompt_contract=spec.prompt_contract,
        button_extension=spec.button_extension,
        button_prompt_reminder=spec.button_prompt_reminder,
        history_mode=spec.history_mode,
        history_image_max=DEFAULT_HISTORY_IMAGE_MAX,
    )


def execute_model_action(
    session: ExplorationTraceSession,
    *,
    variant: str,
    action: Action | OfficialGuiAction,
    observation: Observation | None = None,
) -> StepResult:
    """Execute one model decision under the six-environment action contract.

    A coordinate click in either first-person environment is not an absolute
    screen click. Its coordinate first drives the environment's relative,
    sensitivity-aware ``move_to`` transition; the click is then released at the
    fixed center reticle. The two primitive environment steps remain visible in
    the trace, while the model-facing history retains the original atomic click.
    A first-person drag stays atomic at this layer because the environment macro
    performs its sensitivity-aware move-to-start, center grab, held transport,
    and release internally.
    """

    if isinstance(action, OfficialGuiAction):
        if observation is None:
            raise ValueError("native action execution requires the current observation")
        return execute_official_gui_action(
            session, action, observation,
            lambda command: execute_model_action(session, variant=variant, action=command),
        )

    if (
        variant in FIRST_PERSON_VARIANTS
        and isinstance(action, AtomicAction)
        and action.kind == "click"
    ):
        model_x, model_y = action.points[0]
        move_result = session.step(PrimitiveAction(kind="move_to", x=model_x, y=model_y))
        if move_result.done:
            raise OpenGuiAgentEvalError(
                f"{variant}: sensitivity-aware move unexpectedly terminated click(x,y)"
            )
        result = session.step(PrimitiveAction(kind="left_click"))
        result.action = action
        result.info["model_action_expansion"] = {
            "contract": ACTION_EXECUTION_CONTRACT,
            "model_action": action.to_dict(),
            "executed_action_kinds": ["move_to", "left_click"],
        }
        return result
    return session.step(action)


def _record_path(output_root: Path, episode: Mapping[str, Any]) -> Path:
    return output_root / "records" / str(episode["variant"]) / f"{episode['episode_id']}.json"


def _record_matches(
    record: Mapping[str, Any],
    *,
    spec: OpenGuiAgentSpec,
    checkpoint: Path,
    max_turns: int,
    suite_id: str,
) -> bool:
    return (
        record.get("record_status") == "complete"
        and record.get("suite_id") == suite_id
        and record.get("agent_key") == spec.key
        and record.get("checkpoint_path") == str(checkpoint)
        and record.get("prompt_style") == spec.prompt_style
        and record.get("prompt_contract") == spec.prompt_contract
        and record.get("model_coordinate_schema") == spec.model_coordinate_schema
        and record.get("history_strategy") == HISTORY_STRATEGIES[spec.prompt_style]
        and record.get("max_turns") == max_turns
        and record.get("allowed_action_kinds") == list(model_action_names(spec.prompt_style, button_extension=spec.button_extension))
        and record.get("protocol") == protocol_metadata(spec.prompt_style, button_extension=spec.button_extension, button_prompt_reminder=spec.button_prompt_reminder)
        and record.get("action_execution_contract") == ACTION_EXECUTION_CONTRACT
        and record.get("first_person_action_sequences")
        == FIRST_PERSON_ACTION_SEQUENCES
    )


def run_episode(
    *,
    backend: LocalBaselineVisionBackend,
    spec: OpenGuiAgentSpec,
    episode: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    rotation_base_url: str | None,
    max_turns: int,
    navigation_timeout_ms: int | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    episode_id = str(episode["episode_id"])
    variant = str(episode["variant"])
    case_dir = output_root / "cases" / variant / episode_id
    environment: Any | None = None
    session: ExplorationTraceSession | None = None
    history: list[StepResult] = []
    model_action_counts: Counter[str] = Counter()
    terminal_reason = "max_turns"
    error_payload: dict[str, str] | None = None
    calls_before = backend._call_index
    backend.call_log_dir = case_dir / "model_calls"

    try:
        kwargs: dict[str, Any] = {
            "manifest_path": manifest_path,
            "artifact_dir": case_dir / "frames",
        }
        if variant.startswith("rotation_"):
            if not rotation_base_url:
                raise OpenGuiAgentEvalError("rotation replay server is unavailable")
            kwargs["base_url"] = rotation_base_url
        environment = build_benchmark_variant(variant, **kwargs)
        if navigation_timeout_ms is not None and isinstance(environment, BrowserBenchmarkEnv):
            environment.navigation_timeout_ms = navigation_timeout_ms
        session = ExplorationTraceSession(
            env=environment,
            episode=episode,
            output_dir=case_dir,
        )
        observation = session.reset()
        policy = ClosedLoopPrimitivePolicy(
            backend=backend,
            allowed_kinds=model_action_names(spec.prompt_style, button_extension=spec.button_extension),
            condition_name=variant,
            budget=max_turns,
        )
        policy.reset()
        for _turn_index in range(max_turns):
            try:
                action = policy.predict(observation, history)
                model_action_counts[action.kind] += 1
            except OfficialGuiProtocolError as error:
                terminal_reason = "model_protocol_error"
                error_payload = {"type": type(error).__name__, "message": str(error)}
                break
            except Exception as error:
                terminal_reason = "predict_error"
                error_payload = {"type": type(error).__name__, "message": str(error)}
                break
            try:
                result = execute_model_action(
                    session,
                    variant=variant,
                    action=action,
                    observation=observation,
                )
            except OfficialGuiProtocolError as error:
                terminal_reason = "model_protocol_error"
                error_payload = {"type": type(error).__name__, "message": str(error)}
                break
            except Exception as error:
                terminal_reason = "step_error"
                error_payload = {"type": type(error).__name__, "message": str(error)}
                break
            result.action = action
            if isinstance(backend.last_raw_prediction, str):
                result.info["model_raw_prediction"] = backend.last_raw_prediction
            if isinstance(backend.last_think_text, str) and backend.last_think_text:
                result.info["model_think_text"] = backend.last_think_text
            history.append(result)
            observation = result.observation
            if result.done:
                terminal_reason = "success" if result.reward == 1.0 else "failure"
                break
    except Exception as error:
        terminal_reason = "infra_error"
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        trace_paths = (
            session.write() if session is not None else {"trace_path": None, "audit_path": None}
        )
        if environment is not None:
            environment.close()

    record = {
        "record_status": "complete",
        "suite_id": episode["suite_id"],
        "pair_id": episode["pair_id"],
        "family": episode["family"],
        "variant": variant,
        "exploration_level": episode["exploration_level"],
        "episode_id": episode_id,
        "case_seed": episode["case_seed"],
        "split": episode["split"],
        "agent_key": spec.key,
        "agent_label": spec.label,
        "checkpoint_path": str(backend.checkpoint_path),
        "prompt_style": spec.prompt_style,
        "prompt_contract": backend.prompt_contract,
        "model_coordinate_schema": spec.model_coordinate_schema,
        "history_strategy": HISTORY_STRATEGIES[spec.prompt_style],
        "protocol": protocol_metadata(spec.prompt_style, button_extension=spec.button_extension, button_prompt_reminder=spec.button_prompt_reminder),
        "success": terminal_reason == "success",
        "terminal_reason": terminal_reason,
        "max_turns": max_turns,
        "executed_turns": len(history),
        "vlm_calls": backend._call_index - calls_before,
        "semantic_interactions": (session.semantic_interaction_index if session is not None else 0),
        "model_action_counts": dict(sorted(model_action_counts.items())),
        "allowed_action_kinds": list(model_action_names(spec.prompt_style, button_extension=spec.button_extension)),
        "environment_action_kinds": list(SIX_ACTION_KINDS),
        "action_execution_contract": ACTION_EXECUTION_CONTRACT,
        "first_person_action_sequences": FIRST_PERSON_ACTION_SEQUENCES,
        "strict_native_output": backend.strict_native_output,
        **trace_paths,
        "error": error_payload,
        "latency_s": time.monotonic() - started_at,
    }
    _write_json(_record_path(output_root, episode), record)
    return record


def load_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "records").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def summarize(
    records: list[dict[str, Any]],
    *,
    selected_episodes: list[dict[str, Any]],
    output_root: Path,
    spec: OpenGuiAgentSpec,
    checkpoint: Path,
    max_turns: int,
) -> dict[str, Any]:
    selected_counts = Counter(str(episode["variant"]) for episode in selected_episodes)
    selected_ids = {
        (str(episode["variant"]), str(episode["episode_id"])) for episode in selected_episodes
    }
    matched = [
        record
        for record in records
        if (str(record.get("variant")), str(record.get("episode_id"))) in selected_ids
        and _record_matches(
            record,
            spec=spec,
            checkpoint=checkpoint,
            max_turns=max_turns,
            suite_id=str(selected_episodes[0]["suite_id"]),
        )
    ]
    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [record for record in matched if record.get("variant") == variant]
        expected_count = selected_counts.get(variant, 0)
        successes = sum(record.get("success") is True for record in subset)
        action_counts: Counter[str] = Counter()
        for record in subset:
            action_counts.update(record.get("model_action_counts", {}))
        per_variant[variant] = {
            "expected_count": expected_count,
            "completed_count": len(subset),
            "missing_count": max(0, expected_count - len(subset)),
            "success_count": successes,
            "success_rate": successes / len(subset) if subset else 0.0,
            "terminal_counts": dict(
                sorted(Counter(str(record.get("terminal_reason")) for record in subset).items())
            ),
            "model_action_counts": dict(sorted(action_counts.items())),
        }
    expected_count = len(selected_episodes)
    success_count = sum(record.get("success") is True for record in matched)
    error_count = sum(
        str(record.get("terminal_reason")) in ERROR_TERMINAL_REASONS for record in matched
    )
    latencies = [float(record.get("latency_s", 0.0)) for record in matched]
    return {
        "evaluation": "six_open_gui_agents_official_first_six_env_v1",
        "suite_id": selected_episodes[0]["suite_id"] if selected_episodes else None,
        "agent_key": spec.key,
        "agent_label": spec.label,
        "checkpoint_path": str(checkpoint),
        "prompt_style": spec.prompt_style,
        "prompt_contract": spec.prompt_contract,
        "model_coordinate_schema": spec.model_coordinate_schema,
        "history_strategy": HISTORY_STRATEGIES[spec.prompt_style],
        "protocol": protocol_metadata(spec.prompt_style, button_extension=spec.button_extension, button_prompt_reminder=spec.button_prompt_reminder),
        "output_root": str(output_root),
        "allowed_action_kinds": list(model_action_names(spec.prompt_style, button_extension=spec.button_extension)),
        "environment_action_kinds": list(SIX_ACTION_KINDS),
        "action_execution_contract": ACTION_EXECUTION_CONTRACT,
        "first_person_action_sequences": FIRST_PERSON_ACTION_SEQUENCES,
        "max_turns": max_turns,
        "expected_count": expected_count,
        "completed_count": len(matched),
        "missing_count": max(0, expected_count - len(matched)),
        "success_count": success_count,
        "success_rate": success_count / len(matched) if matched else 0.0,
        "error_count": error_count,
        "model_protocol_error_count": sum(
            record.get("terminal_reason") == "model_protocol_error" for record in matched
        ),
        "mean_latency_s": statistics.fmean(latencies) if latencies else None,
        "variants": per_variant,
        "status": (
            "complete_with_errors"
            if len(matched) == expected_count and error_count
            else "complete"
            if len(matched) == expected_count
            else "incomplete"
        ),
    }


def evaluate(
    *,
    manifest_path: Path,
    output_root: Path,
    spec: OpenGuiAgentSpec,
    checkpoint: Path | None = None,
    variants: Iterable[str] = VARIANTS,
    limit_per_variant: int | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    retry_errors: bool = False,
    episode_shard_index: int = 0,
    episode_shard_count: int = 1,
) -> dict[str, Any]:
    if max_turns != DEFAULT_MAX_TURNS:
        raise OpenGuiAgentEvalError(f"formal evaluation requires max_turns={DEFAULT_MAX_TURNS}")
    selected_variants = tuple(dict.fromkeys(str(value) for value in variants))
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    manifest = load_manifest(manifest_path)
    selected = select_episodes(
        manifest,
        variants=selected_variants,
        limit_per_variant=limit_per_variant,
        episode_shard_index=episode_shard_index,
        episode_shard_count=episode_shard_count,
    )
    checkpoint_path = spec.checkpoint() if checkpoint is None else Path(checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    output_root.mkdir(parents=True, exist_ok=True)
    backend = build_backend(spec, checkpoint=checkpoint_path)
    run_manifest = {
        "evaluation": "six_open_gui_agents_official_first_six_env_v1",
        "suite_manifest": str(manifest_path),
        "suite_id": manifest["suite_id"],
        "agent_key": spec.key,
        "agent_label": spec.label,
        "checkpoint_path": str(checkpoint_path),
        "prompt_style": spec.prompt_style,
        "prompt_contract": spec.prompt_contract,
        "model_coordinate_schema": spec.model_coordinate_schema,
        "history_strategy": HISTORY_STRATEGIES[spec.prompt_style],
        "protocol": protocol_metadata(spec.prompt_style, button_extension=spec.button_extension, button_prompt_reminder=spec.button_prompt_reminder),
        "selected_variants": list(selected_variants),
        "selected_count": len(selected),
        "limit_per_variant": limit_per_variant,
        "episode_shard_index": episode_shard_index,
        "episode_shard_count": episode_shard_count,
        "allowed_action_kinds": list(model_action_names(spec.prompt_style, button_extension=spec.button_extension)),
        "environment_action_kinds": list(SIX_ACTION_KINDS),
        "action_execution_contract": ACTION_EXECUTION_CONTRACT,
        "first_person_action_sequences": FIRST_PERSON_ACTION_SEQUENCES,
        "max_turns": max_turns,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    completion_path = output_root / "evaluation_complete.json"
    if completion_path.is_file():
        completion_path.unlink()

    replay_context = (
        RotationReplayServer(public_root=manifest_path.parent)
        if any(str(episode["variant"]).startswith("rotation_") for episode in selected)
        else nullcontext(None)
    )
    try:
        with replay_context as replay:
            rotation_base_url = replay.base_url if replay is not None else None
            for episode in selected:
                record_path = _record_path(output_root, episode)
                if record_path.is_file():
                    try:
                        existing = json.loads(record_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        existing = {}
                    reusable = _record_matches(
                        existing,
                        spec=spec,
                        checkpoint=checkpoint_path,
                        max_turns=max_turns,
                        suite_id=str(manifest["suite_id"]),
                    )
                    if reusable and not (
                        retry_errors and existing.get("terminal_reason") in ERROR_TERMINAL_REASONS
                    ):
                        record = existing
                    else:
                        record = run_episode(
                            backend=backend,
                            spec=spec,
                            episode=episode,
                            manifest_path=manifest_path,
                            output_root=output_root,
                            rotation_base_url=rotation_base_url,
                            max_turns=max_turns,
                        )
                else:
                    record = run_episode(
                        backend=backend,
                        spec=spec,
                        episode=episode,
                        manifest_path=manifest_path,
                        output_root=output_root,
                        rotation_base_url=rotation_base_url,
                        max_turns=max_turns,
                    )
                print(
                    f"agent={spec.key} variant={record['variant']} "
                    f"episode={record['episode_id']} success={record['success']} "
                    f"terminal={record['terminal_reason']} turns={record['executed_turns']}",
                    flush=True,
                )
                records = load_records(output_root)
                summary = summarize(
                    records,
                    selected_episodes=selected,
                    output_root=output_root,
                    spec=spec,
                    checkpoint=checkpoint_path,
                    max_turns=max_turns,
                )
                _write_json(output_root / "summary.json", summary)
    finally:
        backend.unload()

    records = load_records(output_root)
    summary = summarize(
        records,
        selected_episodes=selected,
        output_root=output_root,
        spec=spec,
        checkpoint=checkpoint_path,
        max_turns=max_turns,
    )
    selected_ids = {(str(episode["variant"]), str(episode["episode_id"])) for episode in selected}
    matched_records = [
        record
        for record in records
        if (str(record.get("variant")), str(record.get("episode_id"))) in selected_ids
        if _record_matches(
            record,
            spec=spec,
            checkpoint=checkpoint_path,
            max_turns=max_turns,
            suite_id=str(manifest["suite_id"]),
        )
    ]
    _write_json(output_root / "summary.json", summary)
    _write_jsonl(
        output_root / "records.jsonl",
        sorted(
            matched_records,
            key=lambda record: (str(record["variant"]), str(record["episode_id"])),
        ),
    )
    if summary["missing_count"] == 0:
        _write_json(
            completion_path,
            {
                "status": summary["status"],
                "suite_id": summary["suite_id"],
                "agent_key": spec.key,
                "prompt_contract": spec.prompt_contract,
                "model_coordinate_schema": spec.model_coordinate_schema,
                "completed_count": summary["completed_count"],
                "error_count": summary["error_count"],
                "summary_path": str(output_root / "summary.json"),
            },
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one of six open GUI agents on the six paired "
            "exploration-depth environments."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--agent-key", choices=tuple(OPEN_GUI_AGENT_BY_KEY), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit-per-variant", type=int)
    parser.add_argument("--episode-shard-index", type=int, default=0)
    parser.add_argument("--episode-shard-count", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--button-extension", action="store_true",
                        help="Advertise and execute independent mouse_down/mouse_up actions")
    parser.add_argument("--button-prompt-reminder", action="store_true",
                        help="Use the tested end-of-prompt mouse tool reminder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = evaluate(
        manifest_path=args.manifest,
        output_root=args.output_root,
        spec=replace(OPEN_GUI_AGENT_BY_KEY[args.agent_key],
                     button_extension=args.button_extension,
                     button_prompt_reminder=args.button_prompt_reminder),
        checkpoint=args.checkpoint,
        variants=args.variants,
        limit_per_variant=args.limit_per_variant,
        max_turns=args.max_turns,
        retry_errors=args.retry_errors,
        episode_shard_index=args.episode_shard_index,
        episode_shard_count=args.episode_shard_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["missing_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
