from __future__ import annotations

import gc
import inspect
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from PIL import Image

from ..actions import Action, AtomicAction, PrimitiveAction, WebAction
from ..core import Observation, StepResult
from ..train.qwen3_vl_sft import (
    HD720_IMAGE_MAX_PIXELS,
    ImageCoordinateTransform,
    deterministic_generation_kwargs,
    qwen3_vl_multimodal_processor_kwargs,
    resize_image_for_max_pixels,
)
from .baseline_adapters import (
    BaselinePromptStyle,
    PyAutoGUICoordinateMode,
    normalize_baseline_output,
    render_for_baseline,
)
from .native_baseline_protocols import (
    holo3_structured_output_schema,
    holo_observation_content,
    holo_tool_name,
    native_action_kinds,
    native_assistant_history_content,
    native_history_action_summary,
    native_six_action_system_prompt,
    native_six_action_task_prompt,
    native_system_prompt,
    native_task_prompt,
    qwen25_native_coordinate_size,
)

ModelLoader = Literal["auto", "image_text_to_text", "vision2seq", "causal_lm"]
BaselinePromptContract = Literal["default", "native_six_action_v1"]
BaselineHistoryMode = Literal[
    "single_current",
    "uitars_osworld",
    "dart_osworld",
    "evocua_s2",
    "opencua_l2",
    "gui_owl15",
    "holo3_tool",
    "mai_ui",
    "ui_voyager",
    "ui_venus",
]


@dataclass
class LocalBaselineVisionBackend:
    """Local inference backend for native GUI-agent checkpoints.

    The model receives its native-ish prompt shell from ``baseline_adapters`` and
    the raw response is normalized back into this repo's Action dataclasses.
    """

    checkpoint_path: Path
    prompt_style: BaselinePromptStyle
    processor: Any | None = None
    model: Any | None = None
    max_new_tokens: int = 256
    image_max_pixels: int | None = HD720_IMAGE_MAX_PIXELS
    device_map: str = "auto"
    trust_remote_code: bool = True
    model_loader: ModelLoader = "auto"
    pyautogui_coordinate_mode: PyAutoGUICoordinateMode | None = None
    strict_native_output: bool = True
    prompt_contract: BaselinePromptContract = "default"
    history_mode: BaselineHistoryMode | None = None
    history_image_max: int = 5
    call_log_dir: Path | None = None
    _call_index: int = field(default=0, init=False)
    _history_image_paths: list[Path] = field(default_factory=list, init=False)
    _history_responses: list[str] = field(default_factory=list, init=False)
    _history_assistant_contents: list[str] = field(default_factory=list, init=False)
    _history_action_summaries: list[str] = field(default_factory=list, init=False)
    _history_actions: list[Action] = field(default_factory=list, init=False)
    _history_observations: list[Observation] = field(default_factory=list, init=False)
    _last_history_length: int = field(default=-1, init=False)
    last_prediction_index: int | None = field(default=None, init=False)
    last_raw_prediction: str | None = field(default=None, init=False)
    last_think_text: str | None = field(default=None, init=False)
    last_prompt_messages: list[dict[str, Any]] | None = field(default=None, init=False)
    last_prompt_text: str | None = field(default=None, init=False)
    last_prompt_image_paths: tuple[str, ...] = field(default_factory=tuple, init=False)
    last_native_image_size_px: tuple[int, int] | None = field(default=None, init=False)
    _holo3_json_processor_templates: dict[tuple[str, ...], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _ensure_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            if hasattr(self.model, "eval"):
                self.model.eval()
            return
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Local baseline GUI-agent evaluation requires torch and transformers."
            ) from exc

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                str(self.checkpoint_path),
                trust_remote_code=self.trust_remote_code,
            )
        if self.model is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            load_errors: list[str] = []
            self._patch_remote_model_compat(AutoConfig)
            for loader_name in self._candidate_loader_names(AutoConfig):
                loader = getattr(transformers, loader_name, None)
                if loader is None:
                    load_errors.append(f"{loader_name}: unavailable")
                    continue
                try:
                    self.model = self._from_pretrained(loader, dtype=dtype)
                    break
                except Exception as exc:
                    load_errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
            if self.model is None:
                joined = "\n".join(load_errors)
                raise RuntimeError(f"failed to load {self.checkpoint_path}:\n{joined}")
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _candidate_loader_names(self, auto_config: Any) -> tuple[str, ...]:
        explicit = {
            "image_text_to_text": ("AutoModelForImageTextToText",),
            "vision2seq": ("AutoModelForVision2Seq",),
            "causal_lm": ("AutoModelForCausalLM",),
        }
        if self.model_loader in explicit:
            return explicit[self.model_loader]
        try:
            config = auto_config.from_pretrained(
                str(self.checkpoint_path),
                trust_remote_code=self.trust_remote_code,
            )
            model_type = str(getattr(config, "model_type", ""))
        except Exception:
            model_type = ""
        if model_type == "opencua":
            return (
                "AutoModelForCausalLM",
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
            )
        if model_type == "qwen3_5_moe":
            return (
                "AutoModelForImageTextToText",
                "AutoModelForCausalLM",
                "AutoModelForVision2Seq",
            )
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return (
                "AutoModelForVision2Seq",
                "AutoModelForImageTextToText",
                "AutoModelForCausalLM",
            )
        return (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        )

    def _patch_remote_model_compat(self, auto_config: Any) -> None:
        try:
            config = auto_config.from_pretrained(
                str(self.checkpoint_path),
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            return
        if str(getattr(config, "model_type", "")) != "opencua":
            return
        auto_map = getattr(config, "auto_map", {}) or {}
        class_ref = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
        if not class_ref:
            return
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            model_cls = get_class_from_dynamic_module(
                class_ref,
                str(self.checkpoint_path),
                local_files_only=True,
            )
        except Exception:
            return
        original = getattr(model_cls, "tie_weights", None)
        if original is not None and not getattr(original, "_gui_captcha_compat", False):
            accepts_kwargs = False
            try:
                signature = inspect.signature(original)
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            except (TypeError, ValueError):
                pass
            if not accepts_kwargs:

                def tie_weights_compat(model_self: Any, *_args: Any, **_kwargs: Any) -> Any:
                    return original(model_self)

                tie_weights_compat._gui_captcha_compat = True  # type: ignore[attr-defined]
                setattr(model_cls, "tie_weights", tie_weights_compat)
        try:
            from transformers.generation import GenerationMixin
        except Exception:
            return
        for name, value in GenerationMixin.__dict__.items():
            if name.startswith("__") or hasattr(model_cls, name):
                continue
            setattr(model_cls, name, value)
        _patch_opencua_dynamic_cache_indexing()
        image_feature_extractor = getattr(model_cls, "_extract_image_features", None)
        if image_feature_extractor is not None and not getattr(
            image_feature_extractor,
            "_gui_captcha_compat",
            False,
        ):

            def _extract_image_features_compat(
                model_self: Any,
                pixel_values: Any,
                image_grid_thw: Any,
            ) -> tuple[Any, list[int]]:
                import torch

                if len(image_grid_thw.shape) != 2 or image_grid_thw.shape[1] != 3:
                    raise AssertionError(
                        "image_grid_thw must be a 2D tensor with shape (batched, 3), "
                        f"but got {image_grid_thw.shape}"
                    )
                if isinstance(pixel_values, list):
                    pixel_values = torch.cat(pixel_values, dim=0)
                image_features = model_self.vision_tower(pixel_values, grid_thw=image_grid_thw)
                image_features = _opencua_vision_tensor(image_features)
                image_features_list = []
                start_idx = 0
                spatial_merge_unit = int(
                    getattr(model_self.vision_tower, "spatial_merge_unit", 4) or 4
                )
                for grid_thw in image_grid_thw:
                    merged_tokens = int((grid_thw[0] * grid_thw[1] * grid_thw[2]).item())
                    end_idx = start_idx + merged_tokens // spatial_merge_unit
                    image_features_list.append(image_features[start_idx:end_idx, :])
                    start_idx = end_idx
                selected_image_feature = torch.cat(image_features_list, dim=0)
                feature_lengths = [int(features.size(0)) for features in image_features_list]
                return selected_image_feature, feature_lengths

            _extract_image_features_compat._gui_captcha_compat = True  # type: ignore[attr-defined]
            setattr(model_cls, "_extract_image_features", _extract_image_features_compat)

    def _from_pretrained(self, loader: Any, *, dtype: Any) -> Any:
        kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        try:
            return loader.from_pretrained(str(self.checkpoint_path), dtype=dtype, **kwargs)
        except TypeError:
            return loader.from_pretrained(str(self.checkpoint_path), torch_dtype=dtype, **kwargs)

    def _load_image_with_transform(self, image_path: Path) -> tuple[Image.Image, ImageCoordinateTransform]:
        image = Image.open(image_path).convert("RGB")
        return resize_image_for_max_pixels(image, self.image_max_pixels)

    def _resolved_history_mode(self) -> BaselineHistoryMode:
        if self.history_mode is not None:
            return self.history_mode
        defaults: dict[BaselinePromptStyle, BaselineHistoryMode] = {
            "uitars": "uitars_osworld",
            "dart_gui": "dart_osworld",
            "evocua_s2": "evocua_s2",
            "opencua_pyautogui": "opencua_l2",
            "gui_owl": "gui_owl15",
            "holo3_tool": "holo3_tool",
            "mai_ui": "mai_ui",
            "ui_voyager": "ui_voyager",
            "ui_venus": "ui_venus",
        }
        return defaults.get(self.prompt_style, "single_current")

    def _native_system_prompt(self, allowed_kinds: Iterable[str]) -> str:
        if self.prompt_contract == "native_six_action_v1":
            return native_six_action_system_prompt(
                self.prompt_style,
                allowed_kinds=allowed_kinds,
            )
        if self.prompt_contract == "default":
            return native_system_prompt(self.prompt_style, allowed_kinds=allowed_kinds)
        raise ValueError(f"unsupported baseline prompt contract: {self.prompt_contract!r}")

    def _native_task_prompt(
        self,
        obs: Observation,
        *,
        allowed_kinds: Iterable[str],
        previous_actions: str | None = None,
    ) -> str:
        if self.prompt_contract == "native_six_action_v1":
            return native_six_action_task_prompt(
                self.prompt_style,
                obs,
                allowed_kinds=allowed_kinds,
                previous_actions=previous_actions,
            )
        if self.prompt_contract == "default":
            return native_task_prompt(
                self.prompt_style,
                obs,
                allowed_kinds=allowed_kinds,
                previous_actions=previous_actions,
            )
        raise ValueError(f"unsupported baseline prompt contract: {self.prompt_contract!r}")

    def _reset_history_if_needed(self, history: list[StepResult]) -> None:
        if len(history) == 0:
            self._clear_native_history()
        elif len(history) < self._last_history_length:
            self._clear_native_history()
        self._last_history_length = len(history)

    def _clear_native_history(self) -> None:
        self._history_image_paths.clear()
        self._history_responses.clear()
        self._history_assistant_contents.clear()
        self._history_action_summaries.clear()
        self._history_actions.clear()
        self._history_observations.clear()

    def _build_prompt_messages(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
        condition: str | None,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        current_image = Path(obs.screenshot_path)
        history_mode = self._resolved_history_mode()
        if history_mode == "single_current":
            prompt = render_for_baseline(
                self.prompt_style,
                obs,
                history,
                allowed_kinds=allowed_kinds,
                budget=budget,
                condition=condition,
            )
            return (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(current_image)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                [current_image],
            )

        native_allowed = (
            tuple(dict.fromkeys(str(kind) for kind in allowed_kinds if str(kind)))
            if self.prompt_style == "ui_venus"
            else native_action_kinds(allowed_kinds)
        )
        system_prompt = self._native_system_prompt(native_allowed)
        if history_mode in {"uitars_osworld", "dart_osworld"}:
            task_prompt = self._native_task_prompt(
                obs,
                allowed_kinds=native_allowed,
            )
            return self._build_uitars_family_messages(
                system_prompt,
                task_prompt,
                current_image,
                historical_image_max=self._history_image_budget(history_mode),
            )
        if history_mode in {"gui_owl15", "evocua_s2"}:
            return self._build_tool_call_family_messages(
                obs,
                system_prompt,
                current_image,
                allowed_kinds=native_allowed,
                historical_image_max=self._history_image_budget(history_mode),
                image_first=history_mode == "evocua_s2",
            )
        if history_mode == "opencua_l2":
            return self._build_opencua_messages(
                obs,
                system_prompt,
                current_image,
                allowed_kinds=native_allowed,
            )
        if history_mode == "holo3_tool":
            return self._build_holo3_messages(
                obs,
                system_prompt,
                current_image,
            )
        if history_mode == "mai_ui":
            return self._build_mai_ui_messages(
                obs,
                system_prompt,
                current_image,
                allowed_kinds=native_allowed,
            )
        if history_mode == "ui_voyager":
            return self._build_ui_voyager_messages(
                obs,
                system_prompt,
                current_image,
                allowed_kinds=native_allowed,
            )
        if history_mode == "ui_venus":
            return self._build_ui_venus_messages(
                obs,
                system_prompt,
                current_image,
                allowed_kinds=native_allowed,
            )
        raise ValueError(f"unsupported baseline history mode: {history_mode}")

    def _history_image_budget(self, history_mode: BaselineHistoryMode) -> int:
        if history_mode in {"uitars_osworld", "dart_osworld"}:
            return min(max(0, self.history_image_max - 1), 4)
        if history_mode in {"gui_owl15", "evocua_s2"}:
            return min(self.history_image_max, 4)
        if history_mode in {"opencua_l2", "holo3_tool"}:
            return min(self.history_image_max, 2)
        if history_mode == "mai_ui":
            # Official MAI-UI history_n=3 means two previous images plus current.
            return min(self.history_image_max, 2)
        return 0

    def _history_record_count(self) -> int:
        return min(
            len(self._history_image_paths),
            len(self._history_responses),
            len(self._history_assistant_contents),
            len(self._history_action_summaries),
            len(self._history_actions),
            len(self._history_observations),
        )

    def _build_uitars_family_messages(
        self,
        system_prompt: str,
        task_prompt: str,
        current_image: Path,
        *,
        historical_image_max: int,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": task_prompt}],
            },
        ]
        image_paths: list[Path] = []
        count = self._history_record_count()
        first_image_index = max(0, count - historical_image_max)
        for index in range(count):
            if index >= first_image_index:
                image_path = self._history_image_paths[index]
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(image_path)}],
                    }
                )
                image_paths.append(image_path)
            messages.append(
                {"role": "assistant", "content": self._history_assistant_contents[index]}
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": str(current_image)}],
            }
        )
        image_paths.append(current_image)
        return messages, image_paths

    def _build_tool_call_family_messages(
        self,
        obs: Observation,
        system_prompt: str,
        current_image: Path,
        *,
        allowed_kinds: Iterable[str],
        historical_image_max: int,
        image_first: bool,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        count = self._history_record_count()
        first_recent_index = max(0, count - historical_image_max)
        previous_actions = "\n".join(
            f"Step {index + 1}: {self._history_action_summaries[index]}"
            for index in range(first_recent_index)
        ) or "None"
        task_prompt = self._native_task_prompt(
            obs,
            allowed_kinds=allowed_kinds,
            previous_actions=previous_actions,
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]
        image_paths: list[Path] = []
        if first_recent_index >= count:
            content = [
                {"type": "image", "image": str(current_image)},
                {"type": "text", "text": task_prompt},
            ]
            if not image_first:
                content.reverse()
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
            image_paths.append(current_image)
            return messages, image_paths

        for index in range(first_recent_index, count):
            image_path = self._history_image_paths[index]
            if index == first_recent_index:
                content = [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": task_prompt},
                ]
                if not image_first:
                    content.reverse()
            else:
                content = [{"type": "image", "image": str(image_path)}]
            messages.append({"role": "user", "content": content})
            image_paths.append(image_path)
            messages.append(
                {"role": "assistant", "content": self._history_assistant_contents[index]}
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": str(current_image)}],
            }
        )
        image_paths.append(current_image)
        return messages, image_paths

    def _build_opencua_messages(
        self,
        obs: Observation,
        system_prompt: str,
        current_image: Path,
        *,
        allowed_kinds: Iterable[str],
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        count = self._history_record_count()
        historical_image_max = self._history_image_budget("opencua_l2")
        first_recent_index = max(0, count - historical_image_max)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        if first_recent_index > 0:
            older = "\n".join(
                f"# Step {index + 1}:\n## Action:\n{self._history_action_summaries[index]}"
                for index in range(first_recent_index)
            )
            messages.append({"role": "assistant", "content": older})
        image_paths: list[Path] = []
        for index in range(first_recent_index, count):
            image_path = self._history_image_paths[index]
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": str(image_path)}],
                }
            )
            image_paths.append(image_path)
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"# Step {index + 1}:\n"
                        f"## Action:\n{self._history_action_summaries[index]}"
                    ),
                }
            )
        task_prompt = self._native_task_prompt(
            obs,
            allowed_kinds=allowed_kinds,
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(current_image)},
                    {"type": "text", "text": task_prompt},
                ],
            }
        )
        image_paths.append(current_image)
        return messages, image_paths

    def _build_holo3_messages(
        self,
        obs: Observation,
        system_prompt: str,
        current_image: Path,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        count = self._history_record_count()
        first_retained_image = max(0, count - self._history_image_budget("holo3_tool"))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        image_paths: list[Path] = []
        for index in range(count):
            keep_image = index >= first_retained_image
            image_path = self._history_image_paths[index]
            historical_obs = self._history_observations[index]
            messages.append(
                {
                    "role": "user",
                    "content": holo_observation_content(
                        historical_obs,
                        image_path=str(image_path) if keep_image else None,
                        evicted=not keep_image,
                    ),
                }
            )
            if keep_image:
                image_paths.append(image_path)
            messages.append(
                {"role": "assistant", "content": self._history_assistant_contents[index]}
            )
            action = self._history_actions[index]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f'<tool_output tool="{holo_tool_name(action)}">\n'
                        "Action executed.\n"
                        "</tool_output>"
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": holo_observation_content(
                    obs,
                    image_path=str(current_image),
                ),
            }
        )
        image_paths.append(current_image)
        return messages, image_paths

    def _build_mai_ui_messages(
        self,
        obs: Observation,
        system_prompt: str,
        current_image: Path,
        *,
        allowed_kinds: Iterable[str],
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        """Match MAI-UI's all-actions/recent-three-images message history."""

        task_prompt = self._native_task_prompt(
            obs,
            allowed_kinds=allowed_kinds,
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": task_prompt}],
            },
        ]
        image_paths: list[Path] = []
        count = self._history_record_count()
        first_retained_image = max(
            0,
            count - self._history_image_budget("mai_ui"),
        )
        for index in range(count):
            if index >= first_retained_image:
                image_path = self._history_image_paths[index]
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(image_path)}],
                    }
                )
                image_paths.append(image_path)
            messages.append(
                {"role": "assistant", "content": self._history_assistant_contents[index]}
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": str(current_image)}],
            }
        )
        image_paths.append(current_image)
        return messages, image_paths

    def _build_ui_voyager_messages(
        self,
        obs: Observation,
        system_prompt: str,
        current_image: Path,
        *,
        allowed_kinds: Iterable[str],
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        """Match UI-Voyager's action-summary textual history."""

        task_prompt = self._native_task_prompt(
            obs,
            allowed_kinds=allowed_kinds,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": task_prompt}]
        count = self._history_record_count()
        if count:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Task progress (You have done the following operation on the current "
                        "device):\n"
                    ),
                }
            )
            for index in range(count):
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Step {index + 1}: "
                            f"{self._history_action_summaries[index]};\n"
                        ),
                    }
                )
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "Current Screenshot: <image>\n\n"
                        "Please analyze the current screenshot and history to generate the next step."
                    ),
                },
                {"type": "image", "image": str(current_image)},
            ]
        )
        return (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            [current_image],
        )

    def _build_ui_venus_messages(
        self,
        obs: Observation,
        system_prompt: str,
        current_image: Path,
        *,
        allowed_kinds: Iterable[str],
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        """Match UI-Venus web navigation's textual history and current-image input."""

        count = self._history_record_count()
        first = max(0, count - 5)
        previous_actions = "\n".join(
            f"Step {step_index}: {self._history_action_summaries[history_index]}"
            for step_index, history_index in enumerate(range(first, count))
        )
        task_prompt = self._native_task_prompt(
            obs,
            allowed_kinds=allowed_kinds,
            previous_actions=previous_actions,
        )
        return (
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_prompt},
                        {"type": "image", "image": str(current_image)},
                    ],
                },
            ],
            [current_image],
        )

    def _build_prompt_text(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
        condition: str | None,
    ) -> tuple[str, list[Path], list[dict[str, Any]]]:
        assert self.processor is not None
        messages, image_paths = self._build_prompt_messages(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
            condition=condition,
        )
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.prompt_style == "holo3_tool":
            template_kwargs["enable_thinking"] = True
        prompt_text = self.processor.apply_chat_template(messages, **template_kwargs)
        return prompt_text, image_paths, messages

    def _resolved_pyautogui_coordinate_mode(self) -> PyAutoGUICoordinateMode:
        if self.pyautogui_coordinate_mode is not None:
            return self.pyautogui_coordinate_mode
        if self.prompt_style == "opencua_pyautogui":
            return "qwen25_smart_resize"
        return "relative_0_1000"

    def _generation_kwargs(self) -> dict[str, Any]:
        if self.prompt_style == "holo3_tool":
            return {
                "max_new_tokens": max(self.max_new_tokens, 512),
                "do_sample": True,
                "temperature": 0.8,
            }
        if self.prompt_style == "uitars":
            return deterministic_generation_kwargs(max(self.max_new_tokens, 1000))
        if self.prompt_style == "dart_gui":
            return {
                "max_new_tokens": max(self.max_new_tokens, 500),
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
            }
        if self.prompt_style == "opencua_pyautogui":
            return deterministic_generation_kwargs(max(self.max_new_tokens, 2048))
        if self.prompt_style == "mai_ui":
            return deterministic_generation_kwargs(max(self.max_new_tokens, 2048))
        if self.prompt_style == "ui_voyager":
            return {
                "max_new_tokens": max(self.max_new_tokens, 2048),
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
        if self.prompt_style == "ui_venus":
            return {
                **deterministic_generation_kwargs(max(self.max_new_tokens, 2048)),
                "repetition_penalty": 1.05,
            }
        return deterministic_generation_kwargs(self.max_new_tokens)

    def _holo3_tokenizer(self) -> Any:
        assert self.processor is not None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None and all(
            hasattr(self.processor, name) for name in ("encode", "get_vocab")
        ):
            tokenizer = self.processor
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise RuntimeError(
                "Holo3 constrained decoding requires a Hugging Face tokenizer on the processor"
            )
        return tokenizer

    def _holo3_json_logits_processor(self, allowed_kinds: Iterable[str]) -> Any:
        actions = native_action_kinds(allowed_kinds)
        template = self._holo3_json_processor_templates.get(actions)
        if template is None:
            try:
                from outlines.models.transformers import TransformerTokenizer
                from outlines.processors import JSONLogitsProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Holo3 local evaluation requires Outlines for decoding-level "
                    "structured JSON generation"
                ) from exc
            template = JSONLogitsProcessor(
                holo3_structured_output_schema(actions),
                TransformerTokenizer(self._holo3_tokenizer()),
                whitespace_pattern=r"[\n ]?",
            )
            self._holo3_json_processor_templates[actions] = template
        return template.copy()

    def _generate_holo3_structured(
        self,
        batch: Any,
        *,
        allowed_kinds: Iterable[str],
    ) -> Any:
        """Generate Holo3 reasoning first, then grammar-constrained assistant JSON.

        The checkpoint chat template ends at ``<think>`` when thinking is enabled.
        Applying a JSON grammar immediately would suppress the native reasoning phase.
        Instead, the first generation stops at ``</think>``; a fresh Outlines state
        then constrains only the assistant-content JSON that follows it.
        """

        assert self.model is not None
        tokenizer = self._holo3_tokenizer()
        close_text = "</think>"
        close_ids = _tokenizer_encode(tokenizer, close_text)
        if not close_ids:
            raise RuntimeError("Holo3 tokenizer cannot encode </think>")

        reasoning_kwargs = dict(self._generation_kwargs())
        reasoning_kwargs["stopping_criteria"] = [_TokenSuffixStoppingCriteria(close_ids)]
        reasoning_output = self.model.generate(**batch, **reasoning_kwargs)
        input_len = batch["input_ids"].shape[-1]
        reasoning_ids = reasoning_output[0][input_len:]
        reasoning_text = self.processor.decode(  # type: ignore[union-attr]
            reasoning_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        reasoning_prefix = _closed_holo3_reasoning_prefix(str(reasoning_text))
        prefix_ids = _tokenizer_encode(tokenizer, reasoning_prefix + "\n\n")
        structured_batch = _append_text_token_ids(batch, prefix_ids)

        structured_kwargs = dict(self._generation_kwargs())
        structured_kwargs["logits_processor"] = [
            self._holo3_json_logits_processor(allowed_kinds)
        ]
        return self.model.generate(**structured_batch, **structured_kwargs)

    def _generate_raw(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
        condition: str | None,
    ) -> tuple[str, ImageCoordinateTransform]:
        assert self.processor is not None
        assert self.model is not None

        image_path = Path(obs.screenshot_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"baseline backend screenshot is missing: {image_path}")

        prompt_text, prompt_image_paths, prompt_messages = self._build_prompt_text(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
            condition=condition,
        )
        all_images: list[Image.Image] = []
        transform: ImageCoordinateTransform | None = None
        for prompt_image_path in prompt_image_paths:
            image, image_transform = self._load_image_with_transform(prompt_image_path)
            all_images.append(image)
            transform = image_transform
        if not all_images:
            image, transform = self._load_image_with_transform(image_path)
            all_images = [image]
            prompt_image_paths = [image_path]
        self.last_prompt_messages = prompt_messages
        self.last_prompt_text = prompt_text
        self.last_prompt_image_paths = tuple(str(path) for path in prompt_image_paths)
        batch = self.processor(
            text=[prompt_text],
            images=all_images,
            **qwen3_vl_multimodal_processor_kwargs(padding=True),
        )
        self.last_native_image_size_px = self._native_coordinate_size_from_batch(batch)
        batch = self._sanitize_processor_batch(batch)
        device = getattr(self.model, "device", None)
        if device is not None:
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }

        try:
            import torch

            no_grad = torch.no_grad()
        except ImportError:
            no_grad = nullcontext()
        with no_grad:
            if self.prompt_style == "holo3_tool":
                output_ids = self._generate_holo3_structured(
                    batch,
                    allowed_kinds=allowed_kinds,
                )
            else:
                output_ids = self.model.generate(
                    **batch,
                    **self._generation_kwargs(),
                )
        input_len = batch["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        raw = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        assert transform is not None
        return raw, transform

    def _native_coordinate_size_from_batch(self, batch: Any) -> tuple[int, int] | None:
        if self.prompt_style not in {
            "uitars",
            "dart_gui",
            "opencua_pyautogui",
            "ui_venus",
        }:
            return None
        grid = batch.get("image_grid_thw") if hasattr(batch, "get") else None
        if grid is None:
            return None
        try:
            last_grid = grid[-1]
            values = last_grid.tolist() if hasattr(last_grid, "tolist") else list(last_grid)
            grid_height = int(values[-2])
            grid_width = int(values[-1])
        except (IndexError, TypeError, ValueError):
            return None
        image_processor = getattr(self.processor, "image_processor", None)
        patch_size = getattr(image_processor, "patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[-1]
        try:
            patch = int(patch_size)
        except (TypeError, ValueError):
            patch = 14
        if grid_width <= 0 or grid_height <= 0 or patch <= 0:
            return None
        return (grid_width * patch, grid_height * patch)

    def _sanitize_processor_batch(self, batch: Any) -> Any:
        if self.prompt_style != "opencua_pyautogui":
            return batch
        if "mm_token_type_ids" not in batch:
            return batch
        cleaned = dict(batch)
        cleaned.pop("mm_token_type_ids", None)
        return cleaned

    def _normalize_action(
        self,
        action: Action,
        _transform: ImageCoordinateTransform,
    ) -> Action:
        strict = self.prompt_style != "repo_json" and self.strict_native_output

        def normalize_coordinate(value: float) -> int:
            rounded = int(round(float(value)))
            if strict and not 0 <= rounded <= 1000:
                raise ValueError(f"canonical action coordinate is outside [0,1000]: {value!r}")
            return rounded if strict else min(1000, max(0, rounded))

        if isinstance(action, PrimitiveAction) and action.kind == "move_to":
            assert action.x is not None and action.y is not None
            return PrimitiveAction(
                kind="move_to",
                x=normalize_coordinate(float(action.x)),
                y=normalize_coordinate(float(action.y)),
            )
        if isinstance(action, AtomicAction):
            points = [
                (
                    normalize_coordinate(float(x)),
                    normalize_coordinate(float(y)),
                )
                for x, y in action.points
            ]
            return AtomicAction(kind=action.kind, points=points, answer=action.answer)
        if isinstance(action, WebAction) and action.points:
            points = [
                (
                    normalize_coordinate(float(x)),
                    normalize_coordinate(float(y)),
                )
                for x, y in action.points
            ]
            return WebAction(
                kind=action.kind,
                points=points,
                direction=action.direction,
                content=action.content,
                key=action.key,
                app=action.app,
                url=action.url,
            )
        return action

    def _write_call_log(
        self,
        *,
        obs: Observation,
        history: list[StepResult],
        allowed_kinds: Iterable[str],
        raw_prediction: str,
        parsed_action: Action | None,
        parse_error: str | None,
        budget: int | None,
        condition: str | None,
    ) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(self.checkpoint_path),
            "prompt_style": self.prompt_style,
            "history_mode": self._resolved_history_mode(),
            "instruction": obs.instruction,
            "screenshot_path": obs.screenshot_path,
            "history_length": len(history),
            "prompt_image_paths": list(self.last_prompt_image_paths),
            "prompt_image_count": len(self.last_prompt_image_paths),
            "native_image_size_px": list(self.last_native_image_size_px)
            if self.last_native_image_size_px is not None
            else None,
            "prompt_messages": self.last_prompt_messages,
            "allowed_kinds": list(allowed_kinds),
            "budget": budget,
            "condition": condition,
            "raw_prediction": raw_prediction,
            "parsed_action": parsed_action.to_dict() if parsed_action is not None else None,
            "parse_error": parse_error,
        }
        path = self.call_log_dir / f"call_{self._call_index:06d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action:
        self._ensure_loaded()
        self._call_index += 1
        self._reset_history_if_needed(history)
        current_image_path = Path(obs.screenshot_path)
        self.last_prediction_index = self._call_index
        self.last_raw_prediction = None
        self.last_think_text = None
        self.last_native_image_size_px = None
        raw_prediction = ""
        parsed_action: Action | None = None
        parse_error: str | None = None
        try:
            raw_prediction, transform = self._generate_raw(
                obs,
                history,
                allowed_kinds=allowed_kinds,
                budget=budget,
                condition=condition,
            )
            native_image_size_px = self.last_native_image_size_px
            if native_image_size_px is None:
                native_image_size_px = qwen25_native_coordinate_size(
                    self.prompt_style,
                    transform.model_size,
                )
            self.last_native_image_size_px = native_image_size_px
            action = normalize_baseline_output(
                raw_prediction,
                self.prompt_style,
                allowed_kinds=allowed_kinds,
                image_size_px=obs.size_px,
                native_image_size_px=native_image_size_px,
                current_cursor_xy=obs.cursor_xy,
                pyautogui_coordinate_mode=self._resolved_pyautogui_coordinate_mode(),
                strict=self.prompt_style != "repo_json" and self.strict_native_output,
            )
            parsed_action = self._normalize_action(action, transform)
            self.last_raw_prediction = raw_prediction
            self.last_think_text = _extract_thought_text(raw_prediction)
            if self._resolved_history_mode() != "single_current":
                assistant_content = native_assistant_history_content(
                    self.prompt_style,
                    raw_prediction,
                    parsed_action,
                )
                action_summary = native_history_action_summary(
                    self.prompt_style,
                    raw_prediction,
                    parsed_action,
                )
                self._history_image_paths.append(current_image_path)
                self._history_responses.append(raw_prediction)
                self._history_assistant_contents.append(assistant_content)
                self._history_action_summaries.append(action_summary)
                self._history_actions.append(parsed_action)
                self._history_observations.append(
                    Observation(
                        instruction=obs.instruction,
                        screenshot_path=obs.screenshot_path,
                        size_px=obs.size_px,
                        cursor_xy=obs.cursor_xy,
                        metadata=dict(obs.metadata),
                    )
                )
            return parsed_action
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            self.last_raw_prediction = raw_prediction or None
            raise
        finally:
            self._write_call_log(
                obs=obs,
                history=history,
                allowed_kinds=allowed_kinds,
                raw_prediction=raw_prediction,
                parsed_action=parsed_action,
                parse_error=parse_error,
                budget=budget,
                condition=condition,
            )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self._clear_native_history()
        self._last_history_length = -1
        self.last_native_image_size_px = None
        self._holo3_json_processor_templates.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _extract_thought_text(raw: str) -> str | None:
    thinking_match = re.search(
        r"<thinking>\s*(.*?)\s*</thinking>",
        raw,
        re.DOTALL,
    )
    if thinking_match and thinking_match.group(1).strip():
        return thinking_match.group(1).strip()
    if "</think>" in raw:
        thought = raw.split("</think>", 1)[0]
        if "<think>" in thought:
            thought = thought.rsplit("<think>", 1)[-1]
        return thought.strip() or None
    marker = "Action:"
    if raw.startswith("Thought:") and marker in raw:
        return raw[len("Thought:"):raw.index(marker)].strip() or None
    return None


class _TokenSuffixStoppingCriteria:
    def __init__(self, suffix_ids: Iterable[int]) -> None:
        self.suffix_ids = tuple(int(value) for value in suffix_ids)

    def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
        if not self.suffix_ids or input_ids.shape[-1] < len(self.suffix_ids):
            return False
        suffix = input_ids[..., -len(self.suffix_ids):]
        expected = suffix.new_tensor(self.suffix_ids)
        matches = suffix.eq(expected).all(dim=-1)
        return bool(matches.all().item())


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def _closed_holo3_reasoning_prefix(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        return cleaned.split("</think>", 1)[0].rstrip() + "\n</think>"
    return cleaned.rstrip() + ("\n" if cleaned else "") + "</think>"


def _append_text_token_ids(batch: Any, token_ids: Iterable[int]) -> dict[str, Any]:
    """Append generated text to a one-item multimodal processor batch."""

    suffix_values = tuple(int(value) for value in token_ids)
    if not suffix_values:
        raise RuntimeError("cannot start Holo3 structured decoding with an empty prefix")
    input_ids = batch["input_ids"]
    if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f"Holo3 constrained decoding expects one prompt, got {tuple(input_ids.shape)}"
        )
    suffix = input_ids.new_tensor([suffix_values])
    extended = dict(batch)
    extended["input_ids"] = _concatenate_tensors(input_ids, suffix)
    attention_mask = batch.get("attention_mask") if hasattr(batch, "get") else None
    if attention_mask is not None:
        attention_suffix = attention_mask.new_ones((attention_mask.shape[0], len(suffix_values)))
        extended["attention_mask"] = _concatenate_tensors(
            attention_mask,
            attention_suffix,
        )
    for key in ("token_type_ids", "mm_token_type_ids"):
        values = batch.get(key) if hasattr(batch, "get") else None
        if values is None or tuple(values.shape) != tuple(input_ids.shape):
            continue
        text_suffix = values.new_zeros((values.shape[0], len(suffix_values)))
        extended[key] = _concatenate_tensors(values, text_suffix)
    # Stale explicit positions no longer match the extended token sequence.  The
    # model can derive them from input_ids, attention_mask, and image_grid_thw.
    extended.pop("position_ids", None)
    extended.pop("cache_position", None)
    return extended


def _concatenate_tensors(left: Any, right: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Holo3 local generation requires torch") from exc
    return torch.cat((left, right), dim=-1)


def _opencua_vision_tensor(output: Any) -> Any:
    """Return the merged OpenCUA vision tensor across transformers versions."""

    for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    if isinstance(output, dict):
        for key in ("pooler_output", "image_embeds", "last_hidden_state"):
            value = output.get(key)
            if value is not None:
                return value
    if isinstance(output, (tuple, list)):
        if len(output) > 1 and output[1] is not None:
            return output[1]
        if output:
            return output[0]
    return output


def _patch_opencua_dynamic_cache_indexing() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "__getitem__"):
        return

    def _dynamic_cache_getitem(cache_self: Any, index: int) -> tuple[Any, ...]:
        layer = cache_self.layers[index]
        return (
            getattr(layer, "keys", None),
            getattr(layer, "values", None),
            getattr(layer, "_sliding_window_tensor", None),
        )

    _dynamic_cache_getitem._gui_captcha_compat = True  # type: ignore[attr-defined]
    setattr(DynamicCache, "__getitem__", _dynamic_cache_getitem)
