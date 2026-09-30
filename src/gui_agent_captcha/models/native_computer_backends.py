from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..core import Observation
from .anthropic_native_computer import (
    COMPUTER_TOOLSET_MEMBERS,
    AnthropicNativeComputerBackend,
)
from .anthropic_native_computer import (
    NATIVE_PROTOCOL_TRACK as ANTHROPIC_NATIVE_PROTOCOL_TRACK,
)
from .frontier_computer_use import FrontierModelConfig
from .gemini_native_computer import (
    BROWSER_NATIVE_ACTIONS,
    GEMINI_NATIVE_PROTOCOL_TRACK,
    GeminiNativeComputerConfig,
    GeminiNativeComputerUseBackend,
)
from .kimi_k3_computer import (
    KIMI_K3_CANONICAL_ACTION_KINDS,
    KIMI_K3_COMPUTER_USE_CONTRACT,
    KIMI_K3_HISTORY_STRATEGY,
    KIMI_K3_MODEL_TOOL_NAMES,
    KIMI_K3_PROMPT_VERSION,
    KIMI_K3_PROTOCOL_TRACK,
    KimiK3ComputerBackend,
    KimiK3ComputerConfig,
)
from .openai_native_computer import (
    OPENAI_COMPUTER_ACTION_KINDS,
    OPENAI_NATIVE_COMPUTER_PROTOCOL,
    OPENAI_NATIVE_HISTORY_STRATEGY,
    OpenAINativeComputerBackend,
    OpenAINativeComputerConfig,
)
from .openai_responses_six_action import (
    OPENAI_RESPONSES_SIX_ACTION_KINDS,
    OPENAI_RESPONSES_SIX_ACTION_PROMPT_VERSION,
)

NATIVE_CANONICAL_ACTION_KINDS = OPENAI_RESPONSES_SIX_ACTION_KINDS

OPENAI_NATIVE_ACTION_KINDS = OPENAI_COMPUTER_ACTION_KINDS

ANTHROPIC_NATIVE_ACTION_KINDS = COMPUTER_TOOLSET_MEMBERS
ANTHROPIC_NATIVE_DISABLED_ACTION_KINDS: tuple[str, ...] = ()
ANTHROPIC_NATIVE_TOOLSET_MEMBERS = COMPUTER_TOOLSET_MEMBERS

NATIVE_PROTOCOL_TRACKS = {
    "gpt56_sol": OPENAI_NATIVE_COMPUTER_PROTOCOL,
    "gemini36_flash": GEMINI_NATIVE_PROTOCOL_TRACK,
    "claude_opus5": ANTHROPIC_NATIVE_PROTOCOL_TRACK,
    "claude_sonnet5": ANTHROPIC_NATIVE_PROTOCOL_TRACK,
    "kimi_k3": KIMI_K3_PROTOCOL_TRACK,
}

NATIVE_MODEL_ACTION_KINDS = {
    "gpt56_sol": OPENAI_NATIVE_ACTION_KINDS,
    "gemini36_flash": BROWSER_NATIVE_ACTIONS,
    "claude_opus5": ANTHROPIC_NATIVE_ACTION_KINDS,
    "claude_sonnet5": ANTHROPIC_NATIVE_ACTION_KINDS,
    "kimi_k3": KIMI_K3_MODEL_TOOL_NAMES,
}

NATIVE_HISTORY_STRATEGIES = {
    "gpt56_sol": OPENAI_NATIVE_HISTORY_STRATEGY,
    "gemini36_flash": "previous_interaction_id_full_server_history",
    "claude_opus5": "native_full_structured_messages_all_screenshots",
    "claude_sonnet5": "native_full_structured_messages_all_screenshots",
    "kimi_k3": KIMI_K3_HISTORY_STRATEGY,
}

NATIVE_COORDINATE_FORMATS = {
    "gpt56_sol": "official_full_screenshot_pixels",
    "gemini36_flash": "native_0_999_official_1000_denominator",
    "claude_opus5": "official_full_screenshot_pixels",
    "claude_sonnet5": "official_full_screenshot_pixels",
    "kimi_k3": "benchmark_normalized_0_1000_image_coordinates",
}


def build_native_computer_backend(
    config: FrontierModelConfig,
    *,
    call_log_dir: Path | None = None,
    request_timeout_s: float = 180.0,
    request_retries: int = 6,
    observation_supplier: Callable[[], Observation] | None = None,
) -> Any:
    """Build the configured GUI backend for a frontier model."""

    if config.provider == "openai":
        return OpenAINativeComputerBackend(
            config=OpenAINativeComputerConfig(
                model=config.model,
                base_url=config.base_url,
                api_key=config.api_key,
            ),
            call_log_dir=call_log_dir,
            request_timeout_s=request_timeout_s,
            request_retries=request_retries,
            observation_supplier=observation_supplier,
        )
    if config.provider == "gemini":
        return GeminiNativeComputerUseBackend(
            config=GeminiNativeComputerConfig(
                model=config.model,
                base_url=config.base_url,
                api_key=config.api_key,
                api_key_env=config.api_key_env or "GEMINI_API_KEY",
            ),
            call_log_dir=call_log_dir,
            request_timeout_s=request_timeout_s,
            request_retries=request_retries,
            observation_supplier=observation_supplier,
        )
    if config.provider == "anthropic":
        return AnthropicNativeComputerBackend(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            system_prompt=None,
            call_log_dir=call_log_dir,
            request_timeout_s=request_timeout_s,
            request_retries=request_retries,
            observation_supplier=observation_supplier,
        )
    if config.provider == "kimi":
        return KimiK3ComputerBackend(
            config=KimiK3ComputerConfig(
                model=config.model,
                base_url=config.base_url,
                api_key=config.api_key,
            ),
            call_log_dir=call_log_dir,
            request_timeout_s=request_timeout_s,
            request_retries=request_retries,
        )
    raise ValueError(f"unsupported native computer provider: {config.provider!r}")


__all__ = [
    "ANTHROPIC_NATIVE_ACTION_KINDS",
    "ANTHROPIC_NATIVE_DISABLED_ACTION_KINDS",
    "ANTHROPIC_NATIVE_TOOLSET_MEMBERS",
    "KIMI_K3_CANONICAL_ACTION_KINDS",
    "KIMI_K3_COMPUTER_USE_CONTRACT",
    "KIMI_K3_MODEL_TOOL_NAMES",
    "KIMI_K3_PROMPT_VERSION",
    "NATIVE_CANONICAL_ACTION_KINDS",
    "NATIVE_COORDINATE_FORMATS",
    "NATIVE_HISTORY_STRATEGIES",
    "NATIVE_MODEL_ACTION_KINDS",
    "NATIVE_PROTOCOL_TRACKS",
    "OPENAI_NATIVE_ACTION_KINDS",
    "OPENAI_RESPONSES_SIX_ACTION_KINDS",
    "OPENAI_RESPONSES_SIX_ACTION_PROMPT_VERSION",
    "build_native_computer_backend",
]
