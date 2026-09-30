"""Shared frozen-v10 evaluation contract for benchmark and ablation runners."""
from collections import Counter

from gui_agent_captcha.benchmarks.exploration_depth.contracts import (
    EXPLORATION_BENCHMARK_VARIANTS,
    FORMAL_SUITE_ID,
    load_manifest,
)


def load_evaluation_manifest(path=None):
    manifest = load_manifest(path)
    if manifest.get("suite_id") != FORMAL_SUITE_ID or manifest.get("frozen") is not True:
        raise ValueError("evaluation requires the frozen v10 suite")
    episodes = manifest.get("episodes", [])
    expected = {variant.key: 150 for variant in EXPLORATION_BENCHMARK_VARIANTS}
    if Counter(episode["variant"] for episode in episodes) != expected:
        raise ValueError("v10 evaluation requires 150 episodes per variant")
    if len({episode["episode_id"] for episode in episodes}) != 900:
        raise ValueError("v10 evaluation requires 900 unique episode IDs")
    if any(episode.get("suite_id") != FORMAL_SUITE_ID for episode in episodes):
        raise ValueError("episode suite IDs must match the v10 manifest")
    return manifest
