"""Public benchmark manifests retain the environment and evaluation contracts."""
import json

import pytest

from gui_agent_captcha.benchmarks.exploration_depth.contracts import (
    PAPER_SUITE_ID, PAPER_VARIANTS, episodes_for_variant,
)
from gui_agent_captcha.custom_envs import build_benchmark_variant
from latentguiworld.evaluate import SUITE
from latentguiworld.suite import load_evaluation_manifest


@pytest.fixture
def paper_manifest(tmp_path):
    source = json.loads(SUITE.read_text())
    source["suite_id"] = PAPER_SUITE_ID
    inverse = {v: k for k, v in PAPER_VARIANTS.items()}
    for episode in source["episodes"]:
        episode["suite_id"] = PAPER_SUITE_ID
        episode["variant"] = inverse[episode["variant"]]
        episode.pop("difficulty", None)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source))
    return path


@pytest.mark.parametrize("name", PAPER_VARIANTS)
def test_public_names_load_and_build_environments(paper_manifest, name):
    original = paper_manifest.read_bytes()
    raw = json.loads(original)
    manifest = load_evaluation_manifest(paper_manifest)
    episodes = episodes_for_variant(manifest, name)
    expected = [e for e in raw["episodes"] if e["variant"] == name]
    assert len(episodes) == 150
    assert [e["shared_scene_config"] for e in episodes] == [e["shared_scene_config"] for e in expected]
    assert all(e["scene_variant"] == name and "difficulty" not in e for e in episodes)
    env = build_benchmark_variant(name, manifest_path=paper_manifest)
    assert len(env.list_task_ids()) == 150
    assert paper_manifest.read_bytes() == original


def test_mixed_public_suite_is_rejected(paper_manifest):
    manifest = json.loads(paper_manifest.read_text())
    manifest["episodes"][0]["suite_id"] = "another-benchmark"
    paper_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="suite IDs"):
        load_evaluation_manifest(paper_manifest)
