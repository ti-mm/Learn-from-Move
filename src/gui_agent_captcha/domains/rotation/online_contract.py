from __future__ import annotations

SHENYIZE_ONLINE_VIEWPORT: tuple[int, int] = (1280, 720)
SHENYIZE_ONLINE_IMAGE_MAX_PIXELS = 1280 * 720
SHENYIZE_ONLINE_SFT_CHECKPOINT = (
    "checkpoints/migrated-20260903/source-artifacts-checkpoints/"
    "qwen35-9b-verl-sft-rotation-teacherthink-english-h200x8-hd720-"
    "hist3-fullhistory-nosystem-ctx16k-mbs1-bs64-ep3-save100-"
    "selected-model"
)
SHENYIZE_ONLINE_MM_PROCESSOR_KWARGS: dict[str, int] = {
    "max_pixels": SHENYIZE_ONLINE_IMAGE_MAX_PIXELS,
}
