"""UI-Venus-2 Computer requests through a local vLLM service."""

from __future__ import annotations

import copy
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..train.qwen3_vl_sft import ImageCoordinateTransform
from .official_gui import OfficialGuiBackend
from .ui_venus2_official import build_messages


@dataclass
class Venus2Backend(OfficialGuiBackend):
    prompt_style: str = "ui_venus2_computer"
    history_mode: str = "ui_venus2_computer"
    max_new_tokens: int = 8192

    def _ensure_loaded(self) -> None:
        pass

    def unload(self) -> None:
        self._clear_native_history()

    def _generate_raw(self, obs, history, **kwargs):
        accepted = [
            {"image": path, "accepted_response": response}
            for path, response in zip(self._history_image_paths, self._history_responses)
        ]
        messages = build_messages(obs.instruction, accepted, obs.screenshot_path, n_img=2)
        paths = [*self._history_image_paths[-2:], Path(obs.screenshot_path)]
        logged = copy.deepcopy(messages)
        image_index = 0
        for message in logged:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "image_url":
                        item["image_url"]["url"] = str(paths[image_index])
                        image_index += 1
        self.last_prompt_messages = logged
        self.last_prompt_image_paths = tuple(str(path) for path in paths)
        payload: dict[str, Any] = {
            "model": "ui-venus2",
            "messages": messages,
            "temperature": 1.0,
            "top_p": 0.7,
            "max_tokens": self.max_new_tokens,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        base_url = os.environ["VENUS2_BASE_URL"].rstrip("/")
        request = urllib.request.Request(
            base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        choice = result["choices"][0]
        message = choice["message"]
        raw = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning and "<think>" not in raw:
            raw = f"<think>\n{reasoning.strip()}\n</think>\n{raw.strip()}"
        self.last_raw_prediction = raw
        if self.call_log_dir is not None:
            self.call_log_dir.mkdir(parents=True, exist_ok=True)
            transport = {k: v for k, v in payload.items() if k != "messages"}
            transport.update(response=result, messages=logged)
            (self.call_log_dir / f"api_{self._call_index:06d}.json").write_text(
                json.dumps(transport, ensure_ascii=False, indent=2) + "\n"
            )
        if choice.get("finish_reason") != "stop":
            from .official_gui_actions import OfficialGuiProtocolError

            raise OfficialGuiProtocolError(f"incomplete model output: {choice.get('finish_reason')}")
        with Image.open(obs.screenshot_path) as image:
            size = image.size
        return raw, ImageCoordinateTransform(size, size)
