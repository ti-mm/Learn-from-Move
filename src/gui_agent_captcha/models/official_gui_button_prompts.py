"""Prompt additions tested in the bounded mouse-tool diagnostic."""

import json

from .official_gui_actions import current_position_click_action


def tool_output(style: str, action: str) -> str:
    if style in ("uitars", "dart_gui"):
        return f"Action: {action}()"
    return (
        "<tool_call>"
        + json.dumps({"name": "computer_use", "arguments": {"action": action}})
        + "</tool_call>"
    )


def guidance(style: str, mode: str) -> str:
    if mode == "examples":
        return ""
    click_action = current_position_click_action(style)
    reminder = (
        "\n\n## Additional mouse operations for this interface\n"
        "The following operations are executable, including those absent from pretraining. "
        "Do not substitute click or drag for a press-only or release-only request. "
        "mouse_down holds the left button across subsequent actions; mouse_up releases it. "
        f"{click_action} presses and releases once at the current cursor. These operations do not move the cursor. "
        "Use the existing independent movement operation to move while held. "
        "Return one action in the original output format.\n"
    )
    if style == "gui_owl":
        reminder += "left_click requires coordinate=[x,y] and moves there before clicking. left_click_current takes no arguments.\n"
    if mode == "reminder":
        return reminder + "\n".join(
            tool_output(style, a) for a in ("mouse_down", "mouse_up", click_action)
        )
    if mode != "few_shot":
        raise ValueError(mode)
    examples = []
    for request, action in [
        ("Press and keep holding at the current cursor.", "mouse_down"),
        ("Release the held button without moving.", "mouse_up"),
        ("Click once at the current cursor without moving.", click_action),
    ]:
        thought = (
            "执行请求的鼠标操作。"
            if style == "uitars"
            else "Perform the requested mouse operation."
        )
        answer = (
            ("Thought: " + thought + "\n")
            if style in ("uitars", "dart_gui")
            else "Action: " + request + "\n"
        )
        examples.append(
            "Example request: "
            + request
            + "\nExample response:\n"
            + answer
            + tool_output(style, action)
        )
    return (
        reminder
        + "\n\n".join(examples)
        + "\n\nNow answer the actual instruction using its screenshot.\n"
    )

