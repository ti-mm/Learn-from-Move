"""Upstream prompt fixtures; sources and local changes are recorded in PROMPTS.

UI-TARS / DART: Copyright ByteDance and DART authors, Apache-2.0.
Other prompt copyrights remain with their respective upstream authors.
"""

PROMPTS = {
    "sources": {
        "dart_gui": {
            "url": "https://github.com/computer-use-agents/dart-gui/blob/e0e606d3082c8ebec296c4dd4b8e01ecde94edd0/mm_agents/prompts.py",
            "license": "Apache-2.0",
            "retrieved_at": "2026-09-07",
            "modifications": [],
        },
        "uitars": {
            "url": "https://github.com/bytedance/UI-TARS/blob/582f3a7ea5d285ee8ed9e2e84048d1ab01453c49/data/test_messages.json",
            "license": "Apache-2.0",
            "retrieved_at": "2026-09-07",
            "modifications": [],
        },
        "gui_owl": {
            "url": "https://github.com/X-PLUG/MobileAgent/blob/11cea575561fb7800b5fb6b6cafa56f7a91de11f/Mobile-Agent-v3.5/cookbook/end2end_usage_computer.ipynb",
            "license": "MIT",
            "retrieved_at": "2026-09-07",
            "modifications": [],
        },
        "evocua_s2": {
            "url": "https://github.com/meituan/EvoCUA/blob/4a0ad5fd4eb1d5b65966e1c7cc3feaa3b534eadd/mm_agents/evocua/prompts.py",
            "license": "Apache-2.0",
            "retrieved_at": "2026-09-07",
            "modifications": [],
        },
        "opencua_pyautogui": {
            "url": "https://github.com/xlang-ai/OpenCUA/blob/dfc91ba89f700d10f26ec50362d308571482ab8b/evaluation/agentnetbench/agent/opencua.py",
            "license": "MIT",
            "retrieved_at": "2026-09-07",
            "modifications": [],
        },
        "holo3_tool": {
            "url": "https://hub.hcompany.ai/agent-loop",
            "checkpoint_template": "Hcompany/Holo-3.1-35B-A3B/chat_template.jinja",
            "retrieved_at": "2026-09-07",
            "modifications": [
                "Harness-defined six mouse tools plus official "
                "click/answer/write examples; no universal desktop "
                "tool list is specified by the guide."
            ],
        },
    },
    "dart_gui": "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task.\n"
    "\n"
    "## Output Format\n"
    "```\n"
    "Thought: ...\n"
    "Action: ...\n"
    "```\n"
    "\n"
    "## Action Space\n"
    "\n"
    "click(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
    "left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
    "right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
    "drag(start_box='<|box_start|>(x1,y1)<|box_end|>', "
    "end_box='<|box_start|>(x3,y3)<|box_end|>')\n"
    "hotkey(key='')\n"
    "type(content='') #If you want to submit your input, use \"\\n\" at the end of "
    "`content`.\n"
    "scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right "
    "or left')\n"
    "wait() #Sleep for 5s and take a screenshot to check for any changes.\n"
    "finished(content='xxx') # Use escape characters \\', \\\", and \\n in content part "
    "to ensure we can parse the content in normal python string format.\n"
    "\n"
    "## Note\n"
    "- Use {language} in `Thought` part.\n"
    "- Write a small plan and finally summarize your next action (with its target "
    "element) in one sentence in `Thought` part.\n"
    "- My computer's password is 'password', feel free to use it when you need sudo "
    "rights.\n"
    "\n"
    "## User Instruction\n"
    "{instruction}\n",
    "uitars": "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task. \n"
    "\n"
    "## Output Format\n"
    "```\n"
    "Thought: ...\n"
    "Action: ...\n"
    "```\n"
    "\n"
    "## Action Space\n"
    "\n"
    "click(start_box='<|box_start|>(x1, y1)<|box_end|>')\n"
    "left_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\n"
    "right_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\n"
    "drag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, "
    "y3)<|box_end|>')\n"
    "hotkey(key='')\n"
    "type(content='') #If you want to submit your input, use \"\\n\" at the end of "
    "`content`.\n"
    "scroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or "
    "left')\n"
    "wait() #Sleep for 5s and take a screenshot to check for any changes.\n"
    "finished(content='xxx') # Use escape characters \\', \\\", and \\n in content part "
    "to ensure we can parse the content in normal python string format.\n"
    "\n"
    "\n"
    "## Note\n"
    "- Use Chinese in `Thought` part.\n"
    "- Write a small plan and finally summarize your next action (with its target element) "
    "in one sentence in `Thought` part.\n"
    "\n"
    "## User Instruction\n"
    "{instruction}",
    "gui_owl": "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse '
    "and keyboard to interact with a computer, and take screenshots.\n"
    "* This is an interface to a desktop GUI. You do not have access to a terminal or "
    "applications menu. You must click on desktop icons to start applications.\n"
    "* Some applications may take time to start or process actions, so you may need to "
    "wait and take successive screenshots to see the results of your actions. E.g. if you "
    "click on Firefox and a window doesn't open, try wait and taking another screenshot.\n"
    "* The screen's resolution is 1000x1000.\n"
    "* Whenever you intend to move the cursor to click on an element like an icon, you "
    "should consult a screenshot to determine the coordinates of the element before moving "
    "the cursor.\n"
    "* If you tried clicking on a program or link but it failed to load, even after "
    "waiting, try adjusting your cursor position so that the tip of the cursor visually "
    "falls on the element that you want to click.\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center "
    'of the element. Don\'t click boxes on their edges unless asked.", "parameters": '
    '{"properties": {"action": {"description": "The action to perform. The available '
    "actions are:\n"
    "* `key`: Performs key down presses on the arguments passed in order, then performs "
    "key releases in reverse order.\n"
    "* `type`: Type a string of text on the keyboard.\n"
    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the "
    "screen.\n"
    "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on "
    "the screen.\n"
    "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate "
    "on the screen.\n"
    "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate "
    "on the screen.\n"
    "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel "
    "coordinate on the screen.\n"
    "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel "
    "coordinate on the screen.\n"
    "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel "
    "coordinate on the screen (simulated as double-click since it's the closest action).\n"
    "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
    "* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\n"
    "* `wait`: Wait specified seconds for the change to happen.\n"
    "* `interact`: Resolve the blocking window by interacting with the user.\n"
    "* `terminate`: Terminate the current task and report its completion status.\n"
    '* `answer`: Answer a question.", "enum": ["key", "type", "mouse_move", "left_click", '
    '"left_click_drag", "right_click", "middle_click", "double_click", "triple_click", '
    '"scroll", "hscroll", "wait", "interact", "terminate", "answer"], "type": "string"}, '
    '"keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": '
    '{"description": "Required only by `action=type`, `action=interact`, and '
    '`action=answer`.", "type": "string"}, "coordinate": {"description": "(x, y): The x '
    "(pixels from the left edge) and y (pixels from the top edge) coordinates to move the "
    'mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", '
    '"type": "array"}, "pixels": {"description": "The amount of scrolling to perform. '
    "Positive values scroll up, negative values scroll down. Required only by "
    '`action=scroll` and `action=hscroll`.", "type": "number"}, "time": {"description": '
    '"The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": '
    '{"description": "The status of the task. Required only by `action=terminate`.", '
    '"type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": '
    '"object"}}}\n'
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
    "\n"
    "# Response format\n"
    "\n"
    "Response format for every step:\n"
    "1) Action: a short imperative describing what to do in the UI.\n"
    '2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": '
    '<function-name>, "arguments": <args-json-object>}.\n'
    "\n"
    "Rules:\n"
    "- Output exactly in the order: Action, <tool_call>.\n"
    "- Be brief: one for Action.\n"
    "- Do not output anything else outside those two parts.\n"
    "- If finishing, use action=terminate in the tool call.\n",
    "evocua_s2": "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name_for_human": "computer_use", "name": '
    '"computer_use", "description": "Use a mouse and keyboard to interact with a '
    "computer, and take screenshots.\\n* This is an interface to a desktop GUI. You must "
    "click on desktop icons to start applications.\\n* Some applications may take time "
    "to start or process actions, so you may need to wait and take successive "
    "screenshots to see the results of your actions. E.g. if you click on Firefox and a "
    "window doesn't open, try wait and taking another screenshot.\\n* The screen's "
    "resolution is 1000x1000.\\n* Whenever you intend to move the cursor to click on an "
    "element like an icon, you should consult a screenshot to determine the coordinates "
    "of the element before moving the cursor.\\n* If you tried clicking on a program or "
    "link but it failed to load even after waiting, try adjusting your cursor position "
    "so that the tip of the cursor visually falls on the element that you want to "
    "click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in "
    "the center of the element. Don't click boxes on their edges unless asked.\", "
    '"parameters": {"properties": {"action": {"description": "\\n* `key`: Performs key '
    "down presses on the arguments passed in order, then performs key releases in "
    "reverse order.\\n* `key_down`: Press and HOLD the specified key(s) down in order "
    "(no release). Use this for stateful holds like holding Shift while clicking.\\n* "
    "`key_up`: Release the specified key(s) in reverse order.\\n* `type`: Type a string "
    "of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) "
    "pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button at a "
    "specified (x, y) pixel coordinate on the screen.\\n* `left_click_drag`: Click and "
    "drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* "
    "`right_click`: Click the right mouse button at a specified (x, y) pixel coordinate "
    "on the screen.\\n* `middle_click`: Click the middle mouse button at a specified (x, "
    "y) pixel coordinate on the screen.\\n* `double_click`: Double-click the left mouse "
    "button at a specified (x, y) pixel coordinate on the screen.\\n* `triple_click`: "
    "Triple-click the left mouse button at a specified (x, y) pixel coordinate on the "
    "screen.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `hscroll`: "
    "Performs a horizontal scroll (mapped to regular scroll).\\n* `wait`: Wait specified "
    "seconds for the change to happen.\\n* `terminate`: Terminate the current task and "
    'report its completion status.\\n* `answer`: Answer a question.\\n", "enum": ["key", '
    '"type", "mouse_move", "left_click", "left_click_drag", "right_click", '
    '"middle_click", "double_click", "triple_click", "scroll", "wait", "terminate", '
    '"key_down", "key_up"], "type": "string"}, "keys": {"description": "Required only by '
    '`action=key`.", "type": "array"}, "text": {"description": "Required only by '
    '`action=type`.", "type": "string"}, "coordinate": {"description": "The x,y '
    'coordinates for mouse actions.", "type": "array"}, "pixels": {"description": "The '
    'amount of scrolling.", "type": "number"}, "time": {"description": "The seconds to '
    'wait.", "type": "number"}, "status": {"description": "The status of the task.", '
    '"type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": '
    '"object"}, "args_format": "Format the arguments as a JSON object."}}\n'
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
    "\n"
    "# Response format\n"
    "\n"
    "Response format for every step:\n"
    "1) Action: a short imperative describing what to do in the UI.\n"
    '2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": '
    '<function-name>, "arguments": <args-json-object>}.\n'
    "\n"
    "Rules:\n"
    "- Output exactly in the order: Action, <tool_call>.\n"
    "- Be brief: one sentence for Action.\n"
    "- Do not output anything else outside those parts.\n"
    "- If finishing, use action=terminate in the tool call.",
    "opencua_pyautogui": "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task.\n"
    "\n"
    "For each step, provide your response in this format:\n"
    "\n"
    "Thought:\n"
    "  - Step by Step Progress Assessment:\n"
    "    - Analyze completed task parts and their contribution to the overall "
    "goal\n"
    "    - Reflect on potential errors, unexpected results, or obstacles\n"
    "    - If previous action was incorrect, predict a logical recovery step\n"
    "  - Next Action Analysis:\n"
    "    - List possible next actions based on current state\n"
    "    - Evaluate options considering current state and previous actions\n"
    "    - Propose most logical next action\n"
    "    - Anticipate consequences of the proposed action\n"
    "  - For Text Input Actions:\n"
    "    - Note current cursor position\n"
    "    - Consolidate repetitive actions (specify count for multiple "
    "keypresses)\n"
    "    - Describe expected final text outcome\n"
    "    - Use first-person perspective in reasoning\n"
    "\n"
    "Action:\n"
    "  Provide clear, concise, and actionable instructions:\n"
    "  - If the action involves interacting with a specific target:\n"
    "    - Describe target explicitly without using coordinates\n"
    "    - Specify element names when possible (use original language if "
    "non-English)\n"
    "    - Describe features (shape, color, position) if name unavailable\n"
    '    - For window control buttons, identify correctly (minimize "—", '
    'maximize "□", close "X")\n'
    "  - if the action involves keyboard actions like 'press', 'write', "
    "'hotkey':\n"
    "    - Consolidate repetitive keypresses with count\n"
    "    - Specify expected text outcome for typing actions\n"
    "\n"
    "Finally, output the action as PyAutoGUI code or the following functions:\n"
    '- {"name": "computer.triple_click", "description": "Triple click on the '
    'screen", "parameters": {"type": "object", "properties": {"x": {"type": '
    '"number", "description": "The x coordinate of the triple click"}, "y": '
    '{"type": "number", "description": "The y coordinate of the triple click"}}, '
    '"required": ["x", "y"]}}\n'
    '- {"name": "computer.terminate", "description": "Terminate the current task '
    'and report its completion status", "parameters": {"type": "object", '
    '"properties": {"status": {"type": "string", "enum": ["success", "failure"], '
    '"description": "The status of the task"}}, "required": ["status"]}}',
}

# License notices for the verbatim upstream prompt fixtures above.
UPSTREAM_LICENSE_NOTICES = {
    "OpenCUA_LICENSE": 'MIT License\n\nCopyright (c) 2025 The OpenCUA Authors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n',
    "MobileAgent_LICENSE": 'MIT License\n\nCopyright (c) 2022 mPLUG\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n',
    "EvoCUA_LICENSE": '                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      "License" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      "Licensor" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      "Legal Entity" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      "control" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      "You" (or "Your") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      "Source" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      "Object" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      "Work" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      "Derivative Works" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      "Contribution" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, "submitted"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as "Not a Contribution."\n\n      "Contributor" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a "NOTICE" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an "AS IS" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets "[]"\n      replaced with your own identifying information. (Don\'t include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same "printed page" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright 2024 XLANG NLP Lab\n\n   Licensed under the Apache License, Version 2.0 (the "License");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an "AS IS" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n',
    "dart-gui_LICENSE": '\n                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      "License" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      "Licensor" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      "Legal Entity" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      "control" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      "You" (or "Your") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      "Source" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      "Object" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      "Work" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      "Derivative Works" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      "Contribution" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, "submitted"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as "Not a Contribution."\n\n      "Contributor" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a "NOTICE" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an "AS IS" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets "[]"\n      replaced with your own identifying information. (Don\'t include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same "printed page" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the "License");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an "AS IS" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n',
}
