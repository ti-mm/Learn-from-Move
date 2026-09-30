# Evaluation adapters

The adapters connect model prompts, history, output parsing, and coordinate
conventions to the six environment implementations. All paths below are relative
to `src/gui_agent_captcha/`.

| Interface | Implementation | Role |
|---|---|---|
| LatentLearner, local checkpoint or vLLM | `models/exploration_depth_with_think.py`, `models/exploration_depth_vllm.py` | Think/action parsing and three-screenshot history |
| DART-GUI, UI-TARS, GUI-Owl, EvoCUA, Holo3.1, OpenCUA | `eval/exploration_depth_open_agents.py`, `models/official_gui.py`, `models/official_gui_actions.py` | Model-specific prompts, history, action parsing, and coordinate conversion |
| OpenAI, Anthropic, Gemini, Kimi interfaces | `models/native_computer_backends.py` and provider modules | Request/response formats and tool-call history |
| Python/PyAutoGUI execution | `models/openai_code_execution_computer.py`, `envs/rotation_code_execution.py` | Persistent execution state and mouse/screenshot operations |
| ScreenSpot-Pro, ScreenSpot-v2, MMBench-GUI, UI-Vision, OSWorld-G | `eval/qwen3vl_paired90_five_bench.py` | Dataset loading, direct-click and move-then-click evaluation |

`envs/official_gui.py` executes parsed GUI actions; `envs/anthropic_computer.py`
maps structured computer tool calls to environment actions.
`eval/exploration_depth_open_agents.py` implements the egocentric click/drag
mapping with sensitivity-aware cursor movement. Coordinate conversion follows
each model's protocol. Evaluation ends at the first terminal release.

The public `latentguiworld-eval` launcher selects the LatentLearner interface.
The other model runners have their own module entry points and arguments:

```bash
python -m gui_agent_captcha.eval.exploration_depth_open_agents --help
python -m gui_agent_captcha.eval.exploration_depth_astra --help
python -m gui_agent_captcha.eval.exploration_depth_opus --help
```

## Runtime endpoints

Supply deployment endpoints when launching evaluation. The portable example uses
a local server; a remote deployment can set the same shell variable externally.

```bash
export MODEL_SERVER_URL=http://localhost:8000/v1
latentguiworld-eval --model models/latentlearner \
  --server-url "$MODEL_SERVER_URL" --served-model latentlearner \
  --output results/latentlearner
```

The API runners read `GPT6_ASTRA_BASE_URL`, `GPT56_SOL_BASE_URL`, or
`CLAUDE_OPUS5_BASE_URL` for their corresponding profiles. Set API credentials
through `OPENAI_API_KEY` or the provider-specific environment variable used by
the runner. Local environment servers bind to loopback addresses.

Set `--output` to choose where the evaluation launcher saves its metrics.

## Ablation evaluation

Use the paper's benchmark environments to evaluate the main model and ablation
checkpoints:

```bash
latentguiworld-eval --model models/paradigm-ablation --output results/paradigm-ablation
latentguiworld-eval --model models/data-ablation --output results/data-ablation
```

For prompt ablation, pass the benchmark manifest as the positional argument to
`gui_agent_captcha.eval.exploration_depth_astra`; select the baseline or prompted
profile with `--profile`, and use `--limit-per-variant 50` for both conditions.
Use `default_formal_manifest_path()` to locate the manifest, as shown in
[data interfaces](data.md#benchmark-scenes). The five grounding datasets use
the loaders and layouts described on that page.
