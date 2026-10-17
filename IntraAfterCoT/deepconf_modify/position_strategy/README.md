# Position-Aware Postprocess Strategy

This folder runs the paper-aligned PosConf strategy on existing offline PKL
rollouts from `examples/example_offline.py`.

## Paper Strategy

| Model | Post-CoT score window | Intra-CoT early-stop active range |
| --- | --- | --- |
| Qwen-SFT | `[0.60, 1.00]` | `[0.00, 1.00]` |
| Qwen-RL | `[0.60, 1.00]` | `[0.20, 1.00]` |
| Qwen-OPD | `[0.00, 0.60]` | `[0.00, 0.60]` |
| Llama-SFT | `[0.55, 1.00]` | `[0.00, 1.00]` |
| Llama-RL | `[0.55, 1.00]` | `[0.25, 1.00]` |
| Llama-OPD | `[0.00, 0.55]` | `[0.00, 0.55]` |

The online simulator uses `W=32`, `N=256`, `M=50`, `R=5`, and adaptive divisor
`D=10` by default. Relative position for online stopping is estimated with the
median warmup trace length, then clamped to `1.0`, so final trace length is not
used before the trace finishes.

## Run The Batch

From `code/IntraAfterCoT/deepconf_modify`:

```bash
PKL_ROOT=/path/to/offline_pkls \
SFT_MODEL=/path/to/qwen_sft \
RL_MODEL=/path/to/qwen_rl \
OPD_MODEL=/path/to/qwen_opd \
  bash position_strategy/postprocess_all_position_strategy_v3.sh
```

`PKL_ROOT` should contain:

```text
sft_aime24/ sft_aime25/ sft_amc23/ sft_math500/
rl_aime24/  rl_aime25/  rl_amc23/  rl_math500/
opd_aime24/ opd_aime25/ opd_amc23/ opd_math500/
```

Each job writes reports under
`position_strategy/runs/position_strategy_batch_<timestamp>/<job>/reports/`.

## Single Job

```bash
PKL_DIR=/path/to/offline_pkls/rl_aime25 \
MODEL_FAMILY=rl \
MODEL=/path/to/qwen_rl \
DATASET=examples/aime_2025.jsonl \
OUT_ROOT=runs/position_strategy_results/rl_aime25 \
  bash position_strategy/postprocess_strategy_single_model.sh
```

For Llama, set `MODEL_FAMILY` to `llama_sft`, `llama_rl`, or `llama_opd`.
For the Qwen-explicit aliases, use `qwen_sft`, `qwen_rl`, or `qwen_opd`.

Use `POST_START_FRACTION`, `POST_END_FRACTION`, `INTRA_START_FRACTION`, and
`INTRA_END_FRACTION` only for ablations; leaving them unset uses the paper
settings above.
