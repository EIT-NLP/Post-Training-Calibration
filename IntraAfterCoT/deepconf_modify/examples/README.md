# DeepConf Examples

Use `example_offline.py` to collect paper rollouts and `analyze_sft_deepconf.py`
to run the base Intra/Post-CoT post-hoc analysis.

## Offline Rollouts

```bash
python example_offline.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model_type qwen \
  --dataset aime_2024.jsonl \
  --qid_start 0 --qid_end 29 \
  --budget 320 \
  --max_tokens 32768 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 0 \
  --rid seed0 \
  --output_dir runs/offline_pkls/instruct_aime24
```

Repeat with `rid=seed0` through `seed4` for the five-run protocol.

## Base DeepConf Postprocess

```bash
python analyze_sft_deepconf.py \
  --results_dir runs/offline_pkls/sft_aime24 \
  --dataset aime_2024.jsonl \
  --online_warmup_traces 32 \
  --online_total_budget 288 \
  --online_sweep_points 50 \
  --online_resamples 5 \
  --offline_sample_size 256 \
  --offline_resamples 5 \
  --adaptive_divisor 10 \
  --output_json runs/posthoc/sft_aime24_full.json \
  --dataset_output_json runs/posthoc/sft_aime24_dataset.json
```

For PosConf, use `../position_strategy/README.md`.
