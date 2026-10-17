# Extended Benchmark Placeholders

This directory documents the expected local placement for the rebuttal's
additional benchmarks. The repository does not redistribute benchmark data.

Expected JSONL schema:

```json
{"question": "...", "answer": "...", "task": "gpqa_diamond"}
```

Recommended local filenames:

- `gpqa_diamond.jsonl`
- `livecodebench_v2.jsonl`
- `ifeval.jsonl`
- `halueval.jsonl`

The rebuttal reports Intra-CoT budget curves for GPQA-Diamond and
LiveCodeBench v2, and Post-CoT aggregation for all four extended benchmarks.
