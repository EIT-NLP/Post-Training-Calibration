# Minimal TRL Runtime for OPD

This directory vendors the minimal TRL runtime used by `train_gold.py` for the
paper's OPD experiments. It keeps the Apache-2.0 license and runtime modules
needed for:

- `GOLDConfig` and `GOLDTrainer`
- `SFTTrainer`-based model training utilities
- `trl.scripts.vllm_serve` for the vLLM generation server
- DeepSpeed/Accelerate config files used by the command examples

Upstream TRL documentation, notebooks, tests, CI workflows, Docker files, and
unrelated trainer examples were removed from this release to avoid confusing
reviewers about the paper setting.

The paper setting is full-parameter OPD. `train_gold.py` force-disables PEFT and
quantized adapter flags even if stale command-line flags are passed.
