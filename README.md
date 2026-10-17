# Post-Training Calibration

Code for **Post-Training Shifts Confidence: A Three-Stage Analysis of How SFT, RL, and OPD Shape CoT Calibration**.

## Repository structure

- `SFT_training/`: supervised fine-tuning scripts.
- `RL_training/`: reinforcement learning training scripts.
- `OPD_training/`: on-policy distillation commands and modified TRL code.
- `PreCoT/`: pre-CoT confidence analysis and calibration metrics.
- `IntraAfterCoT/`: intra-CoT and after-CoT analysis with modified DeepConf code.

See the scripts and component READMEs for environment setup and usage. Adjust model, dataset, and output paths for your environment before running the experiments. Third-party components retain their included licenses.

## Citation

```bibtex
@misc{li2026posttrainingshiftsconfidencethreestage,
  title={Post-Training Shifts Confidence: A Three-Stage Analysis of How SFT, RL, and OPD Shape CoT Calibration},
  author={Shuhao Li and Guodong Du and Anhao Zhao and Wanyu Lin and Tianyu Yuan and Xiaoyu Shen},
  year={2026},
  eprint={2607.13753},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2607.13753},
}
```
