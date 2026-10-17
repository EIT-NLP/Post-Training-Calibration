# OPD Training Data

The OPD launcher expects student-generated rollout data passed through
`--dataset_name` or the `OPD_DATASET` environment variable. The full 48k
training JSONL is not committed because it is a generated experimental artifact
and is easier to audit as an explicit local input.

Expected local formats are `.json`, `.jsonl`, `.parquet`, or a directory
containing those files. See `OPD_training/command.txt` and
`OPD_training/command_llama.txt` for launch examples.
