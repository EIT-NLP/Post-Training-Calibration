#!/usr/bin/env python
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
On-policy GOLD distillation launcher.

Default setup in this script:
- Student: Qwen/Qwen2.5-7B-Instruct
- Teacher: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- Optimization: full-parameter training, not LoRA/QLoRA
- On-policy distillation: lmbda=1.0
- Cross-tokenizer distillation enabled: use_uld_loss=True

Example:
accelerate launch --config_file ds_zero2_3gpu.yaml train_gold.py \
  --dataset_name /path/to/student_generated_math_rollouts \
  --dataset_train_split train \
  --output_dir outputs/qwen25_7b_gold_onpolicy \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --generation_batch_size 1 \
  --max_length 32768 \
  --max_completion_length 16384 \
  --learning_rate 1e-6 \
  --save_steps 100 \
  --logging_steps 1 \
  --uld_distillation_weight 1.0 \
  --uld_crossentropy_weight 0.0 \
  --uld_use_hybrid_loss false \
  --beta 0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig, GOLDTrainer


logger = logging.getLogger(__name__)

DEFAULT_STUDENT = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEACHER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


def _load_dataset(script_args: ScriptArguments):
    if not script_args.dataset_name:
        raise ValueError(
            "Please provide --dataset_name. It can be a HF dataset name, or a local .json/.jsonl/.parquet file path."
        )

    dataset_name = script_args.dataset_name
    local_path = Path(dataset_name)

    if local_path.is_file():
        suffix = local_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            builder = "json"
        elif suffix == ".parquet":
            builder = "parquet"
        else:
            raise ValueError(
                f"Unsupported local dataset file extension: {suffix}. "
                "Supported: .json, .jsonl, .parquet"
            )
        return load_dataset(builder, data_files={"train": str(local_path)})

    if local_path.is_dir():
        parquet_files = sorted(str(p) for p in local_path.rglob("*.parquet"))
        jsonl_files = sorted(str(p) for p in local_path.rglob("*.jsonl"))
        json_files = sorted(str(p) for p in local_path.rglob("*.json"))

        if parquet_files:
            return load_dataset("parquet", data_files={"train": parquet_files})
        if jsonl_files or json_files:
            return load_dataset("json", data_files={"train": jsonl_files + json_files})

        raise ValueError(
            f"No supported dataset files found under directory: {local_path}. "
            "Expected at least one .parquet/.json/.jsonl file."
        )

    return load_dataset(
        dataset_name,
        name=script_args.dataset_config,
        streaming=script_args.dataset_streaming,
    )


def _as_messages(prompt: Any, completion: Any) -> list[dict[str, str]]:
    def _normalize_turns(value: Any, default_role: str) -> list[dict[str, str]]:
        if isinstance(value, list):
            turns: list[dict[str, str]] = []
            for item in value:
                if isinstance(item, dict) and "role" in item and "content" in item:
                    turns.append({"role": str(item["role"]), "content": str(item["content"])})
                else:
                    turns.append({"role": default_role, "content": str(item)})
            return turns

        if isinstance(value, dict):
            if "role" in value and "content" in value:
                return [{"role": str(value["role"]), "content": str(value["content"])}]
            if "from" in value and "value" in value:
                role = "assistant" if str(value["from"]).lower() in {"assistant", "gpt"} else "user"
                return [{"role": role, "content": str(value["value"])}]

        return [{"role": default_role, "content": str(value)}]

    messages = _normalize_turns(prompt, "user")
    messages.extend(_normalize_turns(completion, "assistant"))
    return messages


def _convert_conversations_to_messages(example: dict[str, Any]) -> dict[str, Any]:
    conversations = example["conversations"]
    messages: list[dict[str, str]] = []
    if not isinstance(conversations, list):
        raise ValueError("The `conversations` column must be a list of turns.")

    for turn in conversations:
        if isinstance(turn, dict):
            if "role" in turn and "content" in turn:
                role = str(turn["role"])
                content = str(turn["content"])
            elif "from" in turn and "value" in turn:
                role = "assistant" if str(turn["from"]).lower() in {"assistant", "gpt"} else "user"
                content = str(turn["value"])
            else:
                raise ValueError(
                    "Unsupported conversation turn format. Expected {'role','content'} or {'from','value'}."
                )
            messages.append({"role": role, "content": content})
        else:
            raise ValueError("Each item in `conversations` must be a dict.")

    return {"messages": messages}


def _ensure_messages(dataset: Dataset | IterableDataset):
    column_names = set(dataset.column_names or [])

    if "messages" in column_names:
        return dataset

    if {"prompt", "completion"}.issubset(column_names):
        return dataset.map(lambda ex: {"messages": _as_messages(ex["prompt"], ex["completion"])})

    if {"question", "golden_answer"}.issubset(column_names):
        return dataset.map(
            lambda ex: {
                "messages": [
                    {"role": "user", "content": str(ex["question"])},
                    {"role": "assistant", "content": str(ex["golden_answer"])},
                ]
            }
        )

    if "conversations" in column_names:
        return dataset.map(_convert_conversations_to_messages)

    raise ValueError(
        "Dataset must include one of: `messages`, (`prompt` and `completion`), "
        "(`question` and `golden_answer`), or `conversations`. "
        f"Found columns: {sorted(column_names)}"
    )


def _pick_split(
    dataset: Dataset | IterableDataset | DatasetDict | IterableDatasetDict,
    split_name: str,
    purpose: str,
) -> Dataset | IterableDataset:
    if isinstance(dataset, (Dataset, IterableDataset)):
        return dataset

    if split_name in dataset:
        return dataset[split_name]

    available_splits = ", ".join(dataset.keys())
    raise ValueError(
        f"Could not find {purpose} split `{split_name}`. Available splits: {available_splits}"
    )


def _auto_split_train_eval(
    train_dataset: Dataset | IterableDataset,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset | IterableDataset, Dataset | IterableDataset]:
    if isinstance(train_dataset, Dataset):
        split = train_dataset.train_test_split(test_size=eval_ratio, shuffle=True, seed=seed)
        logger.warning(
            "No eval split found. Auto-splitting %.2f%% of train as eval. "
            "New train size=%d, eval size=%d",
            eval_ratio * 100,
            len(split["train"]),
            len(split["test"]),
        )
        return split["train"], split["test"]

    raise ValueError(
        "No eval split found and the train split is an IterableDataset, which cannot be auto split by ratio. "
        "Please provide an explicit eval split or set --eval_strategy no."
    )


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Hardcode this script as on-policy GOLD for the target pair.
    if model_args.model_name_or_path is None:
        model_args.model_name_or_path = DEFAULT_STUDENT
    # The paper OPD setting is full-parameter training. Keep this launcher
    # robust even if old LoRA/QLoRA flags are accidentally passed.
    model_args.use_peft = False
    model_args.load_in_4bit = False
    model_args.load_in_8bit = False
    if training_args.teacher_model_name_or_path is None:
        training_args.teacher_model_name_or_path = DEFAULT_TEACHER
    training_args.lmbda = 1.0
    training_args.use_uld_loss = True
    training_args.beta = 0.0
    training_args.uld_crossentropy_weight = 0.0
    training_args.uld_distillation_weight = 1.0
    training_args.uld_use_hybrid_loss = False
    if training_args.teacher_tokenizer_name_or_path is None:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=training_args.teacher_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        torch_dtype=model_args.dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = _load_dataset(script_args)
    train_dataset = _ensure_messages(_pick_split(dataset, script_args.dataset_train_split, "train"))

    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_split = None
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            if script_args.dataset_test_split in dataset:
                eval_split = dataset[script_args.dataset_test_split]
            elif "validation" in dataset:
                eval_split = dataset["validation"]
            elif "test" in dataset:
                eval_split = dataset["test"]

        if eval_split is not None:
            eval_dataset = _ensure_messages(eval_split)
        else:
            seed = int(getattr(training_args, "seed", 42))
            train_dataset, eval_dataset = _auto_split_train_eval(
                train_dataset=train_dataset,
                eval_ratio=0.01,
                seed=seed,
            )

    ################
    # Training
    ################
    trainer = GOLDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
