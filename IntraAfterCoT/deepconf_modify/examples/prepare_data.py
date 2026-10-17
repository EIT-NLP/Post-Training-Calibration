import json



'''
import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MathArena/aime_2025", split="train")

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to aime_2025.jsonl")
'''


input_path = "math500.jsonl"
output_path = "math500_convert.jsonl"

count = 0
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        example = json.loads(line)

        ans = str(example["answer"])


        entry = {
            "question": example["problem"],
            "answer": ans
        }

        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count += 1

print(f"Converted {count} examples to {output_path}")

