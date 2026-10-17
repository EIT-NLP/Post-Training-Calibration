"""
Paper-aligned offline rollout collection.

Example:
python examples/example_offline.py \
  --qid_start 0 --qid_end 29 \
  --rid seed0 \
  --dataset examples/aime_2024.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model_type qwen \
  --budget 320 \
  --output_dir runs/offline_pkls/instruct_aime24_seed0
"""
import json
import pickle
import argparse
from datetime import datetime
from vllm import SamplingParams
from deepconf import DeepThinkLLM
from dynasor.core.evaluator import math_equal
import io
import contextlib
import traceback


# ============= PROMPT PREPARATION FUNCTIONS =============

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        # Format prompt using chat template for DeepSeek
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
    else:
        # Format for GPT-like models
        messages = [
            {"role": "user", "content": question}
        ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt


def prepare_prompt_gpt(question: str, tokenizer, reasoning_effort: str = "high") -> str:
    """Prepare prompt for GPT models with reasoning effort"""
    messages = [
        {"role": "user", "content": question}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True
    )
    
    return full_prompt



def prepare_prompt_qwen(question: str, tokenizer) -> str:
    system_msg = "You are a helpful assistant."
    user_msg = (
        f"{question}\n"
        r"Please reason step by step, and put your final answer within \boxed{}."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )

    return tokenizer.apply_chat_template(messages, **kwargs)


def prepare_prompt_llama(question: str, tokenizer) -> str:
    system_msg = "You are a helpful assistant."
    user_msg = (
        f"{question}\n"
        r"Please reason step by step, and put your final answer within \boxed{}."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def evaluate_voting_results(voting_results, ground_truth):
    """Evaluate voting results against ground truth"""
    evaluation = {}
    
    for method, result in voting_results.items():
        if result and result.get('answer'):
            try:
                is_correct = equal_func(result['answer'], ground_truth)
            except:
                is_correct = str(result['answer']) == str(ground_truth)
            
            evaluation[method] = {
                'answer': result['answer'],
                'is_correct': is_correct,
                'confidence': result.get('confidence'),
                'num_votes': result.get('num_votes', 0)
            }
        else:
            evaluation[method] = {
                'answer': None,
                'is_correct': False,
                'confidence': None,
                'num_votes': 0
            }
    
    return evaluation


def print_evaluation_report(question, ground_truth, evaluation, result):
    """Print detailed evaluation report"""
    print(f"\n=== Evaluation Report ===")
    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Total traces generated: {result.total_traces_count}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Generation time: {result.generation_time:.2f}s")
    
    if result.generation_time > 0:
        print(f"Generation throughput: {result.total_tokens / result.generation_time:.1f} tokens/second")
    
    # Count individual trace accuracy
    correct_traces = sum(1 for trace in result.all_traces 
                        if trace.get('extracted_answer') and 
                        equal_func(trace['extracted_answer'], ground_truth))
    total_valid_traces = sum(1 for trace in result.all_traces if trace.get('extracted_answer'))
    
    if total_valid_traces > 0:
        trace_accuracy = correct_traces / total_valid_traces
        print(f"Individual trace accuracy: {correct_traces}/{total_valid_traces} ({trace_accuracy:.1%})")
    
    print(f"\n=== Voting Method Results ===")
    print("-" * 80)
    print(f"{'Method':<25} {'Answer':<20} {'Correct':<8} {'Confidence':<12} {'Votes':<6}")
    print("-" * 80)
    
    correct_methods = []
    for method, eval_result in evaluation.items():
        answer = str(eval_result['answer'])[:18] + '...' if len(str(eval_result['answer'])) > 20 else str(eval_result['answer'])
        is_correct = eval_result['is_correct']
        confidence = eval_result['confidence']
        num_votes = eval_result['num_votes']
        
        correct_str = '✓' if is_correct else '✗'
        conf_str = f"{confidence:.3f}" if confidence is not None else '-'
        
        print(f"{method:<25} {answer:<20} {correct_str:<8} {conf_str:<12} {num_votes:<6}")
        
        if is_correct:
            correct_methods.append(method)
    
    print(f"\nCorrect voting methods: {correct_methods}")
    
    # Find best method by confidence among correct ones
    correct_evals = {method: eval_result for method, eval_result in evaluation.items() 
                    if eval_result['is_correct']}
    
    if correct_evals:
        best_method = max(correct_evals.items(), 
                         key=lambda x: x[1]['confidence'] if x[1]['confidence'] is not None else 0)
        print(f"Best correct method: {best_method[0]} (confidence: {best_method[1]['confidence']:.3f})")
    
    # Method performance summary
    total_methods = len(evaluation)
    correct_count = len(correct_methods)
    print(f"Method accuracy: {correct_count}/{total_methods} ({correct_count/total_methods:.1%})")



def parse_qid_list(args, dataset_len: int):
    """
    Priority:
      1) --qids "1,2,3"
      2) --qid_start/--qid_end (inclusive)
      3) --qid single
    """
    qids = None

    if args.qids:
        qids = []
        for x in args.qids.split(","):
            x = x.strip()
            if not x:
                continue
            qids.append(int(x))

    elif args.qid_start is not None or args.qid_end is not None:
        start = args.qid_start if args.qid_start is not None else 0
        end = args.qid_end if args.qid_end is not None else (dataset_len - 1)
        if end < start:
            raise ValueError(f"Invalid range: qid_end({end}) < qid_start({start})")
        qids = list(range(start, end + 1))

    elif args.qid is not None:
        qids = [args.qid]

    else:
        raise ValueError("You must specify one of: --qid, --qids, or --qid_start/--qid_end")

    # validate
    for qid in qids:
        if qid < 0 or qid >= dataset_len:
            raise ValueError(f"Question ID {qid} is out of range (0-{dataset_len-1})")

    # 去重但保序（防止用户重复传）
    seen = set()
    uniq = []
    for qid in qids:
        if qid not in seen:
            uniq.append(qid)
            seen.add(qid)
    return uniq





def main():
    parser = argparse.ArgumentParser(description='DeepThinkLLM Offline Mode Example')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help='Model path or name')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for model')
    parser.add_argument('--dataset', type=str, default="aime25.jsonl",
                       help='Dataset file path')

    # Single / multi qid options
    parser.add_argument('--qid', type=int, default=None,
                       help='Question ID to process (0-based index)')
    parser.add_argument('--qids', type=str, default=None,
                       help='Comma-separated qids, e.g. "0,1,2,10"')
    parser.add_argument('--qid_start', type=int, default=None,
                       help='Start qid (inclusive) for a range run')
    parser.add_argument('--qid_end', type=int, default=None,
                       help='End qid (inclusive) for a range run')

    parser.add_argument('--rid', type=str, default="offline_run",
                       help='Run ID for identification')
    parser.add_argument('--budget', type=int, default=320,
                       help='Number of traces to generate')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Sliding window size for confidence computation')
    parser.add_argument('--max_tokens', type=int, default=32768,
                       help='Maximum tokens per generation')
    parser.add_argument('--model_type', type=str, default="qwen", choices=["deepseek", "gpt", "qwen", "llama"],
                       help='Model type for prompt formatting')
    parser.add_argument('--reasoning_effort', type=str, default="high",
                       help='Reasoning effort for GPT models')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=0,
                       help='Top-k sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                       help='Repetition penalty')
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Output directory for results')
    parser.add_argument('--no_multiple_voting', action='store_true',
                       help='Disable multiple voting analysis')

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]

    dataset_len = len(data)
    print(f"Loaded {dataset_len} samples.")

    # Determine qids to run (supports --qids / --qid_start --qid_end / --qid)
    qids = parse_qid_list(args, dataset_len)
    print(f"Will process qids: {qids}")

    # Init model ONCE
    print(f"Initializing DeepThinkLLM model={args.model}, tp={args.tensor_parallel_size} ...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False
    )

    # Create sampling params ONCE (reused per question)
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Prepare output dir
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over questions
    for qid in qids:
        evaluation = None  # avoid unbound issues
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        question_data = data[qid]
        question = question_data['question']
        ground_truth = str(question_data.get('answer', '')).strip()

        print(f"\n==============================")
        print(f"Processing qid={qid} | rid={args.rid}")
        print(f"Question preview: {question[:120]}...")
        if ground_truth:
            print(f"Ground truth: {ground_truth}")

        # Prepare prompt
        if args.model_type == "gpt":
            prompt = prepare_prompt_gpt(question, deep_llm.tokenizer, args.reasoning_effort)
        elif args.model_type == "qwen":
            prompt = prepare_prompt_qwen(question, deep_llm.tokenizer)
        elif args.model_type == "llama":
            prompt = prepare_prompt_llama(question, deep_llm.tokenizer)
        else:
            prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

        # Run deep thinking in offline mode
        result = deep_llm.deepthink(
            prompt=prompt,
            mode="offline",
            budget=args.budget,
            window_size=args.window_size,
            sampling_params=sampling_params,
            compute_multiple_voting=not args.no_multiple_voting
        )

        # Evaluate results against ground truth (if available)
        if ground_truth and getattr(result, "voting_results", None):
            evaluation = evaluate_voting_results(result.voting_results, ground_truth)
            print_evaluation_report(question, ground_truth, evaluation, result)

        # Save results per qid
        result_data = result.to_dict()
        result_data.update({
            'question': question,
            'ground_truth': ground_truth,
            'qid': qid,
            'run_id': args.rid,
            'evaluation': evaluation
        })

        result_filename = f"{args.output_dir}/deepthink_offline_qid{qid}_rid{args.rid}_{timestamp}.pkl"
        with open(result_filename, 'wb') as f:
            pickle.dump(result_data, f)

        print(f"Results saved to {result_filename}")



    print("\nAll done.")






'''
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    
    # Validate question ID
    if args.qid >= len(data) or args.qid < 0:
        raise ValueError(f"Question ID {args.qid} is out of range (0-{len(data)-1})")
    
    question_data = data[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    
    print(f"Processing question {args.qid}: {question[:100]}...")
    
    # Initialize DeepThinkLLM
    deep_llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True)
    
    # Prepare prompt
    print("Preparing prompt...")
    if args.model_type == "gpt":
        prompt = prepare_prompt_gpt(question, deep_llm.tokenizer, args.reasoning_effort)
    elif args.model_type == "qwen":
        prompt = prepare_prompt_qwen(question, deep_llm.tokenizer)
    elif args.model_type == "llama":
        prompt = prepare_prompt_llama(question, deep_llm.tokenizer)
    else:
        prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)
    
    # Create custom sampling parameters
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )
    
    # Run deep thinking in offline mode
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=args.budget,
        window_size=args.window_size,
        sampling_params=sampling_params,
        compute_multiple_voting=not args.no_multiple_voting
    )
    
    # Evaluate results against ground truth
    if ground_truth and result.voting_results:
        evaluation = evaluate_voting_results(result.voting_results, ground_truth)
        print_evaluation_report(question, ground_truth, evaluation, result)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = result.to_dict()
    result_data.update({
        'question': question,
        'ground_truth': ground_truth,
        'qid': args.qid,
        'run_id': args.rid,
        'evaluation': evaluation if ground_truth and result.voting_results else None
    })
    
    result_filename = f"{args.output_dir}/deepthink_offline_qid{args.qid}_rid{args.rid}_{timestamp}.pkl"
    
    with open(result_filename, 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\nResults saved to {result_filename}")
'''



if __name__ == "__main__":
    main()
