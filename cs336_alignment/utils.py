from pydantic import BaseModel
import pandas as pd
from typing import Callable
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase
import torch
from torch.nn.functional import softmax

class GenerationOutput(BaseModel):
    text_output: str
    unique_id: str

class EvalResult(BaseModel):
    reward_fn_output: dict[str, float]
    generation_output: GenerationOutput

class EvalInput(BaseModel):
    prompt: str
    unique_id: str
    ground_truth: str


def get_math_benchmark_df(
    split: str,
) -> pd.DataFrame:
    splits = {
        'train': 'data/train-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    df = pd.read_parquet(
        "hf://datasets/nlile/hendrycks-MATH-benchmark/" + splits[split]
    )
    return df


def get_math_benchmark_eval_inputs(
    prompt_template: str,
    split: str,
) -> list[EvalInput]:
    df = get_math_benchmark_df(split)
    eval_inputs = []
    for _, r in df.iterrows():
        eval_input = EvalInput(
            prompt=prompt_template.format(question=r['problem']),
            unique_id=r['unique_id'],
            ground_truth=r['solution'],
        )
        eval_inputs.append(eval_input)
    return eval_inputs


def generate_outputs_for_eval(
    vllm_model: LLM,
    eval_inputs: list[EvalInput],
    eval_sampling_params: SamplingParams,
) -> list[GenerationOutput]:
    input_text = [eval_input.prompt for eval_input in eval_inputs]
    outputs = vllm_model.generate(
        prompts=input_text, sampling_params=eval_sampling_params
    )
    output_texts = [
        output.outputs[0].text
        for output in outputs
    ]
    generation_outputs = []
    for eval_input, output_text in zip(eval_inputs, output_texts):
        generation_output = GenerationOutput(
            text_output=output_text,
            unique_id=eval_input.unique_id
        )
        generation_outputs.append(generation_output)
    return generation_outputs


def generate_eval_results(
    eval_inputs: list[EvalInput],
    generated_outputs: list[GenerationOutput],
    reward_fn: Callable[[str, str], dict[str, float]],
) -> list[EvalResult]:
    eval_results = []
    for eval_input, gen_output in zip(eval_inputs, generated_outputs):
        reward_fn_output = reward_fn(
            gen_output.text_output,
            eval_input.ground_truth,
        )
        eval_result = EvalResult(
            reward_fn_output=reward_fn_output,
            generation_output=gen_output
        )
        eval_results.append(eval_result)
    return eval_results


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_inputs: list[EvalInput],
    eval_sampling_params: SamplingParams
) -> list[EvalResult]:
    generated_outputs = generate_outputs_for_eval(
        vllm_model, eval_inputs, eval_sampling_params
    )
    eval_results = generate_eval_results(
        eval_inputs, generated_outputs, reward_fn
    )
    return eval_results


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
):
    pad_token = tokenizer.pad_token_id
    prompt_tokens = tokenizer(prompt_strs, return_length=True)['input_ids']
    output_tokens = tokenizer(output_strs, return_length=True)['input_ids']

    max_len_tokens = max(map(len, prompt_tokens)) + max(map(len, output_tokens))
    batch_size = len(prompt_strs)
    attn_mask = torch.zeros(size=(batch_size, max_len_tokens), dtype=torch.bool)
    input_ids = torch.ones(size=(batch_size, max_len_tokens), dtype=torch.int) * pad_token
    response_labels = torch.zeros(size=(batch_size, max_len_tokens), dtype=torch.bool)

    for b, (prompt, output) in enumerate(zip(prompt_tokens, output_tokens)):
        prompt_len = len(prompt)
        output_len = len(output)
        tot_len = prompt_len + output_len
        attn_mask[b, :tot_len] = True
        response_labels[b, prompt_len:tot_len] = True
        input_ids[b, :tot_len] = torch.as_tensor(prompt + output, dtype=torch.int)

    # populate token and attention mask tensors
    return {
        'input_ids': input_ids[:, :max_len_tokens - 1],
        'labels': input_ids[:, 1:],
        'response_mask': response_labels[:, 1:]
    }


def compute_entropy(logits: torch.Tensor):
    p = softmax(logits, dim=-1)  # B, S, V
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # B, S, V
    return -torch.sum(torch.mul(p, log_probs), dim=-1)