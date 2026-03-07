from pydantic import BaseModel
import pandas as pd
from typing import Callable
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch
from torch.utils.data import Dataset
import json
from functools import partial
from unittest.mock import patch


from importlib.resources import read_text
from cs336_alignment import prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


import random

R1_ZERO_PROMPT_TEMPLATE = read_text(prompts, "r1_zero.prompt")
R1_ZERO_OUTPUT_PROMPT_TEMPLATE = read_text(prompts, "r1_zero_output.prompt")

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


def format_math_benchmark_train_sample(
    input_prompt: str,
    output_prompt: str,
    question: str,
    reasoning: str,
    answer: str
) -> dict[str, str]:
    return {
        "input": input_prompt.format(question=question),
        "output": output_prompt.format(reasoning=reasoning, answer=answer),
    }


def get_math_benchmark_train_dataset(
    split: str
) -> list[dict]:
    df = get_math_benchmark_df(split)
    dataset = []
    for r in df.itertuples():
        train_sample = format_math_benchmark_train_sample(
            R1_ZERO_PROMPT_TEMPLATE,
            R1_ZERO_OUTPUT_PROMPT_TEMPLATE,
            question=r.problem,
            reasoning=r.solution,
            answer=r.answer,
        )
        dataset.append(train_sample)
    return dataset


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


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_token_len: int | None = None,
):
    pad_token = tokenizer.pad_token_id
    prompt_tokens = tokenizer(prompt_strs, return_length=True)['input_ids']
    output_tokens = tokenizer(output_strs, return_length=True)['input_ids']

    max_len_tokens = max(map(len, prompt_tokens)) + max(map(len, output_tokens))
    if max_seq_token_len is not None:
        max_len_tokens = min(max_len_tokens, max_seq_token_len)
    batch_size = len(prompt_strs)
    input_ids = torch.ones(size=(batch_size, max_len_tokens), dtype=torch.int64) * pad_token
    response_labels = torch.zeros(size=(batch_size, max_len_tokens), dtype=torch.bool)

    for b, (prompt, output) in enumerate(zip(prompt_tokens, output_tokens)):

        prompt_len = len(prompt)
        output_len = len(output)
        tot_len = prompt_len + output_len
        _response_labels_b = torch.zeros(size=(tot_len,), dtype=torch.bool)
        _response_labels_b[prompt_len:tot_len] = True
        _tokens_b = torch.as_tensor(prompt + output, dtype=torch.int64)

        if tot_len > max_len_tokens:
            random_offset = random.randint(0, tot_len - max_len_tokens)
            len_to_take = max_len_tokens
        else:
            random_offset = 0
            len_to_take = tot_len

        response_labels[b, :len_to_take] = _response_labels_b[random_offset:(random_offset + len_to_take)]
        input_ids[b, :len_to_take] = _tokens_b[random_offset:(random_offset + len_to_take)]

    # populate token and attention mask tensors
    return {
        'input_ids': input_ids[:, :max_len_tokens - 1],
        'labels': input_ids[:, 1:],
        'response_mask': response_labels[:, 1:]
    }


def compute_entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # B, S, V
    p = torch.exp(log_probs)
    return -torch.sum(torch.mul(p, log_probs), dim=-1), log_probs


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor, # B, S
    labels: torch.Tensor,  # B, S
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    ent, log_probs = compute_entropy(logits)
    labels = labels.unsqueeze(-1)
    labels_log_prob = torch.gather(log_probs, dim=-1, index=labels)
    ret = {
        "log_probs": labels_log_prob[:, :, 0],
    }
    if return_token_entropy:
        ret["token_entropy"] = ent
    return ret


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    return torch.sum(masked_tensor, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,  # B, S
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masked_policy_log_probs = masked_normalize(
        tensor=policy_log_probs, mask=response_mask, normalize_constant=normalize_constant, dim=1
    )
    loss = -torch.mean(masked_policy_log_probs) / gradient_accumulation_steps
    loss.backward()
    return loss, {}


class MathSFTDataset(Dataset):
    def __init__(self, path: str):
        self.data: list[dict[str, str]] = []
        with open(path, "r") as fp:
            for line in fp.readlines():
                self.data.append(json.loads(line))
        self.data = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def math_sft_collate_fn(
    batch: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_token_len: int | None = None
) -> dict[str, torch.Tensor]:
    return tokenize_prompt_and_output(
        prompt_strs = [b["input"] for b in batch],
        output_strs=[b["output"] for b in batch],
        tokenizer=tokenizer,
        max_seq_token_len=max_seq_token_len
    )

def make_math_sft_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_token_len: int | None = None
) -> Callable[[list[dict[str, str]]], dict[str, torch.Tensor]]:
    return partial(math_sft_collate_fn, tokenizer=tokenizer, max_seq_token_len=max_seq_token_len)

def debug_dataloader():
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import numpy as np
    import os
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    dataset_path = f"{os.path.dirname(__file__)}/sft-train.jsonl"
    dataset = MathSFTDataset(path=dataset_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=make_math_sft_collate_fn(tokenizer=tokenizer, max_seq_token_len=1024),
    )

    lens = []
    response_lens = []
    for batch in dataloader:
        lens.append(len(batch["input_ids"][0]))
        response_lens.append(torch.sum(batch["response_mask"][0]).item())

        print("=" * 10)
        print(tokenizer.decode(batch["input_ids"][0]))
        print("=" * 10)
        print(tokenizer.decode(batch["labels"][0][batch["response_mask"][0]]))
        
        break


    print("total lens percentiles: ", np.percentile(lens, [25, 50, 75, 90, 95, 100], axis=0))
    print("response lens percentiles: ", np.percentile(response_lens, [25, 50, 75, 90, 95, 100], axis=0))

if __name__ == "__main__":
    debug_dataloader()