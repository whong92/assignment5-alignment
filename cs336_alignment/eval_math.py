import pandas as pd
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable
from vllm import LLM, SamplingParams

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/nlile/hendrycks-MATH-benchmark/" + splits["train"]).iloc[:200]

for i, r in df.iterrows():
    print(i)
    print(r['problem'])
    print('-------')
    print(r['solution'])
    print(r['answer'])
    print('====')

    ri_score = r1_zero_reward_fn(
        r['answer'],
        r['solution'],
    )

# bla = df.iloc[191]

# print(bla['problem'])
# print(bla['answer'])
# print(bla['solution'])

# print(r1_zero_reward_fn(
#     "<think>hmmm</think> <answer>1/6</answer>",
#     "\\frac{1}{6}",
# ))

from pydantic import BaseModel

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


def generate_outputs_for_eval(
    eval_inputs: list[EvalInput],
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


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_inputs: list[EvalInput],
    eval_sampling_params: SamplingParams
) -> list[EvalResult]:
    eval_results = []
    for eval_input, output_text in zip(eval_inputs, output_texts):
        generation_output = GenerationOutput(
            text_output=output_text,
            unique_id=eval_input.unique_id
        )
        reward_fn_output = reward_fn(
            eval_input.ground_truth,
            output_text
        )
        eval_result = EvalResult(
            reward_fn_output=reward_fn_output,
            generation_output=generation_output
        )
        print(eval_result)