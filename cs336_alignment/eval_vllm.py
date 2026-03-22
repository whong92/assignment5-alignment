from typing import Callable
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from cs336_alignment.utils import (
    EvalInput,
    EvalResult,
    GenerationOutput,
    generate_eval_results,
    get_math_benchmark_eval_inputs,
    R1_ZERO_PROMPT_TEMPLATE,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import tempfile


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


def log_generations(
    llm: LLM,
    output_path: str
) -> None:
    eval_inputs = get_math_benchmark_eval_inputs(
        prompt_template=R1_ZERO_PROMPT_TEMPLATE,
        split="test",
    )
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    generation_outputs = generate_outputs_for_eval(
        vllm_model=llm,
        eval_inputs=eval_inputs,
        eval_sampling_params=sampling_params,
    )
    eval_results = generate_eval_results(
        eval_inputs=eval_inputs,
        generated_outputs=generation_outputs,
        reward_fn=r1_zero_reward_fn,
    )
    with open(output_path, "w") as fp:
        for eval_result in eval_results:
            fp.write(f"{eval_result.model_dump_json()}\n")


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    print("=" * 100)
    print("Loading policy weights into vLLM model...")
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
    llm_model.load_weights(state_dict_cpu.items())


def policy_to_vllm_model(
    policy: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    vllm_device: str, 
    seed: int
) -> LLM:
    # Transfer policy weights -> vllm by saving it to file first
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        llm = init_vllm(model_id=tmp_dir, device=vllm_device, seed=seed)
    return llm