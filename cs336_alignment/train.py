import os

os.environ["WANDB_API_KEY"] = "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"

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from torch.utils.data import DataLoader
from importlib.resources import read_text
from cs336_alignment import prompts
from cs336_alignment.utils import (
    MathSFTDataset,
    make_math_sft_collate_fn,
    get_response_log_probs,
    sft_microbatch_train_step,
    init_vllm,
    evaluate_vllm,
    load_policy_into_vllm_instance,
    EvalResult,
    get_math_benchmark_eval_inputs
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from pydantic import BaseModel, computed_field
from vllm import LLM, SamplingParams
import pandas as pd
import yaml
from tqdm import tqdm

import wandb

prompt_template = read_text(prompts, "r1_zero.prompt")


class SFTConfig(BaseModel):
    lr: float
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    epochs: int


class ExperimentRunConfig(BaseModel):
    sft_config: SFTConfig
    dataset_path_rel: str
    output_dir_rel: str
    device_model: str
    device_vllm: str
    base_path: str
    max_seq_tok_len: int

    @computed_field
    @property
    def dataset_path(self) -> str:
        return os.path.join(self.base_path, self.dataset_path_rel)

    @computed_field
    @property
    def output_dir(self) -> str:
        return os.path.join(self.base_path, self.output_dir_rel)


MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
WANDB_PROJECT = "cs336-assignment-5"
SAMPLING_PARAMS = SamplingParams(
    temperature=1.0, 
    top_p=1.0, 
    max_tokens=1024, 
    stop=["</answer>"], 
    include_stop_str_in_output=True
)


def eval_model(
    model: PreTrainedModel,
    vllm_instance: LLM,
    eval_sampling_params: SamplingParams,
) -> list[EvalResult]:
    eval_inputs = get_math_benchmark_eval_inputs(
        prompt_template=prompt_template,
        split="test",
    )

    load_policy_into_vllm_instance(
        model,
        vllm_instance
    )
    return evaluate_vllm(
        vllm_model=vllm_instance,
        reward_fn=r1_zero_reward_fn,
        eval_inputs=eval_inputs,
        eval_sampling_params=eval_sampling_params
    )


def sft_loop(
    experiment_config: ExperimentRunConfig
):
    print("Initializing.")
    
    sft_config = experiment_config.sft_config
    if not os.path.exists(experiment_config.output_dir):
        os.makedirs(experiment_config.output_dir, exist_ok=True)

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="wai-ong11",
        # Set the wandb project where this run will be logged.
        project = WANDB_PROJECT,
        # Track hyperparameters and run metadata.
        config=experiment_config.model_dump(mode="json"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(experiment_config.device_model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    vllm_instance = init_vllm(
        MODEL_NAME,
        device=experiment_config.device_vllm,
        seed=sft_config.seed,
    )

    dataset = MathSFTDataset(path=experiment_config.dataset_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=sft_config.batch_size,
        shuffle=True,
        collate_fn=make_math_sft_collate_fn(tokenizer=tokenizer, max_seq_token_len=experiment_config.max_seq_tok_len),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_config.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    print("Start Training Loop.")
    for e in range(sft_config.epochs):
        num_samples_seen = 0
        for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            step_count = e * len(dataloader) + idx
            input_ids = data['input_ids'].to(experiment_config.device_model)
            labels = data['labels'].to(experiment_config.device_model)
            response_mask = data['response_mask'].to(experiment_config.device_model)
            # Forward pass.
            ret = get_response_log_probs(
                model,
                input_ids=input_ids,
                labels=labels,
            )
            log_probs = ret['log_probs']

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
            )
            run.log({"loss": loss.cpu().item()}, step=step_count)

            num_samples_seen += len(input_ids)
            if ((idx + 1) % sft_config.gradient_accumulation_steps == 0) or (num_samples_seen == len(dataset)):
                # Update weights every `gradient_accumulation_steps` batches.
                optimizer.step()
                # Zero gradients every `gradient_accumulation_steps` batches.
                optimizer.zero_grad()

        print("Evaluating.")
        eval_results = eval_model(
            model=model,
            vllm_instance=vllm_instance,
            eval_sampling_params=SAMPLING_PARAMS,
        )

        rewards = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])
        rewards = rewards.aggregate(['mean'])
        for reward_name in rewards.index:
            run.log({reward_name: rewards.loc[reward_name].item()}, step=step_count)

        # save eval results
        eval_results_output_path = os.path.join(experiment_config.output_dir, f"eval_results.jsonl")
        with open(eval_results_output_path, "w") as fp:
            for gen_output in eval_results:
                fp.write(f"{gen_output.model_dump_json()}\n")
        artifact = wandb.Artifact(name="eval_results", type="eval_results")
        artifact.add_file(local_path=eval_results_output_path, name=f"{run.name}-epoch-{e:03d}")
        run.log_artifact(artifact)

        # save model
        model_output_dir = os.path.join(experiment_config.output_dir, f"model")
        os.makedirs(model_output_dir, exist_ok=True)
        model.save_pretrained(model_output_dir)
        artifact = wandb.Artifact(name="model", type="model")
        artifact.add_dir(local_path=model_output_dir, name=f"{run.name}-epoch-{e:03d}")
        run.log_artifact(artifact)


def main():
    with open("/workspace/assignment5-alignment/cs336_alignment/basic_config.yaml", "r") as fp:
        config_dict = yaml.safe_load(fp)
    experiment_config = ExperimentRunConfig.model_validate(config_dict)
    sft_loop(experiment_config)


if __name__ == "__main__":
    main()