import tempfile

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from torch.utils.data import DataLoader
from cs336_alignment.utils import (
    MathSFTDataset,
    make_math_sft_collate_fn,
    get_response_log_probs,
    sft_microbatch_train_step,
    init_vllm,
    evaluate_vllm,
    load_policy_into_vllm_instance,
    EvalResult
)
from drgrpo_grader import r1_zero_reward_fn
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import pd

import wandb


class SFTConfig(BaseModel):
    lr: float
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    epochs: int
    eval_sampling_params: SamplingParams

class ExperimentRunConfig(BaseModel):
    sft_config: SFTConfig
    dataset_path: str
    output_dir: str

device = "cpu"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
WANDB_PROJECT = "cs336-assignment-5"

def eval_model(
    model: PreTrainedModel,
    vllm_instance: LLM,
    eval_sampling_params: SamplingParams,
) -> list[EvalResult]:
    load_policy_into_vllm_instance(
        model,
        vllm_instance
    )
    return evaluate_vllm(
        vllm_model=vllm_instance,
        reward_fn=r1_zero_reward_fn,
        eval_inputs=[],
        eval_sampling_params=eval_sampling_params
    )


def sft_loop(
    experiment_config: ExperimentRunConfig
):
    sft_config = experiment_config.sft_config
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
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    vllm_instance = init_vllm(
        MODEL_NAME,
        device=device,
        seed=sft_config.seed,
    )

    dataset = MathSFTDataset(path=sft_config.dataset_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=sft_config.batch_size,
        shuffle=True,
        collate_fn=make_math_sft_collate_fn(tokenizer=tokenizer),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_config.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    for e in range(sft_config.epochs):
        num_samples_seen = 0
        for idx, data in enumerate(dataloader):
            step_count = e * len(dataloader) + idx
            input_ids = data['input_ids'].to(device)
            labels = data['labels'].to(device)
            response_mask = data['response_mask'].to(device)
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

        eval_results = eval_model(
            model=model,
            vllm_instance=vllm_instance,
        )

        rewards = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])
        rewards = rewards.aggregate(['mean'])
        for reward_name in rewards.index:
            run.log({reward_name: rewards[reward_name]}, step=step_count)

        with tempfile.NamedTemporaryFile() as fp:
            for gen_output in eval_results:
                fp.write(f"{gen_output.model_dump_json()}\n")
            artifact = wandb.Artifact(name="eval_results", type="eval_results")
            artifact.add_file(local_path=fp.name, name=f"{run.name}-epoch-{e:03d}")
            run.log_artifact(artifact)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            model.save_pretrained(temp_dir_name)
            artifact = wandb.Artifact(name="model", type="model")
            artifact.add_dir(local_path=temp_dir_name, name=f"{run.name}-epoch-{e:03d}")
            run.log_artifact(artifact)


