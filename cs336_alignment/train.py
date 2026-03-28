import gc
import os

os.environ["WANDB_API_KEY"] = "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"

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.optimizer import ParamsT
from torch import Tensor
from importlib.resources import read_text
from cs336_alignment import prompts
from cs336_alignment.utils import (
    MathSFTDatasetFromFile,
    make_math_sft_collate_fn,
    get_response_log_probs,
    sft_microbatch_train_step,
    EvalResult,
    get_math_benchmark_eval_inputs
)
from cs336_alignment.eval_vllm import (
    evaluate_vllm,
    policy_to_vllm_model,
    init_vllm,
    load_policy_into_vllm_instance
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from pydantic import BaseModel, computed_field
from vllm import LLM, SamplingParams
import pandas as pd
import yaml
from tqdm import tqdm
from vllm.distributed.parallel_state import destroy_model_parallel

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
    model_ckpt_path: str | None = None

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

class AdamWClipped(AdamW):
    def __init__(
        self,
        params: ParamsT,
        lr: float| Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        for group in self.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)
        super().step(closure)


def eval_model_on_vllm(
    model: PreTrainedModel,
    # tokenizer: AutoTokenizer,
    vllm_instance: LLM,
    # vllm_device: str,
    eval_sampling_params: SamplingParams,
) -> list[EvalResult]:
    eval_inputs = get_math_benchmark_eval_inputs(
        prompt_template=prompt_template,
        split="test",
    )
    load_policy_into_vllm_instance(
        policy=model,
        llm=vllm_instance
    )

    # vllm_instance = policy_to_vllm_model(
    #     model, tokenizer, vllm_device=vllm_device, seed=42
    # )

    res = evaluate_vllm(
        vllm_model=vllm_instance,
        reward_fn=r1_zero_reward_fn,
        eval_inputs=eval_inputs,
        eval_sampling_params=eval_sampling_params
    )

    return res


def run_sft_epoch_train(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: AdamWClipped,
    scheduler: CosineAnnealingLR,
    run: wandb.Run,
    experiment_config: ExperimentRunConfig,
    num_train_steps_so_far: int
) -> int:
    sft_config = experiment_config.sft_config
    total_num_samples = len(dataloader.dataset)
    num_samples_seen = 0
    tot_num_steps = num_train_steps_so_far
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
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
        run.log({"loss": loss.cpu().item()}, step=tot_num_steps)
        tot_num_steps += 1

        num_samples_seen += len(input_ids)
        if ((idx + 1) % sft_config.gradient_accumulation_steps == 0) or (num_samples_seen == total_num_samples):
            # Update weights every `gradient_accumulation_steps` batches.
            optimizer.step()
            # Zero gradients every `gradient_accumulation_steps` batches.
            optimizer.zero_grad()
            scheduler.step()
    return tot_num_steps


def data_setup_from_config(experiment_config: ExperimentRunConfig) -> DataLoader:
    return data_setup(
        dataset_path=experiment_config.dataset_path,
        batch_size=experiment_config.sft_config.batch_size,
        max_seq_token_len=experiment_config.max_seq_tok_len,
    )


def data_setup(dataset_path: str, batch_size: int, max_seq_token_len: int) -> DataLoader:
    dataset = MathSFTDatasetFromFile(path=dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_math_sft_collate_fn(
            tokenizer=tokenizer, 
            max_seq_token_len=max_seq_token_len
        ),
    )
    return dataloader


def model_setup(
    experiment_config: ExperimentRunConfig,
    tot_num_optimizer_steps: int
) -> tuple[
    PreTrainedModel, 
    AdamWClipped, 
    CosineAnnealingLR, 
    wandb.Run
]:
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

    model_path = experiment_config.model_ckpt_path if experiment_config.model_ckpt_path is not None else MODEL_NAME
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(experiment_config.device_model)

    vllm_instance = init_vllm(
        model_path,
        device=experiment_config.device_vllm,
        seed=sft_config.seed,
    )

    optimizer = AdamWClipped(
        model.parameters(),
        lr=sft_config.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        max_grad_norm=1.0,
    )

    num_steps = tot_num_optimizer_steps
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_steps,
        eta_min=sft_config.lr * 0.1,
    )

    return model, vllm_instance, optimizer, scheduler, run
    # return model, optimizer, scheduler, run


def eval_model_and_save_results(
    model: PreTrainedModel,
    # tokenizer: AutoTokenizer,
    vllm_instance: LLM,
    run: wandb.Run,
    experiment_config: ExperimentRunConfig,
    epoch: int,
) -> None:
    eval_results = eval_model_on_vllm(
        model=model,
        vllm_instance=vllm_instance,
        # tokenizer=tokenizer,
        # vllm_device=experiment_config.device_vllm,
        eval_sampling_params=SAMPLING_PARAMS,
    )
    rewards = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])
    rewards = rewards.aggregate(['mean'])
    for reward_name in rewards.columns:
        run.log({reward_name: rewards.loc['mean', reward_name].item()}, step=epoch)

    # save eval results
    eval_results_output_path = os.path.join(experiment_config.output_dir, f"eval_results.jsonl")
    with open(eval_results_output_path, "w") as fp:
        for gen_output in eval_results:
            fp.write(f"{gen_output.model_dump_json()}\n")
    artifact = wandb.Artifact(name="eval_results", type="eval_results")
    artifact.add_file(local_path=eval_results_output_path, name=f"{run.name}-epoch-{epoch-1:03d}")
    run.log_artifact(artifact)


def sft_loop(
    experiment_config: ExperimentRunConfig
):
    print("Initializing.")

    dataloader = data_setup_from_config(experiment_config)
    model, vllm_instance, optimizer, scheduler, run = model_setup(
        experiment_config,
        experiment_config.sft_config.epochs * len(dataloader) // experiment_config.sft_config.gradient_accumulation_steps
    )
    tot_num_steps = 0

    print("Start Training Loop.")
    for e in range(experiment_config.sft_config.epochs):
        print("Evaluating.")

        eval_model_and_save_results(
            model=model,
            # tokenizer=tokenizer,
            vllm_instance=vllm_instance,
            run=run,
            experiment_config=experiment_config,
            epoch=e,
        )

        tot_num_steps = run_sft_epoch_train(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            run=run,
            experiment_config=experiment_config,
            num_train_steps_so_far=tot_num_steps,
        )

        # save mode[l
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