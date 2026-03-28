from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from importlib.resources import read_text
from vllm import LLM, SamplingParams
import yaml
import json
from cs336_alignment import prompts
from cs336_alignment.utils import (
    EvalResult,
    get_math_benchmark_eval_inputs
)
from pydantic import BaseModel
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval_vllm import (
    evaluate_vllm,
    load_policy_into_vllm_instance,
)
from cs336_alignment.train import model_setup, eval_model_and_save_results, run_sft_epoch_train, data_setup_from_config, data_setup
from cs336_alignment.train import ExperimentRunConfig as SFTExperimentRunConfig
import pandas as pd
import os
import wandb
import numpy as np
from cs336_alignment.train import SAMPLING_PARAMS

prompt_template = read_text(prompts, "r1_zero.prompt")


class ExpertIterationConfig(BaseModel):
    num_ei_steps: int
    num_rollouts: int
    data_batch_size: int


def expert_rollout(
    model: PreTrainedModel,
    vllm_instance: LLM,
    eval_sampling_params: SamplingParams,
    output_path: str,
    wandb_run: wandb.Run,
    expert_iteration_config: ExpertIterationConfig,
    expert_iteration: int,
) -> None:
    eval_inputs = get_math_benchmark_eval_inputs(
        prompt_template=prompt_template,
        split="train",
    )

    num_data_to_sample = min(expert_iteration_config.data_batch_size, len(eval_inputs))
    selected_indices = np.random.choice(len(eval_inputs), num_data_to_sample, replace=False)
    eval_inputs_selected = [
        eval_inputs[i]
        for _ in range(expert_iteration_config.num_rollouts)
        for i in selected_indices
    ]

    load_policy_into_vllm_instance(
        model,
        vllm_instance
    )
    eval_results = evaluate_vllm(
        vllm_model=vllm_instance,
        reward_fn=r1_zero_reward_fn,
        eval_inputs=eval_inputs_selected,
        eval_sampling_params=eval_sampling_params
    )

    eval_inputs_df = pd.DataFrame.from_records([e.model_dump() for e in eval_inputs_selected])
    rewards_df = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])
    generated_outputs_df = pd.DataFrame.from_records([r.generation_output.model_dump() for r in eval_results])
    combined_df = pd.concat([eval_inputs_df, rewards_df, generated_outputs_df], axis=1)
    good_outputs = combined_df.loc[
        combined_df['reward'] > 0, ['prompt', 'text_output']
    ].rename(columns={'prompt': 'input', 'text_output': 'output'})

    good_outputs_json = good_outputs.to_dict(orient='records')
    
    with open(output_path, "w") as fp:
        for record in good_outputs_json:
            fp.write(f"{json.dumps(record)}\n")

    # artifact = wandb.Artifact(name="expert_rollout", type="expert_rollout")
    # artifact.add_file(local_path=output_path, name=f"{wandb_run.name}-epoch-{expert_iteration-1:03d}")
    # wandb_run.log_artifact(artifact)


class ExperimentRunConfig(SFTExperimentRunConfig):
    expert_iteration_config: ExpertIterationConfig


def expert_iteration_sft(
    experiment_config: ExperimentRunConfig
):

    dataloader = data_setup_from_config(experiment_config) # original dataloader, will need to add expert rollouts to this dataloader
    model, vllm_instance, optimizer, scheduler, run = model_setup(
        experiment_config,
        experiment_config.sft_config.epochs * len(dataloader) // experiment_config.sft_config.gradient_accumulation_steps
    )
    tot_num_steps = 0

    print("Start Training Loop.")
    for ei_step in range(experiment_config.expert_iteration_config.num_ei_steps):
        print("Performing expert rollout.")
        rollout_output_path = os.path.join(experiment_config.output_dir, f"expert_rollout_ei_step_{ei_step:03d}.jsonl")
        expert_rollout(
            model=model,
            vllm_instance=vllm_instance,
            eval_sampling_params=SAMPLING_PARAMS,
            output_path=rollout_output_path,
            wandb_run=run,
            expert_iteration_config=experiment_config.expert_iteration_config,
            expert_iteration=ei_step,
        )

        dataloader = data_setup(
            dataset_path=rollout_output_path,
            batch_size=experiment_config.sft_config.batch_size,
            max_seq_token_len=experiment_config.max_seq_tok_len,
        )

        for inner_epoch in range(experiment_config.sft_config.epochs):
            epoch = ei_step * experiment_config.sft_config.epochs + inner_epoch
            print("Evaluating.")
            eval_model_and_save_results(
                model=model,
                vllm_instance=vllm_instance,
                run=run,
                experiment_config=experiment_config,
                epoch=epoch,
                step=tot_num_steps,
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

        # # save model
        # model_output_dir = os.path.join(experiment_config.output_dir, f"model")
        # os.makedirs(model_output_dir, exist_ok=True)
        # model.save_pretrained(model_output_dir)
        # artifact = wandb.Artifact(name="model", type="model")
        # artifact.add_dir(local_path=model_output_dir, name=f"{run.name}-epoch-{e:03d}")
        # run.log_artifact(artifact)


if __name__ == "__main__":
        with open("/workspace/assignment5-alignment/cs336_alignment/expert_iter_config.yaml", "r") as fp:
            config_dict = yaml.safe_load(fp)
        experiment_config = ExperimentRunConfig(**config_dict)
        expert_iteration_sft(experiment_config)
