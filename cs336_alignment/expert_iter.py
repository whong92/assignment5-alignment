from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from importlib.resources import read_text
from vllm import LLM, SamplingParams
from cs336_alignment import prompts
from cs336_alignment.utils import (
    EvalResult,
    get_math_benchmark_eval_inputs
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval_vllm import (
    evaluate_vllm,
    load_policy_into_vllm_instance,
)
from cs336_alignment.train import ExperimentRunConfig, model_setup, eval_model_and_save_results, run_sft_epoch_train, data_setup
import pandas as pd
import os
import wandb

prompt_template = read_text(prompts, "r1_zero.prompt")

def expert_rollout(
    model: PreTrainedModel,
    vllm_instance: LLM,
    eval_sampling_params: SamplingParams,
    output_path: str,
    wandb_run: wandb.Run,
    expert_iteration: int,
) -> None:
    eval_inputs = get_math_benchmark_eval_inputs(
        prompt_template=prompt_template,
        split="train",
    )

    load_policy_into_vllm_instance(
        model,
        vllm_instance
    )
    eval_results = evaluate_vllm(
        vllm_model=vllm_instance,
        reward_fn=r1_zero_reward_fn,
        eval_inputs=eval_inputs,
        eval_sampling_params=eval_sampling_params
    )

    eval_inputs_df = pd.DataFrame.from_records([e.model_dump() for e in eval_inputs])
    rewards_df = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])
    generated_outputs_df = pd.DataFrame.from_records([r.generation_output.model_dump() for r in eval_results])
    combined_df = pd.concat([eval_inputs_df, rewards_df, generated_outputs_df], axis=1)
    good_outputs = combined_df.loc[
        combined_df['reward'] > 0, ['prompt', 'text_output']
    ].rename(columns={'prompt': 'input', 'text_output': 'output'})

    good_outputs_json = good_outputs.to_dict(orient='records')
    
    with open(output_path, "w") as fp:
        for record in good_outputs_json:
            fp.write(f"{record}\n")

    artifact = wandb.Artifact(name="expert_rollout", type="expert_rollout")
    artifact.add_file(local_path=output_path, name=f"{wandb_run.name}-epoch-{expert_iteration-1:03d}")
    wandb_run.log_artifact(artifact)


def expert_iteration_sft(
    experiment_config: ExperimentRunConfig
):

    dataloader = data_setup(experiment_config). # original dataloader, will need to add expert rollouts to this dataloader
    model, vllm_instance, optimizer, scheduler, run = model_setup(
        experiment_config,
        experiment_config.sft_config.epochs * len(dataloader) // experiment_config.sft_config.gradient_accumulation_steps
    )

    print("Start Training Loop.")
    for e in range(experiment_config.sft_config.epochs):
        print("Evaluating.")

        eval_model_and_save_results(
            model=model,
            vllm_instance=vllm_instance,
            run=run,
            experiment_config=experiment_config,
            epoch=e,
            steps_per_epoch=len(dataloader),
        )

        run_sft_epoch_train(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            run=run,
            experiment_config=experiment_config,
            epoch=e,
        )

        # save model
        model_output_dir = os.path.join(experiment_config.output_dir, f"model")
        os.makedirs(model_output_dir, exist_ok=True)
        model.save_pretrained(model_output_dir)
        artifact = wandb.Artifact(name="model", type="model")
        artifact.add_dir(local_path=model_output_dir, name=f"{run.name}-epoch-{e:03d}")
        run.log_artifact(artifact)