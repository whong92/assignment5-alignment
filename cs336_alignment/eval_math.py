from importlib.resources import read_text
from cs336_alignment import prompts
from cs336_alignment.utils import generate_eval_results, GenerationOutput, get_math_benchmark_eval_inputs, EvalResult
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pandas as pd
from pathlib import Path

prompt_template = read_text(prompts, "r1_zero.prompt")
curfile = Path(__file__)

generation_outputs = []
with open(f"{curfile.parent.parent.resolve().as_posix()}/outputs.jsonl", "r") as fp:
    for line in fp.readlines():
        gen_output = GenerationOutput.model_validate_json(line)
        generation_outputs.append(gen_output)

eval_inputs = get_math_benchmark_eval_inputs(
    prompt_template=prompt_template,
    split="test",
)

eval_results = generate_eval_results(
    eval_inputs=eval_inputs,
    generated_outputs=generation_outputs,
    reward_fn=r1_zero_reward_fn,
)

with open(f"{curfile.parent.parent.resolve().as_posix()}/eval_results.jsonl", "w") as fp:
    for eval_result in eval_results:
        fp.write(f"{eval_result.model_dump_json()}\n")

rewards = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])
rewards = rewards.aggregate('mean')
print(rewards.index)
print(rewards.loc['format_reward'])