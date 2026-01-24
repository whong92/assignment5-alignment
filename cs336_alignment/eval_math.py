from importlib.resources import read_text
from cs336_alignment import prompts
from cs336_alignment.utils import generate_eval_results, GenerationOutput, get_math_benchmark_eval_inputs, EvalResult
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pandas as pd

prompt_template = read_text(prompts, "r1_zero.prompt")

generation_outputs = []
with open("/workspace/assignment5-alignment/outputs.jsonl", "r") as fp:
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

with open("/workspace/assignment5-alignment/eval_results.jsonl", "w") as fp:
    for eval_result in eval_results:
        fp.write(f"{eval_result.model_dump_json()}\n")

rewards = pd.DataFrame.from_records([r.reward_fn_output for r in eval_results])

print(rewards.aggregate(['sum', 'count']))