import pandas as pd
import numpy as np
from importlib.resources import read_text
from cs336_alignment import prompts
from vllm import LLM, SamplingParams
import json

output_file = "/workspace/outputs/outputs.json"

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/nlile/hendrycks-MATH-benchmark/" + splits["train"]).iloc[:200]

prompt = read_text(prompts, "r1_zero.prompt")
batch_size = 25

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
)

# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-Math-7B",
    download_dir="/workspace/vllm"
)

outputs = []

for _, g in df.groupby(np.arange(len(df)) // batch_size):
    problems_batch = df['problem'].to_list()
    problem_ids_batch = df['unique_id'].to_list()

    input_text = [
        prompt.format(question=problem)
        for problem in problems_batch
    ]

    outputs_batch = llm.generate(
        input_text, sampling_params=sampling_params
    )

    outputs.extend([
        {
            "model_output": output.outputs[0].text,
            "unique_id": problem_id
        }
        for output, problem_id in zip(outputs_batch, problem_ids_batch)
    ])


with open(output_file, "w") as fp:
    json.dump(
        outputs, fp, indent=4
    )