from importlib.resources import read_text
from cs336_alignment import prompts
from vllm import LLM, SamplingParams
from cs336_alignment.utils import get_math_benchmark_eval_inputs, generate_outputs_for_eval

prompt_template = read_text(prompts, "r1_zero.prompt")

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=1.0, 
    top_p=1.0, 
    max_tokens=1024, 
    stop=["</answer>"], 
    include_stop_str_in_output=True
)

# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-Math-7B",
    download_dir="/workspace/vllm"
)

eval_inputs = get_math_benchmark_eval_inputs(
    prompt_template=prompt_template,
    split="test",
)

generation_outputs = generate_outputs_for_eval(
    vllm_model=llm,
    eval_inputs=eval_inputs,
    eval_sampling_params=sampling_params,
)

with open("/workspace/assignment5-alignment/outputs.jsonl", "w") as fp:
    for gen_output in generation_outputs:
        fp.write(f"{gen_output.model_dump_json()}\n")