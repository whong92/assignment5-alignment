from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen2.5/Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen2.5/Math-1.5B")



train_batch = tokenize_prompt_and_output(
    prompt_strs,
    output_strs,
    tokenizer,
)
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)