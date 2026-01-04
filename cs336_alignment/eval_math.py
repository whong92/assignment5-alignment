import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/nlile/hendrycks-MATH-benchmark/" + splits["train"]).iloc[:200]

# for i, r in df.iterrows():
#     print(i)
#     print(r['problem'])
#     # print(r['solution'])
#     print(r['answer'])
#     print('====')

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

bla = df.iloc[191]

print(bla['problem'])
print(bla['answer'])

print(r1_zero_reward_fn(
    "<think>hmmm</think> <answer>1/6</answer>",
    "\\frac{1}{6}",
))