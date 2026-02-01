from cs336_alignment.utils import get_math_benchmark_df
import pandas as pd

pd.set_option("display.max_colwidth", 0)
pd.set_option("display.max_columns", None)

df = get_math_benchmark_df("train")

print(df.head())
