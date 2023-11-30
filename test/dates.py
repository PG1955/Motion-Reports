import pandas as pd, numpy as np

df = pd.DataFrame(list(range(365)))

# these lines are for demonstration purposes only
df['date'] = pd.date_range('2023-1-1', periods=365, freq='D').astype(str)
df = df.set_index('date')

# print(df)


df.index = pd.to_datetime(df.index)

res = df[pd.Timestamp('2023-11-01'):pd.Timestamp('2023-11-10')]

print(res)