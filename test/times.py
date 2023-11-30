import numpy as np
import pandas as pd

df = pd.read_csv('peakMovement.csv')
df['Timestamp_idx'] = pd.to_datetime(df['Timestamp'])
df.set_index("Timestamp_idx", inplace=True)

# Use this if the timestamp is the index of the DataFrame
last_ts = df.index[-1]
first_ts = last_ts - pd.Timedelta(50, 'minutes')

print(f'First time: {first_ts} last time: {last_ts}')
# # first_ts = last_ts - pd.Timedelta(0.5, 'seconds')
# # first_ts = last_ts - pd.Timedelta(30, 'minutes')
# # first_ts = last_ts - pd.Timedelta(12, 'hours')

time = df['Timestamp']

st = int(df['Subtraction Threshold'].iloc[-1])
sh = int(df['Subtraction History'].iloc[-1])
tp = int(df['Trigger Point'].iloc[-1])
tpb = int(df['Trigger Point Base'].iloc[-1])
mhw = int(df['Movement History Window'].iloc[-1])
mha = int(df['Movement History Age'].iloc[-1])
ylim = int(df['Trigger Point'].iloc[-1] * 3)
# df.loc[df['Trigger Value'] >= ylim, 'Trigger Value'] = ylim
# df['Trigger Value'] = df['Trigger Value'].replace(0, np.nan)

