import argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser(description="Report Trigger point")
parser.add_argument('--title', default="Trigger Point", help="Enter the Title of the report.")
parser.add_argument('--filename', default="trigPoint.csv", help="Input csv file name, with no extension.")

args = parser.parse_args()
title = args.title
filename = args.filename

df = pd.read_csv(f'{filename}.csv')
plt.style.use('seaborn-v0_8')
fig, (ax1) = plt.subplots(1)


st = int(df['Subtraction Threshold'].iloc[-1])
sh = int(df['Subtraction History'].iloc[-1])
tp = int(df['Trigger Point'].iloc[-1])
tpb = int(df['Trigger Point Base'].iloc[-1])
mhw = int(df['Movement History Window'].iloc[-1])
mha = int(df['Movement History Age'].iloc[-1])
y_lim = int(df['Trigger Point'].iloc[-1] * 3)
df.loc[df['Trigger Value'] >= y_lim, 'Trigger Value'] = y_lim
df['Trigger Value'] = df['Trigger Value'].replace(0, np.nan)

if os.name == 'nt':
    ax1.set_title(f'Windows - {title}')
else:
    ax1.set_title(f'{str(os.uname()[1])} - {title}')


if os.name == 'nt':
    ax1.set_title(f'Windows - {title}')
else:
    ax1.set_title(f'{str(os.uname()[1])} - {title}')

ax1.plot(df['Frame'], df['Average'],
         color='deepskyblue',
         alpha=0.5,
         label='Average Peak Movement'
         )

ax1.plot(df['Frame'], df['Variable Trigger Point'],
         color='Green',
         alpha=0.5,
         label='Variable Trigger Point'
         )

ax1.plot(df['Frame'], df['Variable Trigger Point Base'],
         color='Red',
         alpha=0.5,
         label='Variable Trigger Point Base'
         )

ax1.scatter(df['Frame'], df['Trigger Value'], alpha=0.5, color='Black')

ax1.set_ylim([0, y_lim])

plt.ylabel('Movement')
ax1.legend(loc='best')

plt.savefig(f'{filename}.png')

if os.name == 'nt':
    plt.show()
