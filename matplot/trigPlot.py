import argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import dates as mpl_dates
import os

parser = argparse.ArgumentParser(description="Report Movement")
parser.add_argument('--hours', type=int, default=0, help="Enter the number of hours to report.")
parser.add_argument('--minutes', type=int, default=0, help="Enter the number of minutes to report.")
parser.add_argument('--title', default="Latest Movement", help="Enter the Title of the report.")
parser.add_argument('--filename', default="trigPlot",
                    help="Enter the output filename, an extension of png will be added.")

args = parser.parse_args()
hours = args.hours
minutes = args.minutes
duration = args.minutes + hours*60
if not duration:
    duration = 15

title = args.title
filename = args.filename

df = pd.read_csv('peakMovement.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp_idx'] = pd.to_datetime(df['Timestamp'])
df.set_index("Timestamp_idx", inplace=True)
last_ts = df.index[-1]
first_ts = last_ts - pd.Timedelta(duration, 'minutes')
df = df[df.index >= first_ts]

# df.sort_values('Timestamp', inplace=True)

plt.style.use('seaborn-v0_8')
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

time = df['Timestamp']

st = int(df['Subtraction Threshold'].iloc[-1])
sh = int(df['Subtraction History'].iloc[-1])
tp = int(df['Trigger Point'].iloc[-1])
tpb = int(df['Trigger Point Base'].iloc[-1])
mhw = int(df['Movement History Window'].iloc[-1])
mha = int(df['Movement History Age'].iloc[-1])
ylim = int(df['Trigger Point'].iloc[-1] * 3)
df.loc[df['Trigger Value'] >= ylim, 'Trigger Value'] = ylim
df['Trigger Value'] = df['Trigger Value'].replace(0, np.nan)

if os.name == 'nt':
    ax1.set_title(f'Windows - {title}')
else:
    ax1.set_title(f'{str(os.uname()[1])} - {title}')

ax1.plot(time, df['Highest Peak'],
         color='Purple',
         alpha=0.5,
         label='Highest Peak Movement'
         )

ax1.plot(time, df['Trigger Point'],
         color='Green',
         alpha=0.5,
         label=f'Trigger Point {tp}'
         )

ax1.plot(time, df['Trigger Point Base'],
         color='Red',
         alpha=0.5,
         label=f'Trigger Point Base {tpb}'
         )

ax1.plot(time, df['Movement History Window'],
         color='Blue',
         alpha=0.5,
         label=f'Movement History Window {mhw}'
         )

ax1.plot(time, df['Movement History Age'],
         color='Orange',
         alpha=0.5,
         label=f'Movement History Age {mha}'
         )

ax1.set_ylim([0, ylim])

if os.name == 'nt':
    ax2.set_title(f'Windows - {title}')
else:
    ax2.set_title(f'{str(os.uname()[1])} - {title}')

ax2.plot(time, df['Average'],
         color='deepskyblue',
         alpha=0.5,
         label='Average Peak Movement'
         )

ax2.plot(time, df['Variable Trigger Point'],
         color='Green',
         alpha=0.5,
         label='Variable Trigger Point'
         )

ax2.plot(time, df['Variable Trigger Point Base'],
         color='Red',
         alpha=0.5,
         label='Variable Trigger Point Base'
         )

ax2.scatter(time, df['Trigger Value'], alpha=0.5, color='Black')

ax2.set_ylim([0, ylim])

plt.gcf().autofmt_xdate()
if duration > 1440:
    # date_format = mpl_dates.DateFormatter('%d/%m')
    date_format = mpl_dates.DateFormatter('%a')
else:
    date_format = mpl_dates.DateFormatter('%H:%M')

plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()
plt.ylabel('Movement')
ax1.legend(loc='best')
ax2.legend(loc='best')

plt.savefig(f'{filename}.png')

if os.name == 'nt':
    plt.show()
