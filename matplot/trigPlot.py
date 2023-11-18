import argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import dates as mpl_dates
import os

parser = argparse.ArgumentParser(description="Report Movement")
parser.add_argument('--records', type=int, default=10000, help="Enter the number of records to report.")
parser.add_argument('--title', default="Latest Movement", help="Enter the Title of the report.")
parser.add_argument('--filename', default="trigPlot",
                   help="Enter the output filename, aqn extension of png will be added.")

args = parser.parse_args()
rec_cnt = args.records
title = args.title
filename = args.filename


plt.style.use('seaborn-v0_8')

data = pd.read_csv('peakMovement.csv')
data = data.tail(rec_cnt)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data.sort_values('Timestamp', inplace=True)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

time = data['Timestamp']

st = int(data['Subtraction Threshold'].iloc[-1])
sh = int(data['Subtraction History'].iloc[-1])
tp = int(data['Trigger Point'].iloc[-1])
tpb = int(data['Trigger Point Base'].iloc[-1])
mhw = int(data['Movement History Window'].iloc[-1])
mha = int(data['Movement History Age'].iloc[-1])
ylim = int(data['Trigger Point'].iloc[-1] * 3)
data.loc[data['Trigger Value'] >= ylim, 'Trigger Value'] = ylim
data['Trigger Value'] = data['Trigger Value'].replace(0, np.nan)

if os.name == 'nt':
    ax1.set_title(f'Windows - {title}')
else:
    ax1.set_title(f'{str(os.uname()[1])} - {title}')

ax1.plot(time, data['Highest Peak'],
         color='Purple',
         alpha=0.5,
         label='Highest Peak Movement'
         )

ax1.plot(time, data['Trigger Point'],
         color='Green',
         alpha=0.5,
         label=f'Trigger Point {tp}'
         )

ax1.plot(time, data['Trigger Point Base'],
         color='Red',
         alpha=0.5,
         label=f'Trigger Point Base {tpb}'
         )

ax1.plot(time, data['Movement History Window'],
         color='Blue',
         alpha=0.5,
         label=f'Movement History Window {mhw}'
         )

ax1.plot(time, data['Movement History Age'],
         color='Yellow',
         alpha=0.5,
         label=f'Movement History Age {mha}'
         )


ax1.set_ylim([0, ylim])

if os.name == 'nt':
    ax2.set_title(f'Windows - {title}')
else:
    ax2.set_title(f'{str(os.uname()[1])} - {title}')

ax2.plot(time, data['Average'],
         color='deepskyblue',
         alpha=0.5,
         label='Average Peak Movement'
         )

ax2.plot(time, data['Variable Trigger Point'],
         color='Green',
         alpha=0.5,
         label='Variable Trigger Point'
         )

ax2.plot(time, data['Variable Trigger Point Base'],
         color='Red',
         alpha=0.5,
         label='Variable Trigger Point Base'
         )

ax2.scatter(time, data['Trigger Value'], alpha=0.5, color='Black')

ax2.set_ylim([0, ylim])

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%H %M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()
plt.ylabel('Movement')
ax1.legend(loc='best')
ax2.legend(loc='best')

plt.savefig(f'{filename}.png')

if os.name == 'nt':
    plt.show()
