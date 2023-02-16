import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import dates as mpl_dates
import os

plt.style.use('seaborn-v0_8')

data = pd.read_csv('peakMovement.csv')
data = data.tail(60)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data.sort_values('Timestamp', inplace=True)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

time = data['Timestamp']

# highest_max = data['Highest Peak'].max() * 1.20 # Add 20%.
# trigger_max = data['Trigger Value'].max() * 1.20 # Add 20%.
# peak_mean = data['Highest Peak'].mean()
# trigger_mean = data['Trigger Value'].mean()
st = int(data['Subtraction Threshold'].iloc[-1])
sh = int(data['Subtraction History'].iloc[-1])
tp = int(data['Trigger Point'].iloc[-1])
tpb = int(data['Trigger Point Base'].iloc[-1])


data['Trigger Value']=data['Trigger Value'].replace(0, np.nan)

if os.name == 'nt':
    ax1.set_title('Windows -  Most Recent Average Peak Movement')
else:
    ax1.set_title(str(os.uname()[1]) + ' -  Most Recent Average Peak Movement')
    
ax1.plot(time, data['Highest Peak'],
         color='purple',
         alpha=0.5,
         label='Highest Peak Movement'
        )

ax1.plot(time, data['Average'],
         color='deepskyblue',
         alpha=0.5,
         label='Average Peak Movement'
        )
        
        
ax1.plot(time, data['Trigger Point Base'],
         color='yellow',
         alpha=0.5,
         label=f'Trigger Point Base {tpb}'
        )
        
ax1.plot(time, data['Trigger Point'],
         color='Green',
         alpha=0.5,
         label=f'Trigger point {tp}'
        )

ax1.set_ylim([0,data['Trigger Point'].max() * 2])

if os.name == 'nt':
    ax2.set_title('Windows - Most Recent Peak Movement')
else:
    ax2.set_title(str(os.uname()[1]) + ' - Most Recent Peak Movement')


ax2.plot(time, data['Trigger Point'],
         color='Green',
         alpha=0.5,
         label=f'Trigger point {tp}'
        )
        
        
ax2.plot(time, data['Trigger Point Base'],
         color='yellow',
         alpha=0.5,
         label=f'Trigger Point Base {tpb}'
        )

ax2.plot(time, data['Subtraction Threshold'] * 10,
         color='pink',
         alpha=0.5,
         label=f'Subtraction Threshold {st}'
         )

ax2.plot(time, data['Subtraction History'],
         color='brown',
         alpha=0.5,
         label=f'Subtraction History {sh}'
         )

ax2.scatter(time, data['Trigger Value'], alpha=0.5, color='Red')

ax2.set_ylim([0,data['Trigger Point'].max() * 2])

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%H %M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()
plt.ylabel('Movement')
ax1.legend(loc='best')
ax2.legend(loc='best')

plt.savefig('peak30.png')

if os.name == 'nt':
    plt.show()
