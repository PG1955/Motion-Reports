import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import dates as mpl_dates
import os

plt.style.use('seaborn-v0_8')

data = pd.read_csv('peakMovement.csv')
data = data.tail(1440)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data.sort_values('Timestamp', inplace=True)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

time = data['Timestamp']


st = int(data['Subtraction Threshold'].iloc[-1])
sh = int(data['Subtraction History'].iloc[-1])


ylim = int(data['Trigger Point'].iloc[-1]* 2)
data.loc[data['Trigger Value'] >= ylim, 'Trigger Value'] = ylim
data['Trigger Value']=data['Trigger Value'].replace(0, np.nan)

if os.name == 'nt':
    ax1.set_title('Windows - Average Peak Movement')
else:
    ax1.set_title(str(os.uname()[1]) + ' - Average Peak Movement')
    
ax1.plot(time, data['Subtraction Threshold'],
         color='red',
         alpha=0.5,
         label='Subtraction Threshold'
        )
        
ax1.plot(time, data['Subtraction History'],
         color='black',
         alpha=0.5,
         label='Subtraction History'
        )
    
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
         label='Trigger Point Base'
        )
        
ax1.plot(time, data['Trigger Point Frames'],
         color='orange',
         alpha=0.5,
         label='Trigger Point Frames'
        )
        
ax1.plot(time, data['Trigger Point'],
         color='Green',
         alpha=0.5,
         label='Trigger point'
        )

ax1.set_ylim([0, ylim])

# plt.legend()

if os.name == 'nt':
    ax2.set_title('Windows - Maximum Peak Movement')
else:
    ax2.set_title(str(os.uname()[1]) + ' - Maximum Peak Movement')


ax2.plot(time, data['Trigger Point'],
         color='Green',
         alpha=0.5,
         label='Trigger point'
        )
        
        
ax2.plot(time, data['Trigger Point Frames'],
         color='orange',
         alpha=0.5,
         label='Trigger Point Frames'
        )

ax1.plot(time, data['Trigger Point Base'],
         color='yellow',
         alpha=0.5,
         label='Trigger Point Base'
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

ax2.set_ylim([0, ylim])

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%a %H')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()
plt.ylabel('Movement')
# plt.legend()
# position at which legend to be added
ax1.legend(loc='best')
ax2.legend(loc='best')

plt.savefig('peak.png')

if os.name == 'nt':
    plt.show()
