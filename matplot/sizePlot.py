import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
import os

plt.style.use('seaborn')

data = pd.read_csv('size.csv')
data = data.tail(800)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.sort_values('Timestamp', inplace=True)
time = data['Timestamp']
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%a %H %M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()

plt.plot(time, data['Size'], 
         color='Black',
         label='Total Size'
        )
        
plt.plot(time, data['Used'], 
         color='Blue',
         label='Used'
        )

plt.plot(time, data['Available'], 
         color='Red',
         label='Available'
         )
         
plt.ylim(bottom=0)

if os.name == 'nt':
    plt.title('Windows - Disk Size')
else:
    plt.title(str(os.uname()[1]) + ' - Disk Size')
plt.ylabel('Megabytes')
plt.legend()
plt.savefig('size.png')

if os.name == 'nt':
    plt.show()
