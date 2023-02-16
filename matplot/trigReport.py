import pandas as pd
import numpy as np
import openpyxl
import os

df = pd.read_csv('peakMovement.csv')
# df = df.tail(240)

if os.name == 'nt':
    host = 'Windows'
else:
    host = str(os.uname()[1])
    
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df = df.loc[df['Trigger Value'] > 0]
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour

df = df.groupby(['Day','Hour','Trigger Point', 'Frames Checked', 'Subtraction Threshold', 'Subtraction History']).size().reset_index(name='Triggers')

df.to_csv('triggers.csv', sep=',', encoding='utf-8')

df.to_excel('triggers.xlsx', sheet_name=host + ' triggers')

if os.name == 'nt':
    print(df)