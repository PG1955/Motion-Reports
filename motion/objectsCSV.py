import os
import time
from datetime import datetime
import csv

"""
MovementsCSV class.

Writes a CSV file containing peak movements CSV file.
"""

class ObjectsCSV:
    def __init__(self, debug=False):
        self.debug = debug
        self.filename = None
        self.object = None
        self.now = datetime.now()
        self.csv_file = "objects.csv"
        self.columns = ['Timestamp', 'Filename', 'Object']
        if not os.path.isfile(self.csv_file):
            self.create()

    def create(self):
        if self.debug:
            print('CSV:create')
        with open(self.csv_file, 'w', newline='') as file:
            # creating a csv dict writer object
            _writer = csv.DictWriter(file, fieldnames=self.columns)
            # writing headers (field names)
            _writer.writeheader()


    """
    Write a record
    """
    def write(self, filename, item):
        if self.debug:
            print('CSV:write')
        if not os.path.isfile(self.csv_file):
            self.create()
        with open(self.csv_file, 'a', newline='') as file:
            _writer = csv.DictWriter(file, fieldnames=self.columns)
            timestamp = self.now.strftime("%Y-%m-%d %H:%M")
            return _writer.writerow({"Timestamp": timestamp,
                                     'Filename': filename,
                                     'Object': item})


# if os.path.exists('objects.csv'):
#     os.remove('objects.csv')
# ocsv = ObjectsCSV(debug=True)
# objects = ['Person','Chair','Sunhat','Sheep','Person']
# for n in range(5):
#     ocsv.write('file1.jpg', objects[n - 1])
