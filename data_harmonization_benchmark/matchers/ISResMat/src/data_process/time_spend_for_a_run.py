'''
This is simply a tool used in development, not intended for
reporting the algorithm's execution time (which is documented in the result log file).
'''

import datetime
import os
import re

import pytz


def get_creation_time(file_path):
    ctime = os.path.getctime(file_path)
    dt = datetime.datetime.fromtimestamp(ctime, pytz.UTC)
    dt=dt.astimezone(pytz.timezone('Asia/Shanghai'))
    return dt.strftime('%Y-%m-%d_%H.%M.%S--')

all_files=[]
directory_path = '../../data/output/inst-001'
for root,_,files in os.walk(directory_path):
    for f in files:
        if f.endswith('.json'):
            ctime=get_creation_time(os.path.join(root,f))
            all_files.append(ctime+f)

all_files=sorted(all_files)
print(directory_path.split('/')[-1])
print(len(all_files))

begin_time=all_files[0]
end_time=all_files[-1]

print(begin_time)
print(end_time)

# Define a regular expression pattern to match the time part at the beginning of the strings
pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}'

# Extract the time part from the strings
begin_time_str = re.match(pattern, begin_time).group()
end_time_str = re.match(pattern, end_time).group()

# Convert the time strings to datetime objects
time1 = datetime.datetime.strptime(begin_time_str, '%Y-%m-%d_%H.%M.%S')
time2 = datetime.datetime.strptime(end_time_str, '%Y-%m-%d_%H.%M.%S')

# Calculate the time difference
time_diff = time2 - time1

# Convert the time difference to hours and minutes
hours = time_diff.seconds // 3600
minutes = (time_diff.seconds % 3600) // 60

# Format the result as hours:min
result = f"{hours}:{minutes:02d}"

print('spend: ',result)