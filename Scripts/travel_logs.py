import settings
import os
from datetime import datetime

def get_transition_times_from_file(file_path):
    transition_times = set()
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split(',')
            log_date = split_line[1]
            log_start = split_line[3]
            log_end = split_line[4]
            if not split_line[0].isdigit(): # Header or separator row
                continue
            try: # Check if times have seconds included
                start_time = datetime.strptime(log_date + ' ' + log_start, '%m/%d/%Y %H:%M:%S %p')
                end_time = datetime.strptime(log_date + ' ' + log_end, '%m/%d/%Y %H:%M:%S %p')
            except Exception as e:
                try: # Time just has hours and minutes
                    start_time = datetime.strptime(log_date + ' ' + log_start, '%m/%d/%Y %H:%M %p')
                    end_time = datetime.strptime(log_date + ' ' + log_end, '%m/%d/%Y %H:%M %p')
                except Exception as e:
                    print(e)

            transition_times.add(start_time.isoformat())
            transition_times.add(end_time.isoformat())
        transition_list = list(transition_times)
        transition_list.sort()

    return transition_list
