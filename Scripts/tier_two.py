import os
import csv
from datetime import datetime
import settings
import travel_logs

# Generate features for 1 person given a list of their phase 1 output files in time order and the travel log as phase 2 raw file
# File names are relative to settings.phase_1_out and settings.phase_2_raw
# Also takes the number of pre windows and post windows to generate the probability features for
def generate_features(phase_1_out_files, phase_2_raw_file, pre_windows, post_windows):
    window_data = []

    for phase_1_file in phase_1_out_files:
        # Get window data
        firstLine = True
        with open(os.path.join(settings.phase_1_output, phase_1_file), 'r') as f:
            for line in f:
                if firstLine:
                    firstLine = False
                    data_keys = line.strip().split(',')
                    continue

                temp = {}
                split_line = line.split(',')
                for i in range(len(data_keys)):
                    temp[data_keys[i]] = split_line[i].strip()
                window_data.append(temp)

    data_points = [] # Array of data point dicts
    for i in range(pre_windows, len(window_data) - post_windows): # Pre window indices will be i-1 to i-pre_windows. Post window indices will be i to i+post_windows-1
        point = {}
        point['time'] = datetime.strptime(window_data[i]['Start'], '%m-%d-%Y %H:%M:%S.%f')
        point['primary_class_change'] = window_data[i]['Primary Prediction'] != window_data[i - 1]['Primary Prediction']
        point['secondary_class_change'] = window_data[i]['Secondary Prediction'] != window_data[i - 1]['Secondary Prediction']

        for j in range(i-pre_windows, i):
            point_idx = 'n-' + str(i - j) + '_'
            window = window_data[j]
            point[point_idx + 'primary_proba'] = float(window['Primary Probability'])
            point[point_idx + 'secondary_proba'] = float(window['Secondary Probability'])
            point[point_idx + 'confidence_margin'] = point[point_idx + 'primary_proba'] - point[point_idx + 'secondary_proba']
            point[point_idx + 'confidence_margin_normalized'] = point[point_idx + 'confidence_margin'] / point[point_idx + 'primary_proba']

        for j in range(i, i + post_windows):
            point_idx = 'n+' + str(j - i + 1) + '_'
            window = window_data[j]
            point[point_idx + 'primary_proba'] = float(window['Primary Probability'])
            point[point_idx + 'secondary_proba'] = float(window['Secondary Probability'])
            point[point_idx + 'confidence_margin'] = point[point_idx + 'primary_proba'] - point[point_idx + 'secondary_proba']
            point[point_idx + 'confidence_margin_normalized'] = point[point_idx + 'confidence_margin'] / point[point_idx + 'primary_proba']

        point['output'] = 0
        data_points.append(point)

    # Change output class to 1 if it is a transition
    transition_times = travel_logs.get_transition_times_from_file(os.path.join(settings.phase_2_raw, phase_2_raw_file))
    data_idx = 0
    for time in transition_times:
        while data_idx < len(data_points):
            if data_points[data_idx]['time'] > time:
                data_points[data_idx]['output'] = 1
                break # Out of while loop but continue for loop
            data_idx += 1

    with open(os.path.join(settings.phase_2_features, phase_2_raw_file), 'w+') as fout:
        writer = csv.writer(fout)
        header = data_points[0].keys()
        header.sort()
        header.remove('time')
        header.remove('primary_class_change')
        header.remove('secondary_class_change')
        header.remove('output')

        header.insert(0, 'time')
        header.insert(1, 'primary_class_change')
        header.insert(2, 'secondary_class_change')
        header.append('output')

        writer.writerow(header)
        for point in data_points:
            writer.writerow([point[key] for key in header])

settings.init()
paths = []
base_name = ''
for subdir, dirs, files in os.walk(settings.phase_1_output):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                cur_base_name = '_'.join(cur_file.split('_')[:-3]) # Removes _YYYY-MM-DD_Hr_H.csv
                if base_name == '':
                    base_name = cur_base_name
                if cur_base_name != base_name: # Changed to a new user
                    generate_features(paths, base_name + '.csv', 5, 5)
                    base_name = cur_base_name
                    paths = []
                else:
                    paths.append(cur_file)

