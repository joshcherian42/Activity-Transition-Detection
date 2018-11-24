import extract_features
import numpy as np
import os
import settings
import subprocess
import csv

features_header = ['Avg Jerk E',
                   'Avg Height E',
                   'Stdev Height E',
                   'Energy E',
                   'Entropy E',
                   'Average E',
                   'Standard Deviation E',
                   'RMS E',
                   'Num Peaks E',
                   'Average Peaks E',
                   'Standard Deviation Peaks E',
                   'Num Valleys E',
                   'Average Valleys E',
                   'Standard Deviation Valleys E',
                   'Axis Overlap',
                   'Activity',
                   'Start', 'End']


def process_file(window_size, cur_file):

    cols = ["Epoch_Time", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "Speed", "Avg_Speed", "Standard_Distance", "Activity"]

    print 'Extracting Testing Features for ' + cur_file[:-4] + "\n"

    features_file_write = settings.phase_1_features + "/" + cur_file
    if os.path.isfile(features_file_write):
        os.remove(features_file_write)
    raw_data = np.genfromtxt(settings.phase_1_raw + "/" + cur_file, delimiter=',', dtype=None, names=True, usecols=cols, unpack=True)

    w = open(features_file_write, 'a+')
    writer = csv.writer(w)

    writer.writerow(features_header)

    T = raw_data["Epoch_Time"]
    X = raw_data["Accelerometer_X"]
    Y = raw_data["Accelerometer_Y"]
    Z = raw_data["Accelerometer_Z"]

    activity = raw_data["Activity"]

    window_time = 0
    cur_window_time = window_time

    while T[len(T) - 1] - T[window_time] >= window_size:
        if T[window_time] - T[cur_window_time] > window_size:
            extract_features.parse_data(window_size,
                                        T[cur_window_time:window_time],
                                        X[cur_window_time:window_time],
                                        Y[cur_window_time:window_time],
                                        Z[cur_window_time:window_time],
                                        T[cur_window_time],
                                        T[window_time],
                                        activity[cur_window_time:window_time],
                                        features_file_write,
                                        cur_file,
                                        writer)

            cur_window_time = window_time  # assumes overlap size is half
        window_time += 1
    w.close()
    print 'Done'
    print ''


def process_file_bash(cur_file, window_size):
    settings.init()
    process_file(int(window_size), cur_file)


def extract_peaks(window_size):
    """Segment data into windows, extracts features

    Given raw accelerometer data, this function segments the data into windows, and calls the function to extract features

    Args:
        window_size (int): Size of the window
        overlap_size (int): Size of the overlap between windows. Typically half of the window size
    """

    subprocess.check_call([settings.scripts + "/process_test_data.sh", settings.phase_1_raw, str(window_size)])


def main(window_size):

    extract_peaks(window_size)
