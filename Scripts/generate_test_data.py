import extract_features
import numpy as np
import os
import settings
import subprocess
import csv
import sys

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


def process_file(window_size, overlap_size, cur_file):

    cols = ["Time_s", "hand_accx_16g", "hand_accy_16g", "hand_accz_16g", "gesture"]

    print 'Extracting Testing Features for ' + cur_file[:-4] + "\n"

    features_file_write = settings.phase_1_features + "/PAMAP2/Protocol/" + cur_file
    if os.path.isfile(features_file_write):
        os.remove(features_file_write)
    raw_data = np.genfromtxt(settings.phase_1_processed + "/PAMAP2/Protocol/" + cur_file, delimiter=',', dtype=None, names=True, usecols=cols, unpack=True)

    w = open(features_file_write, 'a+')
    writer = csv.writer(w)

    writer.writerow(features_header)

    T = [round(elem, 2) for elem in raw_data["Time_s"]]
    X = raw_data["hand_accx_16g"]
    Y = raw_data["hand_accy_16g"]
    Z = raw_data["hand_accz_16g"]
    activity = raw_data["gesture"]

    window_time = 0
    cur_window_time = window_time
    print T[len(T) - 1]
    while T[len(T) - 1] - T[window_time] >= float(window_size):

        if round(T[window_time] - T[cur_window_time], 2) >= float(window_size):
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

            window_time = cur_window_time

            while T[cur_window_time] < round(T[window_time] + float(overlap_size), 2):
                cur_window_time += 1

            window_time = cur_window_time
        else:
            window_time += 1
    print T[window_time]
    w.close()
    print 'Done'
    print ''


def process_file_bash(cur_file, window_size, overlap_size):
    settings.init()

    process_file(float(window_size), float(overlap_size), cur_file)


def extract_peaks(window_size, overlap_size):
    """Segment data into windows, extracts features

    Given raw accelerometer data, this function segments the data into windows, and calls the function to extract features

    Args:
        window_size (int): Size of the window
        overlap_size (int): Size of the overlap between windows. Typically half of the window size
    """
    # process_file(window_size, overlap_size, 'subject101.csv')
    subprocess.check_call([settings.scripts + "/process_test_data.sh", settings.phase_1_processed + "/PAMAP2/Protocol/", str(window_size), str(overlap_size)])


def main(window_size, overlap_size):

    extract_peaks(window_size, overlap_size)


if __name__ == '__main__':
    settings.init()

    main(*sys.argv[1:])
