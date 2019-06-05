import extract_features
import pamap_features
import numpy as np
import os
import settings
import subprocess
import csv
import sys
from importlib import import_module

def process_file(window_size, overlap_size, cur_file):

    print 'Extracting Testing Features for ' + cur_file + "\n"

    features_file_write = cur_file.replace('processed', 'features')
    if os.path.isfile(features_file_write):
        os.remove(features_file_write)

    raw_data = np.genfromtxt(cur_file, delimiter=',', dtype=None, names=True, usecols=settings.raw_data_cols, unpack=True, encoding=None)
    
    w = open(features_file_write, 'a+')
    writer = csv.writer(w)

    writer.writerow(settings.features_header)

    T = [round(elem, 2) for elem in raw_data["Time_s"]]

    window_time = 0
    cur_window_time = window_time

    while T[len(T) - 1] - T[window_time] >= float(window_size):

        if round(T[window_time] - T[cur_window_time], 2) >= float(window_size):

            dataset.parse_data(raw_data, cur_window_time, window_time, cur_file.split("/")[-1][7:10], writer)

            window_time = cur_window_time

            while T[cur_window_time] < round(T[window_time] + float(overlap_size), 2):
                cur_window_time += 1

            window_time = cur_window_time
        else:
            window_time += 1

    w.close()
    print 'Done'
    print ''


def process_file_bash(cur_file, window_size, overlap_size, dataset):
    settings.init(dataset)

    import_features(settings.dataset)

    process_file(float(window_size), float(overlap_size), cur_file)


def import_features(data):
    global dataset

    try:
        if dataset:
            print 'already imported'
            return
    except NameError:
        pass

    if data == 'PAMAP2':
        import pamap_features as dataset


def extract_peaks(window_size, overlap_size):
    """Segment data into windows, extracts features

    Given raw accelerometer data, this function segments the data into windows, and calls the function to extract features

    Args:
        window_size (int): Size of the window
        overlap_size (int): Size of the overlap between windows. Typically half of the window size
    """
    import_features(settings.dataset)

    # process_file(window_size, overlap_size, '/Users/joshcherian/Documents/GitHub/Activity-Transition-Detection/Data/phase_1/processed/PAMAP2/Protocol/subject101.csv')


    subprocess.check_call([settings.scripts + "/process_test_data.sh", str(window_size), str(overlap_size), settings.dataset, " ".join(get_file_names())])


def get_file_names():
    filenames = []
    print settings.phase_1_processed
    for subdir, dirs, files in os.walk(settings.phase_1_processed):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                filenames.append(subdir + "/" + cur_file)

    return filenames

def main(window_size, overlap_size):

    extract_peaks(window_size, overlap_size)


if __name__ == '__main__':
    settings.init()

    main(*sys.argv[1:])
