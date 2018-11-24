import numpy as np
import os
import pandas as pd
import csv
from datetime import datetime
import settings
import generate_models
from shutil import copyfile

lag = 1000
threshold = 10
influence = 1


def gen_output(algorithm):
    """Generates Test Data

    Iterates throughtest data and calls gen_output_window to classify each file

    Args:
        algorithm (str): Algorithm to use when classifying each window
    """

    for subdir, dirs, files in os.walk(settings.phase_1_features):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):

                print "Classifying Activities for " + cur_file[:-4] + "\n"

                output_file = settings.phase_1_output + "/" + cur_file

                gen_output_window(algorithm, subdir + "/" + cur_file, output_file)


def gen_output_window(algorithm, features_file, output_file):  # mueller_col, dates_col, prev_mueller_index, mueller_index):
    """Classifies windows of data into activities

    Takes the features extracted from raw accelerometer and GPS data and classifies each window as an activity

    Args:
        algorithm (str): Algorithm to use when classifying each window
        features_file (str): path to file containing features to be classified
        output_file (str): path to file containing classifications

    Returns:
        list: returns the time spent doing each activity, the previous mueller index, and the current mueller index
    """

    header = ['Predicted', 'Start', 'End', 'Primary Probability', 'Primary Prediction', 'Secondary Probability', 'Secondary Prediction']
    test_data = pd.read_csv(features_file)

    try:
        os.remove(output_file)
    except OSError:
        pass

    w = open(output_file, 'a+')
    writer = csv.writer(w)
    writer.writerow(header)

    total_time = 0

    clf_activities = generate_models.test_algorithm(algorithm, test_data)

    for index, activity in enumerate(clf_activities[0]):

        cur_activity = activity

        start = clf_activities[1][index]
        # start_next = test_data.iloc[index + 1, 16]
        end = clf_activities[2][index]
        primary_prob = clf_activities[3][index]
        primary_pred = clf_activities[4][index]
        secondary_prob = clf_activities[5][index]
        secondary_pred = clf_activities[6][index]
        cur_time = float(end) - float(start)

        start_time = datetime.utcfromtimestamp(float(float(start) / 1000)).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3]
        # start_time_next = datetime.utcfromtimestamp(float(start_next / 1000)).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3]
        end_time = datetime.utcfromtimestamp(float(float(end) / 1000)).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3]
        total_time += cur_time / 2

        # TEMPORARY FIX
        if cur_activity == 'driv':
            cur_activity = 'driving'
        elif cur_activity == 'biki':
            cur_activity = 'biking'
        elif cur_activity == 'walk':
            cur_activity = 'walking'
        elif cur_activity == 'stro':
            cur_activity = 'strolling'
        elif cur_activity == 'jogg':
            cur_activity = 'jogging'
        elif cur_activity == 'runn':
            cur_activity = 'running'
        elif cur_activity == 'putt':
            cur_activity = 'putting on the table'
        elif cur_activity == 'tran':
            cur_activity = 'transit'
        elif cur_activity == 'stan':
            cur_activity = 'standing'
        elif cur_activity == 'sitt':
            cur_activity = 'sitting'

        writer.writerow([cur_activity, start_time, end_time, primary_prob, primary_pred, secondary_prob, secondary_pred])
    w.close()