import numpy as np
import os
import pandas as pd
import csv
from datetime import datetime
import tiertwo
import settings
import generate_models
from shutil import copyfile

lag = 1000
threshold = 10
influence = 1


def gen_output(algorithm, trial, paths):
    """Generates Test Data

    Iterates throughtest data and calls gen_output_window to classify each file

    Args:
        algorithm (str): Algorithm to use when classifying each window
    """

    log_header = ['User', 'Date', 'Activity', 'Time (min)', 'Percentage (%)']

    activities = list()
    for activity in settings.confusion_matrix_labels:
        activities.append([activity, 0])  # Not Mueller
        if settings.trial != 'Team Data':
            activities.append([activity, 0])  # Mueller

    w = open(settings.log_file, 'wb+')
    log_writer = csv.writer(w)
    log_writer.writerow(log_header)

    cur_user = ''
    cur_date = ''
    case_control = ''
    mueller_index = 0
    prev_mueller_index = 0
    print settings.test_features
    for subdir, dirs, files in os.walk(settings.test_features):
        for cur_file in sorted(files, key=settings.natural_keys):
            path_check = subdir + "/" + cur_file
            path_check = path_check[path_check.index(trial) + len(trial) + 1:]
            if cur_file.endswith('.csv') and any(path in path_check for path in paths):
                if cur_user != '' and subdir.split("/")[-2] != cur_user:
                    total_time = sum(act_time for _, act_time in activities)
                    for activitiy in activities:
                        log_writer.writerow([cur_user, cur_date, activity[0], activity[1] / 60000, activity[1] / total_time * 100])
                        activity[1] = 0

                cur_user = subdir.split("/")[-2]
                case_control = subdir.split("/")[-3]
                if cur_date != subdir.split("/")[-1]:
                    cur_date = subdir.split("/")[-1]

                    if settings.trial != 'Team Data':
                        mueller_file = np.genfromtxt(settings.gps_folder + "/" + case_control + "/" + cur_user + '/All_GPS.csv', delimiter=',', dtype=None, names=True, unpack=True, encoding=None)  # change to gps_edited folder
                        mueller_col = mueller_file['IN_NB']
                        dates = mueller_file['Epoch_Time']

                if not os.path.exists(settings.output_src + "/" + case_control + "/" + cur_user + "/" + cur_date):
                        os.makedirs(settings.output_src + "/" + case_control + "/" + cur_user + "/" + cur_date)

                update_log_file = open(settings.website_log_local, "a")
                update_log_file.write(' ' * int(1.5 * len("Classifying Activities....")) + 'for ' + cur_user + "/" + cur_date + "/" + cur_file[:-4] + "\n")
                update_log_file.close()

                output_file = settings.output_src + "/" + case_control + "/" + cur_user + "/" + cur_date + "/" + cur_file
                if settings.trial == 'Team Data':
                    times = gen_output_window(algorithm, subdir + "/" + cur_file, output_file)
                else:
                    out = gen_output_window(algorithm, subdir + "/" + cur_file, output_file, mueller_col=mueller_col, dates=dates, prev_mueller_index=prev_mueller_index, mueller_index=mueller_index)
                    times = out[0]
                    prev_mueller_index = out[1]
                    mueller_index = out[2]

                for act_index, activity in enumerate(activities):  # need to iterate over half of this
                    if sum(times) != 0:
                        log_writer.writerow([cur_user, cur_date + ' ' + cur_file[:-4], activity[0], times[act_index] / (60000), times[act_index] / sum(times) * 100])  # 2*60000, the 2 is a temp fix
                        activity[1] += (times[act_index])

    # w.close()


def gen_output_window(algorithm, features_file, output_file, **kwargs):  # mueller_col, dates_col, prev_mueller_index, mueller_index):
    """Classifies windows of data into activities

    Takes the features extracted from raw accelerometer and GPS data and classifies each window as an activity

    Args:
        algorithm (str): Algorithm to use when classifying each window
        features_file (str): path to file containing features to be classified
        output_file (str): path to file containing classifications
        **mueller_col (list): list of 0s (Not Mueller) and 1s (Mueller)
        **dates (list): list of dates related to the mueller_col
        **prev_mueller_index (int): index of previous row in the gps file containing the mueller information
        **mueller_index (int): index of current row in the gps file containing the mueller information

    Returns:
        list: returns the time spent doing each activity, the previous mueller index, and the current mueller index
    """

    header = ['Predicted', 'Start', 'End', 'Time', 'AccGPS']
    test_data = pd.read_csv(features_file)

    try:
        os.remove(output_file)
    except OSError:
        pass

    w = open(output_file, 'a+')
    writer = csv.writer(w)
    writer.writerow(header)

    total_time = 0
    if 'mueller_col' in kwargs:
        activity_time = [0] * len(settings.confusion_matrix_labels) * 2
        mueller_index = 0
    else:
        activity_time = [0] * len(settings.confusion_matrix_labels)
    clf_activities = generate_models.test_algorithm(algorithm, test_data)

    #for index in range(len(test_data) - 1):
    for index, activity in enumerate(clf_activities[0]):

        # activity = j48.j48_algorithm(test_data.iloc[index], False)
        cur_activity = activity

        start = clf_activities[1][index]
        # start_next = test_data.iloc[index + 1, 16]
        end = clf_activities[2][index]
        cur_time = float(end) - float(start)

        start_time = datetime.utcfromtimestamp(float(float(start) / 1000)).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3]
        # start_time_next = datetime.utcfromtimestamp(float(start_next / 1000)).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3]
        end_time = datetime.utcfromtimestamp(float(float(end) / 1000)).strftime('%m-%d-%Y %H:%M:%S.%f')[:-3]
        total_time += cur_time / 2

        start_time_date = datetime.strptime(start_time, '%m-%d-%Y %H:%M:%S.%f')
        # if acc_time[act_id] >= cur_GPS_time and acc_time[act_id] <= next_GPS_time and next_GPS_time < acc_time[act_id] + 60000:

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
        if 'dates' in kwargs:
            dates = kwargs["dates"]
            mueller_col = kwargs["mueller_col"]
            prev_mueller_index = kwargs["prev_mueller_index"]
            while mueller_index < len(dates) and datetime.utcfromtimestamp(dates[mueller_index] / 1000) < start_time_date:  # WON'T WORK FOR DATA LOSS
                mueller_index += 1

            if mueller_index == len(dates):
                try:
                    mueller = prev_mueller_index
                except UnboundLocalError:
                    print 'Last file reached' + "\n"
                    mueller = 0
            else:
                mueller = mueller_col[mueller_index]
                if mueller != 1:
                    mueller = 0
                prev_mueller_index = mueller

            activity_time[2 * settings.confusion_matrix_labels.index(cur_activity) + mueller] += cur_time / 2
        else:

            activity_time[settings.confusion_matrix_labels.index(cur_activity)] += cur_time / 2
        writer.writerow([cur_activity, start_time, end_time, (float(end) - float(start)) / 60000, clf_activities[3][index]])
    w.close()
    if 'mueller_col' in kwargs:
        return [activity_time, prev_mueller_index, mueller_index]
    else:
        return activity_time


def correct_output(mueller_flag, phase_one, tier_two_size, scoring_function, trial, paths):
    """Corrects the output using a tier two algorithm

    Corrects the output classification using a tier two algorithm

    Args:
        mueller_flag (bool): If true, add mueller/not mueller data
        phase_one (int): Size of Tier I window
        tier_two_size (int): Size of Tier II window
    """

    header = ["Predicted", "Start", "End", "Time", "AccGPS"]
    log_header = ['User', 'Date', 'Activity', 'Time (min)', 'Percentage (%)']

    w = open(settings.log_file_corrected, 'wb+')
    log_writer = csv.writer(w)
    log_writer.writerow(log_header)

    cur_id = ''
    cur_day = ''
    for subdir, dirs, files in os.walk(settings.output_src):
        for cur_file in sorted(files, key=settings.natural_keys):
            path_check = subdir + "/" + cur_file
            path_check = path_check[path_check.index(trial) + len(trial) + 1:]
            if cur_file.endswith('.csv') and any(path in path_check for path in paths):
                update_log_file = open(settings.website_log_local, "a")
                update_log_file.write(' ' * int(1.5 * len("Correcting Classifications....")) + 'for ' + subdir.split("/")[-4] + "/" + subdir.split("/")[-3] + "/" + subdir.split("/")[-2] + "/" + subdir.split("/")[-1] + "/" + cur_file[:-4] + "\n")
                update_log_file.close()
                mueller_index = 0
                prev_mueller_index = 0

                tier_two = tiertwo.weighted_moving_average(tier_two_size, scoring_function)

                output_file = np.genfromtxt(subdir + "/" + cur_file, delimiter=',', dtype=None, names=True, usecols=header, unpack=True, encoding=None)
                if len(np.atleast_1d(output_file["Predicted"])) == 1:
                    copyfile(subdir + "/" + cur_file, settings.output_corrected + "/" + cur_id + "/" + cur_day + "/" + cur_file)
                else:
                    corrected_output = [[] for x in xrange(len(output_file["Predicted"]))]

                    cur_trial = subdir.split("/")[-4]
                    case_control = subdir.split("/")[-3]
                    cur_user = subdir.split("/")[-2]
                    cur_date = subdir.split("/")[-1]
                    if not os.path.exists(settings.output_corrected + "/" + cur_trial + "/" + case_control + "/" + cur_user + "/" + cur_date):
                        os.makedirs(settings.output_corrected + "/" + cur_trial + "/" + case_control + "/" + cur_user + "/" + cur_date)

                    predicted = output_file["Predicted"]
                    start = output_file["Start"]
                    end = output_file["End"]
                    time = output_file["Time"]
                    accgps = output_file["AccGPS"]

                    for index, pred_activity in enumerate(predicted):
                        corrected_activities = tier_two.new_window(pred_activity)
                        corrected_output[index] = [pred_activity, start[index], end[index], time[index], accgps[index]]

                        if corrected_activities != '':
                            corrected_output[index - tier_two_size / 2 - 1][0] = corrected_activities

                    # Commented out code below is for log file
                    activities = list()
                    for activity in settings.confusion_matrix_labels:
                        activities.append([activity, 0])  # Not Mueller
                        if mueller_flag:
                            activities.append([activity, 0])  # Mueller
                    if mueller_flag:
                        mueller_file = np.genfromtxt(settings.gps_folder + "/" + case_control + "/" + cur_user + '/All_GPS.csv', delimiter=',', dtype=None, names=True, unpack=True, encoding=None)  # change to gps_edited folder
                        mueller_col = mueller_file['IN_NB']
                        dates = mueller_file['Epoch_Time']

                    for activity in corrected_output:

                        start_time_date = datetime.strptime(activity[1], '%m-%d-%Y %H:%M:%S.%f')
                        if mueller_flag:

                            while mueller_index < len(dates) and datetime.utcfromtimestamp(dates[mueller_index] / 1000) < start_time_date:  # WON'T WORK FOR DATA LOSS
                                mueller_index += 1
                            if mueller_index == len(dates):
                                mueller = prev_mueller_index
                            else:
                                mueller = mueller_col[mueller_index]
                                if mueller != 1:
                                    mueller = 0
                                prev_mueller_index = mueller
                            activities[2 * settings.confusion_matrix_labels.index(activity[0]) + mueller][1] += phase_one / 2.0
                        else:
                            activities[settings.confusion_matrix_labels.index(activity[0])][1] += phase_one / 2.0

                    for act_index, activity in enumerate(activities):
                        # this is temporary, really need to fix folders and filenames throughout
                        if sum(act_time for _, act_time in activities) != 0:
                            # log_writer.writerow([cur_user, cur_date + ' ' + cur_file[:-4], activity[0], times[act_index] / (60000), times[act_index] / sum(times) * 100])  # 2*60000, the 2 is a temp fix
                            if activity[1] != 0:
                                activity[1] += phase_one / 2
                            log_writer.writerow([cur_user, cur_date + ' ' + cur_file[:-4], activity[0], (activity[1]) / 60000, round(activity[1] / sum(act_time for _, act_time in activities) * 100, 2)])

                    w_corrected = open(settings.output_corrected + "/" + cur_trial + "/" + case_control + "/" + cur_user + "/" + cur_date + "/" + cur_file, 'wb')
                    writer_corrected = csv.writer(w_corrected)
                    writer_corrected.writerow(header)
                    writer_corrected.writerows(corrected_output)
                # print "Num Corrected: ", tier_two.num_corrected


def correct_wear(mueller_flag, phase_one, trial, paths):
    header = ["Predicted", "Start", "End", "Time", "AccGPS"]
    log_header = ['Trial', 'User', 'Date', 'Activity', 'Time (min)', 'Percentage (%)']

    w = open(settings.log_file_wear, 'wb+')
    log_writer = csv.writer(w)
    log_writer.writerow(log_header)
    for subdir, dirs, files in os.walk(settings.weartime):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('data validation.csv'):
                cur_trial = subdir.split("/")[-2]
                cur_user = cur_file.split("_")[0]
                with open(subdir + "/" + cur_file, 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    row_start = 0
                    for linenum, line in enumerate(reader):
                        if line[0] == 'Date/Time Start':
                            row_start = linenum
                            break
                wearcols = ['DateTime_Start', 'DateTime_Stop', 'Wear_or_NonWear']
                wear_reader = np.genfromtxt(subdir + "/" + cur_file, delimiter=',', dtype=None, names=True, unpack=True, usecols=wearcols, skip_header=row_start, encoding=None)
                starts = wear_reader['DateTime_Start']
                stops = wear_reader['DateTime_Stop']
                wear_nonwears = wear_reader['Wear_or_NonWear']

                wear_index = 0
                cur_wear_start = datetime.strptime(starts[wear_index], '%m/%d/%Y %H:%M')
                cur_wear_end = datetime.strptime(stops[wear_index], '%m/%d/%Y %H:%M')
                cur_wear = wear_nonwears[wear_index]
                print settings.output_corrected + "/" + trial + "/" + cur_trial + "/" + cur_user
                for output_subdir, output_dirs, output_files in os.walk(settings.output_corrected + "/" + trial + "/" + cur_trial + "/" + cur_user):
                    for cur_dir in sorted(output_dirs, key=settings.natural_keys):
                        if 'ALA' not in cur_dir and 'ALD' not in cur_dir and 'Case' not in cur_dir and 'Control' not in cur_dir:
                            for output_subsubdir, output_subdirs, output_subfiles in os.walk(output_subdir + "/" + cur_dir):
                                for output_file in sorted(output_subfiles, key=settings.natural_keys):
                                    path_check = output_subsubdir + "/" + output_file
                                    path_check = path_check[path_check.index(trial) + len(trial) + 1:]
                                    if output_file.endswith('.csv') and any(path in path_check for path in paths):
                                        update_log_file = open(settings.website_log_local, "a")
                                        update_log_file.write(' ' * int(1.5 * len("Removing Non-Wear Time....")) + 'for ' + output_subsubdir.split("/")[-4] + "/" + output_subsubdir.split("/")[-2] + "/" + output_subsubdir.split("/")[-1] + "/" + output_file[:-4] + "\n")
                                        update_log_file.close()

                                        corrected_lines = []

                                        cur_date = output_subsubdir.split("/")[-1]
                                        cur_user = output_subsubdir.split("/")[-2]
                                        cur_trial = output_subsubdir.split("/")[-4]

                                        mueller_index = 0
                                        prev_mueller_index = 0

                                        with open(output_subsubdir + "/" + output_file, 'rb') as out_file:

                                            output_reader = csv.reader(out_file, delimiter=',')
                                            next(output_reader, None)

                                            for linenum, line in enumerate(output_reader):

                                                out_start = datetime.strptime(line[1], '%m-%d-%Y %H:%M:%S.%f')
                                                out_stop = datetime.strptime(line[2], '%m-%d-%Y %H:%M:%S.%f')

                                                while (wear_index < len(starts) - 2 and
                                                        ((out_start < cur_wear_start and out_stop <= cur_wear_start) or
                                                         (out_start >= cur_wear_end or out_stop > cur_wear_end))):

                                                    wear_index += 1
                                                    cur_wear_start = datetime.strptime(starts[wear_index], '%m/%d/%Y %H:%M')
                                                    cur_wear_end = datetime.strptime(stops[wear_index], '%m/%d/%Y %H:%M')
                                                    cur_wear = wear_nonwears[wear_index]

                                                if line[0] == 'strolling':
                                                    line[0] = 'walking'
                                                elif line[0] == 'jogging':
                                                    line[0] = 'running'
                                                elif line[0] == 'sitting' or line[0] == 'standing' or line[0] == 'putting on the table':
                                                    line[0] = 'sedentary'
                                                elif line[0] == 'transit':
                                                    line[0] = 'driving'

                                                if cur_wear == 'Non-Wear':
                                                    line[0] = 'Non-Wear'

                                                corrected_lines.append(line)

                                        activities = list()
                                        for activity in settings.confusion_matrix_labels:
                                            activities.append([activity, 0])  # Not Mueller
                                            if mueller_flag:
                                                activities.append([activity, 0])  # Mueller
                                        activities.insert(0, ['Just Acc', 0])
                                        if mueller_flag:
                                            mueller_file = np.genfromtxt(settings.gps_folder + "/" + "Case" + "/" + cur_user + '/All_GPS.csv', delimiter=',', dtype=None, names=True, unpack=True, encoding=None)  # change to gps_edited folder
                                            mueller_col = mueller_file['IN_NB']
                                            dates = mueller_file['Epoch_Time']

                                        for activity in corrected_lines:

                                            start_time_date = datetime.strptime(activity[1], '%m-%d-%Y %H:%M:%S.%f')
                                            if mueller_flag:

                                                while mueller_index < len(dates) and datetime.utcfromtimestamp(dates[mueller_index] / 1000) < start_time_date:  # WON'T WORK FOR DATA LOSS
                                                    mueller_index += 1
                                                if mueller_index == len(dates):
                                                    mueller = prev_mueller_index
                                                else:
                                                    mueller = mueller_col[mueller_index]
                                                    if mueller != 1:
                                                        mueller = 0
                                                    prev_mueller_index = mueller
                                                if activity[0] != 'Non-Wear':
                                                    activities[2 * settings.confusion_matrix_labels.index(activity[0]) + mueller + 1][1] += phase_one / 2.0
                                            else:
                                                if activity[0] != 'Non-Wear':
                                                    activities[settings.confusion_matrix_labels.index(activity[0])][1] += phase_one / 2.0

                                            if activity[4] == 'Just Acc' and activity[0] != 'Non-Wear':
                                                activities[0][1] += phase_one / 2

                                        for act_index, activity in enumerate(activities):
                                            if sum(act_time for _, act_time in activities) != 0:
                                                if activity[1] != 0:
                                                    activity[1] += phase_one / 2
                                                log_writer.writerow([cur_trial, cur_user, cur_date + ' ' + output_file[:-4], activity[0], (activity[1]) / 60000, round(activity[1] / sum(act_time for _, act_time in activities) * 100, 2)])

                                        if not os.path.exists(settings.output_wear + "/" + cur_trial + "/" + "Case" + "/" + cur_user + "/" + cur_date):
                                            os.makedirs(settings.output_wear + "/" + cur_trial + "/" + "Case" + "/" + cur_user + "/" + cur_date)
                                        w_corrected = open(settings.output_wear + "/" + cur_trial + "/" + "Case" + "/" + cur_user + "/" + cur_date + "/" + output_file, 'wb')
                                        writer_corrected = csv.writer(w_corrected)
                                        writer_corrected.writerow(header)
                                        writer_corrected.writerows(corrected_lines)


def main(valid_hours, trial, paths, params):
    algorithm = params['classifier']
    tier_two_size = params['phase_two_window_size']
    scoring_function = params['scoring_function']
    phase_one = params['phase_one_window_size']

    overall_file = os.getcwd() + "/Process Data/Separated Raw Test Data/log-overall.txt"
    
    log_files = os.getcwd() + "/Process Data/Separated Raw Test Data/log-files.txt"
    
