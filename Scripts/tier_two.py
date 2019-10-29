import os
import csv
from datetime import datetime
import settings
import travel_logs
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import generate_models
import shutil
import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier


count = 0
best = 0
best_params = {}
# Generate features for 1 person given a list of their phase 1 output files in time order and the travel log as phase 2 raw file
# File names are relative to settings.phase_1_out and settings.phase_2_raw
# Also takes the number of pre windows and post windows to generate the probability features for


def generate_features(phase_1_out_files, phase_2_raw_file, pre_windows, post_windows, train_test):
    window_data = []

    for phase_1_file in os.scandir(phase_1_out_files):

        if phase_1_file.path.endswith('.csv'):
            print('Processing:', phase_1_file.path.split('/')[-1])
    # for phase_1_file in phase_1_out_files:
    #     # Get window data
    #     firstLine = True
    #     with open(os.path.join(settings.phase_1_output, phase_1_file), 'r') as f:
    #         for line in f:
    #             if firstLine:
    #                 firstLine = False
    #                 data_keys = line.strip().split(',')
    #                 continue

    #             temp = {}
    #             split_line = line.split(',')
    #             for i in range(len(data_keys)):
    #                 temp[data_keys[i]] = split_line[i].strip()
    #             window_data.append(temp)

            window_data = pd.read_csv(phase_1_file.path)
            data_points = []  # Array of data point dicts
            for i in range(pre_windows, len(window_data.index) - post_windows):  # Pre window indices will be i-1 to i-pre_windows. Post window indices will be i to i+post_windows-1
                point = {}
                # point['time'] = datetime.strptime(window_data[i]['Start'], '%m-%d-%Y %H:%M:%S.%f')
                point['primary_class_change'] = window_data.iloc[i]['Primary Prediction'] != window_data.iloc[i - 1]['Primary Prediction']
                point['secondary_class_change'] = window_data.iloc[i]['Secondary Prediction'] != window_data.iloc[i - 1]['Secondary Prediction']

                for j in range(i - pre_windows, i):
                    point_idx = 'n-' + str(i - j) + '_'
                    window = window_data.iloc[j]
                    point[point_idx + 'primary_proba'] = float(window['Primary Probability'])
                    point[point_idx + 'secondary_proba'] = float(window['Secondary Probability'])
                    point[point_idx + 'confidence_margin'] = point[point_idx + 'primary_proba'] - point[point_idx + 'secondary_proba']
                    point[point_idx + 'confidence_margin_normalized'] = point[point_idx + 'confidence_margin'] / point[point_idx + 'primary_proba']

                for j in range(i, i + post_windows):
                    point_idx = 'n+' + str(j - i + 1) + '_'
                    window = window_data.iloc[j]
                    point[point_idx + 'primary_proba'] = float(window['Primary Probability'])
                    point[point_idx + 'secondary_proba'] = float(window['Secondary Probability'])
                    point[point_idx + 'confidence_margin'] = point[point_idx + 'primary_proba'] - point[point_idx + 'secondary_proba']
                    point[point_idx + 'confidence_margin_normalized'] = point[point_idx + 'confidence_margin'] / point[point_idx + 'primary_proba']

                point['Actual'] = window_data.iloc[i]['Actual']
                data_points.append(point)

            # Change output class to 1 if it is a transition. Remove if it's more than 5 min from a transition time
            # transition_times = travel_logs.get_transition_times_from_file(os.path.join(raw_features, phase_2_raw_file))
            # data_idx = 0
            # for i, transition_time in enumerate(transition_times):
            #     while data_idx < len(data_points):
            #         diff = data_points[data_idx]['time'] - transition_time  # Time diff to next transition
            #         diff = diff.total_seconds()
            #         diff2 = 301
            #         if i != 0:
            #             diff2 = data_points[data_idx]['time'] - transition_times[i - 1]  # Time diff to previous transition
            #             diff2 = diff2.total_seconds()

            #         # Remove data more than this many seconds away from a transition
            #         if abs(diff) > settings.phase_2_remove_window_size_secs and abs(diff2) > settings.phase_2_remove_window_size_secs:
            #             data_points.pop(data_idx)
            #             continue

            #         if data_points[data_idx]['time'] > transition_time:
            #             data_points[data_idx]['output'] = 1
            #             break  # Out of while loop but continue for loop

            #         data_idx += 1

            with open(os.path.join(settings.phase_2_features, train_test, phase_1_file.path.split('/')[-1]), 'w+') as fout:
                writer = csv.writer(fout)
                header = data_points[0].keys()
                # print(header)
                # # header.sort()
                # # header.remove('time')
                # header.remove('primary_class_change')
                # header.remove('secondary_class_change')
                # header.remove('output')

                # # header.insert(0, 'time')
                # header.insert(1, 'primary_class_change')
                # header.insert(2, 'secondary_class_change')
                # header.append('output')

                writer.writerow(header)
                for point in data_points:
                    # point['time'] = (point['time'] - datetime(1970,1,1)).total_seconds()  # Convert to seconds since epoch
                    writer.writerow([point[key] for key in header])


def hyperparameter_tuning(params):
    # Clear features directory
    filelist = [f for f in os.listdir(settings.phase_2_features)]
    for f in filelist:
        os.remove(os.path.join(settings.phase_2_features, f))

    global best, count, best_params
    count += 1

    print('Iteration', count)
    print('Current Parameters:', params)

    pre_windows = params['pre_wind']
    post_windows = params['post_wind']
    learn_rate = params['learning_rate']
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']

    paths = []
    base_name = ''
    for subdir, dirs, files in os.walk(settings.phase_1_output):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                cur_base_name = '_'.join(cur_file.split('_')[:-3])  # Removes _YYYY-MM-DD_Hr_H.csv
                if base_name == '':
                    base_name = cur_base_name
                if cur_base_name != base_name:  # Changed to a new user
                    if os.path.isfile(settings.phase_2_raw_subset + os.sep + base_name + '.csv'):
                        generate_features(paths, base_name + '.csv', pre_windows, post_windows, settings.phase_2_raw_subset)
                    base_name = cur_base_name
                    paths = [cur_file]
                else:
                    paths.append(cur_file)
    f_measure = generate_models.generate_models(learn_rate, n_estimators, max_depth, False)
    print('')
    return {'loss': -f_measure, 'status': STATUS_OK}


def tune_hyperparameters():
    space = {
        'pre_wind': hp.choice('previous_windows', range(1, 20)),
        'post_wind': hp.choice('post_windows', range(1, 20)),
        'learning_rate': hp.uniform('learn_rate', 0.01, 2),
        'n_estimators': hp.choice('num_trees', range(10, 1000)),
        'max_depth': hp.choice('depth', range(1, 5))
    }

    trial = Trials()
    best = fmin(hyperparameter_tuning, space, algo=tpe.suggest, max_evals=50, trials=trial)
    # print('best:', best)

    best_params = space_eval(space, best)
    print('best:', best_params)

    return best_params


# Genreates features for testing the models. Does not use the same data as hyperparameter tuning
def generate_and_write_features(best_params):
    # Clear features directory
    filelist = [f for f in os.listdir(settings.phase_2_features)]
    for f in filelist:
        os.remove(os.path.join(settings.phase_2_features, f))

    pre_windows = best_params['pre_wind']
    post_windows = best_params['post_wind']
    paths = []
    base_name = ''
    for subdir, dirs, files in os.walk(settings.phase_1_output):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                cur_base_name = '_'.join(cur_file.split('_')[:-3])  # Removes _YYYY-MM-DD_Hr_H.csv
                if base_name == '':
                    base_name = cur_base_name
                if cur_base_name != base_name:  # Changed to a new user
                    if os.path.isfile(settings.phase_2_raw + os.sep + base_name + '.csv'):
                        generate_features(paths, base_name + '.csv', pre_windows, post_windows, settings.phase_2_raw)
                    base_name = cur_base_name
                    paths = [cur_file]
                else:
                    paths.append(cur_file)


def hapt_tiertwo():

    df_test = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(settings.phase_2_features, 'Test', '*.csv'))], ignore_index=True)
    df_train = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(settings.phase_2_features, 'Train', '*.csv'))], ignore_index=True)
    test_users = pd.read_csv(os.path.join(settings.phase_1_features, 'Test', 'subject_id_test.txt'), sep=" ", header=None)

    X_train = df_train.drop('Actual', axis=1).values
    Y_train = df_train['Actual']

    X_test = df_test.drop('Actual', axis=1).values
    Y_test = df_test['Actual']

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, Y_train.values.ravel())

    # Y_pred = clf.predict(X_test)
    Y_proba = clf.predict_proba(X_test)

    # # accuracy = accuracy_score(Y_test, Y_pred)
    # # print('Accuracy:', accuracy)
    # # print(Y_proba[0])
    # # header = ['Actual', 'Primary Prediction', 'Primary Probability', 'Secondary Probability', 'Secondary Prediction']

    user_indices = [test_users.loc[test_users[0] == index].index[0] for index in test_users[0].unique()]  # row_num of users

    for i, index in enumerate(user_indices[:-1]):
        # print(test_users[i])
        # print(type(test_users[i]))
        filename = 'User_' + str(test_users[0].unique()[i]) + '.csv'
        print('Creating:', filename)
        # print(Y_test[index:user_indices[i + 1]])
        user = pd.DataFrame()
        user['Actual'] = Y_test[index:user_indices[i + 1]]
        # user['Primary Prediction'] = Y_pred[index:user_indices[i + 1]][0]
        user['Primary Prediction'] = [clf.classes_.tolist()[np.where(probabilities == sorted(probabilities)[-1])[0][0]] for probabilities in Y_proba[index:user_indices[i + 1]]]
        user['Primary Probability'] = [max(probabilities) for probabilities in Y_proba[index:user_indices[i + 1]]]
        user['Secondary Prediction'] = [clf.classes_.tolist()[np.where(probabilities == sorted(probabilities)[-2])[0][0]] for probabilities in Y_proba[index:user_indices[i + 1]]]
        user['Secondary Probability'] = [sorted(probabilities)[-2] for probabilities in Y_proba[index:user_indices[i + 1]]]

        user.to_csv(os.path.join(settings.phase_2_output, filename), index=False)
