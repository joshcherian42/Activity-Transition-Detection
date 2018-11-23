import os
import settings
import tiertwo
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import copy


def train_algorithms(params):
    """Trains algorithms

    Trains algorithms and calls pickle_write to save them to disk

    """

    print 'Training algorithms'
    algorithm = params['classifier']
    tier_two_size = params['phase_two_window_size']
    scoring_function = params['scoring_function']

    df_all = pd.read_csv(settings.training_features_all, sep=',')
    df_all = df_all[df_all.Activity != '-']
    training_users = set(df_all['Filename'].values)

    # if settings.include_GPS:
    df_slow = df_all[~df_all.Activity.isin(['driving', 'transit', 'biking'])]
    df_fast = df_all[df_all.Activity.isin(['driving', 'transit', 'biking'])]

    df_fastslow = pd.read_csv(settings.training_features_fastslow, sep=',')
    df_fastslow = df_fastslow[df_fastslow.Activity != '-']

    X_fastslow = np.array(df_fastslow['GPS Speed']).reshape(-1, 1)
    Y_fastslow = np.array(df_fastslow['Activity'])

    X_fast = np.array(df_fast.iloc[:, 0:18])
    Y_fast = np.array(df_fast['Activity'])

    X_slow = np.array(df_slow.iloc[:, 0:18])
    Y_slow = np.array(df_slow['Activity'])
    #else:
    X_noGPS = np.array(df_all.iloc[:, 0:15])
    Y_noGPS = np.array(df_all['Activity'])

    alg_type = algorithm['type']
    del algorithm['type']

    if alg_type == 'naive_bayes':

        #if settings.include_GPS:
        fastslow, fast, slow = (GaussianNB(priors=None) for i in range(3))
        #else:
        noGPS = GaussianNB(priors=None)

    elif alg_type == 'svm':

        #if settings.include_GPS:
        fastslow, fast, slow = (svm.SVC(C=algorithm['C'],
                                            kernel=algorithm['kernel'],
                                            gamma=algorithm['gamma'])
                                    for i in range(3))
        #else:
        noGPS = svm.SVC(C=algorithm['C'],
                            kernel=algorithm['kernel'],
                            gamma=algorithm['gamma'])

    elif alg_type == 'dtree':

        #if settings.include_GPS:
        fastslow, fast, slow = (DecisionTreeClassifier(criterion=algorithm['criterion'],
                                                           max_depth=algorithm['max_depth'],
                                                           min_samples_split=algorithm['min_samples_split'])
                                    for i in range(3))
        #else:
        noGPS = DecisionTreeClassifier(criterion=algorithm['criterion'],
                                           max_depth=algorithm['max_depth'],
                                           min_samples_split=algorithm['min_samples_split'])

    elif alg_type == 'rf':
        #if settings.include_GPS:
        fastslow, fast, slow = (RandomForestClassifier(n_estimators=algorithm['n_estimators'],
                                                           max_features=algorithm['max_features'],
                                                           min_samples_split=algorithm['min_samples_split'],
                                                           min_samples_leaf=algorithm['min_samples_leaf'])
                                    for i in range(3))
        #else:
        noGPS = RandomForestClassifier(n_estimators=algorithm['n_estimators'],
                                           max_features=algorithm['max_features'],
                                           min_samples_split=algorithm['min_samples_split'],
                                           min_samples_leaf=algorithm['min_samples_leaf'])

    elif alg_type == 'knn':
        #if settings.include_GPS:
        fastslow, fast, slow = (neighbors.KNeighborsClassifier(n_neighbors=algorithm['n_neighbors']) for i in range(3))
        #else:
        noGPS = neighbors.KNeighborsClassifier(n_neighbors=algorithm['n_neighbors'])

    elif alg_type == 'mlp':
        #if settings.include_GPS:
        fastslow, fast, slow = (MLPClassifier(activation=algorithm['activation'],
                                                  solver=algorithm['solver'],
                                                  alpha=algorithm['alpha'],
                                                  max_iter=algorithm['max_iter'])
                                    for i in range(3))
        #else:
        noGPS = MLPClassifier(activation=algorithm['activation'],
                                  solver=algorithm['solver'],
                                  alpha=algorithm['alpha'],
                                  max_iter=algorithm['max_iter'])

    print 'Fitting Algorithms'
    print ''

    # if settings.include_GPS:

    print 'Fastslow...'
    
    # avg_score = looc(fastslow, fast, slow, df_fastslow, df_all, 'fastslow', tier_two_size, scoring_function)
    fastslow.fit(X_fastslow, Y_fastslow)
    pickle_write(settings.models + alg_type + '_fastslow.pkl', fastslow)

    print 'Fast...'
    #score_fast = looc(training_users, fast, df_fast, 'fast', tier_two_size, scoring_function)
    #score_fast = cross_val_score(fast, X_fast, Y_fast, cv=shuffle, scoring='accuracy')
    fast.fit(X_fast, Y_fast)
    pickle_write(settings.models + alg_type + '_fast.pkl', fast)

    print 'Slow...'
    #score_slow = looc(training_users, slow, df_slow, 'slow', tier_two_size, scoring_function)
    #score_slow = cross_val_score(slow, X_slow, Y_slow, cv=shuffle, scoring='accuracy')
    slow.fit(X_slow, Y_slow)
    pickle_write(settings.models + alg_type + '_slow.pkl', slow)

    # avg_score = (score_fastslow + score_fast + score_slow) / 3.0
    # print 'Fastslow:', score_fastslow
    # print 'Fast:', score_fast
    # print 'Slow:', score_slow
    # print 'Avg Accuracy:', avg_score
    # else:

    print 'noGPS...'
    # avg_score = looc(training_users, noGPS, df_all, 'noGPS')
    # avg_score = cross_val_score(noGPS, X_noGPS, Y_noGPS, cv=shuffle, scoring='accuracy').mean()
    noGPS.fit(X_noGPS, Y_noGPS)
    pickle_write(settings.models + alg_type + '_noGPS.pkl', noGPS)

    print 'Done Fitting'
    print ''
    #return avg_score


def looc(fastslow, fast, slow, df_fastslow, df_all, df_type, tier_two_size, scoring_function):
    accuracies = []

    training_users = set([x.split('_')[0] for x in df_all['Filename'].values])

    df_slow = df_all[~df_all.Activity.isin(['driving', 'transit', 'biking'])]
    df_fast = df_all[df_all.Activity.isin(['driving', 'transit', 'biking'])]

    for test_user in training_users:

        df_train_fastslow = df_fastslow[~df_fastslow.Filename.str.contains(test_user)]
        df_train_fast = df_fast[~df_fast.Filename.str.contains(test_user)]
        df_train_slow = df_slow[~df_slow.Filename.str.contains(test_user)]

        df_test = df_all[df_all.Filename.str.contains(test_user)]

        print test_user
        print ''
        if df_type == 'fastslow':

            X_fastslow = np.array(df_train_fastslow['GPS Speed']).reshape(-1, 1)
            Y_fastslow = np.array(df_train_fastslow['Activity'])

            X_fast = np.array(df_train_fast.iloc[:, 0:18])
            Y_fast = np.array(df_train_fast['Activity'])

            X_slow = np.array(df_train_slow.iloc[:, 0:18])
            Y_slow = np.array(df_train_slow['Activity'])

        elif df_type == 'noGPS':
            X_train = np.array(df_all.iloc[:, 0:15])

        fastslow.fit(X_fastslow, Y_fastslow)
        fast.fit(X_fast, Y_fast)
        slow.fit(X_slow, Y_slow)

        X_test = np.array(df_test['GPS Speed']).reshape(-1, 1)
        Y_test = np.array(df_test['Activity'])
        predictions = fastslow.predict(X_test)

        slow_preds = list()
        fast_preds = list()
        for pred_num, fastslow_pred in enumerate(predictions):
            if fastslow_pred == 'Slow':
                slow_preds.append(df_test.iloc[pred_num].tolist()[0:18])
            elif fastslow_pred == 'Fast':
                fast_preds.append(df_test.iloc[pred_num].tolist()[0:18])

        if len(fast_preds) > 0:
            fast_predictions = fast.predict(np.array(fast_preds))
        if len(slow_preds) > 0:
            slow_predictions = slow.predict(np.array(slow_preds))

        fast_index = 0
        slow_index = 0
        for pred_num, fastslow_pred in enumerate(predictions):
            if fastslow_pred == 'Slow':
                predictions[pred_num] = slow_predictions[slow_index]
                slow_index += 1
            elif fastslow_pred == 'Fast':
                predictions[pred_num] = fast_predictions[fast_index]
                fast_index += 1

        tier_two = tiertwo.weighted_moving_average(tier_two_size, scoring_function)

        corrected_output = ['act' for x in xrange(len(predictions))]
        for index, pred_activity in enumerate(predictions):
            corrected_activities = tier_two.new_window(pred_activity)
            corrected_output[index] = pred_activity

            if corrected_activities != '':
                corrected_output[index - tier_two_size / 2 - 1] = corrected_activities

        # get the accuracy
        accuracies.append(accuracy_score(Y_test, corrected_output))
    print accuracies
    return sum(accuracies)/len(accuracies)


def pickle_write(filename, model):
    """Save model to disk

    Saves model to disk

    Args:
        filename (dataframe): test data to evaluate
        model: training model to save

    """
    model_pkl = open(filename, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def test_algorithm(algorithm, df_test):
    """Classifies test data

    Trains algorithm if it has not already been trained and evaluates test data with that algorithm

    Args:
        algorithm (str): what algorithm to run WRONG
            rf: Random Forest
            dt: Decision Tree
            svm: Support Vector Machine
            knn: k-Nearest Neighbor
            nb: Naive Bayes
            mlp: Multi-Layer Perceptron
        df_test (dataframe): test data to evaluate

    Returns:
        list: predictions for every instance in the test data
    """
    #NEED TO FIX HERE
    #if settings.include_GPS:
    fastslow_filename = settings.models + algorithm['type'] + '_fastslow.pkl'
    noGPS_filename = settings.models + algorithm['type'] + '_noGPS.pkl'
    fast_filename = settings.models + algorithm['type'] + '_fast.pkl'
    slow_filename = settings.models + algorithm['type'] + '_slow.pkl'

    # if not os.path.isfile(fastslow_filename):
    #     train_algorithms(copy.copy(algorithm))

    # print 'Testing algorithms'
    fastslow_model_pkl = open(fastslow_filename, 'rb')
    fastslow_model = pickle.load(fastslow_model_pkl)
    fastslow_model_pkl.close()

    noGPS_model_pkl = open(noGPS_filename, 'rb')
    noGPS_model = pickle.load(noGPS_model_pkl)
    noGPS_model_pkl.close()

    fast_model_pkl = open(fast_filename, 'rb')
    fast_model = pickle.load(fast_model_pkl)
    fast_model_pkl.close()

    slow_model_pkl = open(slow_filename, 'rb')
    slow_model = pickle.load(slow_model_pkl)
    slow_model_pkl.close()

    df_fastslow = df_test[df_test['GPS Speed'] != -1]
    fastslow_predictions = list()

    if len(np.array(df_fastslow['GPS Speed']).reshape(-1, 1)) > 0:
        fastslow_predictions = fastslow_model.predict(np.array(df_fastslow['GPS Speed']).reshape(-1, 1))

    df_noGPS = df_test[df_test['GPS Speed'] == -1]
    if df_noGPS.empty:
        predictions = fastslow_predictions
        accgps = ['Acc + GPS'] * len(predictions)

    else:
        noGPS_predictions = noGPS_model.predict(np.array(df_noGPS.iloc[:, 0:15]))

        predictions = list()
        accgps = list()
        fastslow_index = 0
        noGPS_index = 0
        for i, row in df_test.iterrows():
            if i in df_fastslow.index:
                predictions.append(fastslow_predictions[fastslow_index])
                accgps.append('Acc + GPS')
                fastslow_index += 1
            elif i in df_noGPS.index:
                predictions.append(noGPS_predictions[noGPS_index])
                accgps.append('Just Acc')
                noGPS_index += 1

    slow_preds = list()
    fast_preds = list()
    for pred_num, fastslow_pred in enumerate(predictions):
        if fastslow_pred == 'Slow':
            slow_preds.append(df_test.iloc[pred_num].tolist()[0:18])
        elif fastslow_pred == 'Fast':
            fast_preds.append(df_test.iloc[pred_num].tolist()[0:18])

    if len(fast_preds) > 0:
        fast_predictions = fast_model.predict(np.array(fast_preds))
    if len(slow_preds) > 0:
        slow_predictions = slow_model.predict(np.array(slow_preds))

    fast_index = 0
    slow_index = 0
    for pred_num, fastslow_pred in enumerate(predictions):
        if fastslow_pred == 'Slow':
            predictions[pred_num] = slow_predictions[slow_index]
            slow_index += 1
        elif fastslow_pred == 'Fast':
            predictions[pred_num] = fast_predictions[fast_index]
            fast_index += 1
    predictions = np.append([predictions], [df_test['Start'].values], axis=0)
    predictions = np.append(predictions, [df_test['End'].values], axis=0)
    predictions = np.append(predictions, [accgps], axis=0)
    return predictions
