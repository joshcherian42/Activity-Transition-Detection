import settings
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from operator import itemgetter

def get_confusion_matrix(y_predict, y_actual, margin, remove_fp_in_margin):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    accounted_for = [False for i in y_predict]

    def nearest_predict_true(predict, actual, actual_idx, margin, remove_fp_in_margin):
        delta = 0
        l_idx = max(0, i - delta)
        r_idx = min(i + delta, len(predict) - 1)
        return_val = -1
        while l_idx > actual_idx - margin - 1 and r_idx < actual_idx + margin + 1:
            if predict[l_idx] == 1 and not accounted_for[l_idx]:
                accounted_for[l_idx] = True
                if not remove_fp_in_margin:
                    return l_idx
                if return_val == -1:
                    return_val = l_idx
            if predict[r_idx] == 1 and not accounted_for[r_idx]:
                accounted_for[r_idx] = True
                if not remove_fp_in_margin:
                    return r_idx
                if return_val == -1:
                    return_val = r_idx

            delta += 1
            l_idx = max(0, i - delta)
            r_idx = min(i + delta, len(predict) - 1)
        return return_val

    for i in range(len(y_actual)):
        if y_actual[i] == 1: # Actual positive. We either got it (TP) or we didn't (FN)
            nearest_pred = nearest_predict_true(y_predict, y_actual, i, margin, remove_fp_in_margin)
            if nearest_pred == -1 and not accounted_for[i]: # We did not predict this transition so add a FN
                fn += 1
                accounted_for[i] = True
            #elif not accounted_for[nearest_pred]: # Prediction in the window and we haven't added it to values
            else:
                tp += 1
                #accounted_for[nearest_pred] = True

    for i in range(len(y_actual)): # Now count everything else which is either a FP or TN since all TP and FN have been counted
        if not accounted_for[i]:
            if y_predict[i] == 1:
                fp += 1
            else:
                tn += 1
            accounted_for[i] = True

    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def print_confusion_matrix(m):
    print 'TP:', m['tp']
    print 'FP:', m['fp']
    print 'FN:', m['fn']
    print 'TN:', m['tn']
    recall = float(m['tp']) / (m['tp'] + m['fn'])
    precision = float(m['tp']) / (m['tp'] + m['fp'])
    print 'Recall:', recall
    print 'Precision:', precision
    print 'f-score:', 2*precision*recall / (precision + recall)

def generate_models():
    df = pd.DataFrame()
    for subdir, dirs, files in os.walk(settings.phase_2_features):
        for cur_file in sorted(files, key=settings.natural_keys):
            temp_df = pd.read_csv(os.path.join(subdir, cur_file))
            df = pd.concat([df, temp_df], ignore_index=True)
    times = df['time'].values
    df = df.drop('time', axis=1)
    #df.to_csv('all_features.csv')
    labels = df.drop('output', axis=1).keys().values
    x = df.drop('output', axis=1).values
    y = df['output']
    y_predict_all = [0 for i in range(len(y))]

    #clf = RandomForestClassifier(random_state=42)
    clf = GradientBoostingClassifier(random_state=42, learning_rate=0.5)
    #clf = AdaBoostClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, random_state=42)

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)

        print "Predicted", sum(y_predict)
        print "Actual", sum(y_test)

        for i in range(len(y_predict)):
            y_predict_all[test_index[i]] = y_predict[i]

    # Avg distance to a positive point
    total_dist = 0
    dists = {}
    for i in range(len(y_predict_all)):
        if y_predict_all[i] == 1 and y[i] != 1:
            delta = 1  # +/- 1, 2, 3, etc.
            l_idx = max(0, i - delta)
            r_idx = min(i + delta, len(y_predict_all) - 1)
            while l_idx != 0 and r_idx != len(y_predict_all) - 1:
                if y[l_idx] == 1 or y[r_idx] == 1:
                    total_dist += delta
                    dists[times[i]] = delta
                    # print times[i], delta
                    break
                delta += 1
                l_idx = max(0, i - delta)
                r_idx = min(i + delta, len(y_predict_all) - 1)

    confusion_matrix = get_confusion_matrix(y_predict_all, y, 10, True)
    print_confusion_matrix(confusion_matrix)

    # with open('temp.csv', 'w+') as f:
    #     for i in range(len(y_predict_all)):
    #         f.write(str(y[i]) + ',' + str(y_predict_all[i]) + '\n')

    # for k in dists:
    #     print k, dists[k]
    print total_dist / float(sum(y_predict_all))

settings.init()
generate_models()

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

    noGPS_filename = settings.models + "/" + algorithm['type'] + '_noGPS.pkl'

    # if not os.path.isfile(fastslow_filename):
    #     train_algorithms(copy.copy(algorithm))

    # print 'Testing algorithms'

    noGPS_model_pkl = open(noGPS_filename, 'rb')
    noGPS_model = pickle.load(noGPS_model_pkl)
    noGPS_model_pkl.close()

    noGPS_predictions = noGPS_model.predict(np.array(df_test.iloc[:, 0:15]))
    probabilities = noGPS_model.predict_proba(np.array(df_test.iloc[:, 0:15]))

    primary_probabilities = list()
    primary_predictions = list()
    secondary_probabilities = list()
    secondary_predictions = list()

    for row, row_prob in enumerate(probabilities):
        pairs = [(i, row_prob[i]) for i in range(len(row_prob))]  # (index, prob) for each class for data point i
        pairs.sort(key=itemgetter(1), reverse=True)  # Sort list of tuples based on prob value

        primary_predictions.append(noGPS_model.classes_[pairs[0][0]])
        primary_probabilities.append(pairs[0][1])
        secondary_predictions.append(noGPS_model.classes_[pairs[1][0]])
        secondary_probabilities.append(pairs[1][1])

    predictions = list()
    accgps = list()
    noGPS_index = 0
    for i, row in df_test.iterrows():
        predictions.append(noGPS_predictions[noGPS_index])
        accgps.append('Just Acc')
        noGPS_index += 1

    predictions = np.append([predictions], [df_test['Start'].values], axis=0)
    predictions = np.append(predictions, [df_test['End'].values], axis=0)
    predictions = np.append(predictions, [primary_probabilities], axis=0)
    predictions = np.append(predictions, [primary_predictions], axis=0)
    predictions = np.append(predictions, [secondary_probabilities], axis=0)
    predictions = np.append(predictions, [secondary_predictions], axis=0)
    return predictions
