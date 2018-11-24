import os
import settings
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
    noGPS_filename = settings.models + "/" + algorithm['type'] + '_noGPS.pkl'

    # if not os.path.isfile(fastslow_filename):
    #     train_algorithms(copy.copy(algorithm))

    # print 'Testing algorithms'

    noGPS_model_pkl = open(noGPS_filename, 'rb')
    noGPS_model = pickle.load(noGPS_model_pkl)
    noGPS_model_pkl.close()

    noGPS_predictions = noGPS_model.predict(np.array(df_test.iloc[:, 0:15]))

    predictions = list()
    accgps = list()
    noGPS_index = 0
    for i, row in df_test.iterrows():
        predictions.append(noGPS_predictions[noGPS_index])
        accgps.append('Just Acc')
        noGPS_index += 1

    predictions = np.append([predictions], [df_test['Start'].values], axis=0)
    predictions = np.append(predictions, [df_test['End'].values], axis=0)
    predictions = np.append(predictions, [accgps], axis=0)
    return predictions
