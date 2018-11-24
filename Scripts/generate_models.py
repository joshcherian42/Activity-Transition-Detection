import settings
import pickle
import numpy as np

from operator import itemgetter


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
