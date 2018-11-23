import os
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def init(trials):
    global sampling_rate

    global train_features_all
    global train_features_fastslow
    global training_features_all
    global training_features_fastslow
    global test_features
    global training_src
    global test_separated_src
    global train_separated_src
    global test_src
    global models
    global scripts
    global weartime

    global output_src
    global output_corrected
    global output_wear

    global log_file
    global log_file_corrected
    global log_file_wear
    global user_logs

    global graph_src
    global graphs
    global graphs_corrected
    global output_graph

    global gps_folder
    global gps_edited_folder

    global travel_logs
    global travel_codes
    global activity_codes
    global confusion_matrices
    global activity_var_names
    global activity_names
    global confusion_matrix_labels
    global trial
    global include_GPS

    global website_log_overall
    global website_log_local

    sampling_rate = 30
    include_GPS = True

    scripts = os.getcwd()
    os.chdir(os.path.dirname(os.getcwd()))
    if 'Demo' in trials:
        trial = 'Demo'
    else:
        trial = '1st Follow-Up'

    # trial = '1st Follow-up'
    train_features_all = os.getcwd() + "/Process Data/Training Data/Training Parts All/"
    train_features_fastslow = os.getcwd() + "/Process Data/Training Data/Training Parts Fastslow/"
    training_src = os.getcwd() + "/Process Data/Raw Training Data Corrected"
    training_features_all = os.getcwd() + "/Process Data/Training Data/training_data_allactivities.csv"
    training_features_fastslow = os.getcwd() + "/Process Data/Training Data/training_data_fastslow.csv"
    test_separated_src = os.getcwd() + "/Process Data/Separated Raw Test Data/" + trial
    train_separated_src = os.getcwd() + "/Process Data/Separated Raw Training Data"
    test_src = os.getcwd() + "/Process Data/Raw Test Data/Input-Data/" + trial
    test_features = os.getcwd() + "/Process Data/Test Data/" + trial
    models = os.getcwd() + "/Process Data/Models/"
    weartime = os.getcwd() + "/Process Data/Weartime/" + trial

    output_src = os.getcwd() + "/Process Data/Output/" + trial
    output_corrected = os.getcwd() + "/Process Data/Output_corrected"
    output_wear = os.getcwd() + "/Process Data/Output_wear"

    log_file = os.getcwd() + "/Process Data/Logs/output_log.csv"
    log_file_corrected = os.getcwd() + "/Process Data/Logs/output_corrected_log.csv"
    log_file_wear = os.getcwd() + "/Process Data/Logs/output_wear_log.csv"
    user_logs = os.getcwd() + "/Process Data/Raw Test Data/Output-Tables"

    graph_src = os.getcwd() + "/Process Data/Raw Training Data"  # /To Be Graphed/" + trial
    graphs = os.getcwd() + "/Process Data/Graphs"
    graphs_corrected = os.getcwd() + "/Process Data/Graphs Corrected"
    output_graph = os.getcwd() + "/Process Data/Raw Test Data/Output-Graphs/" + trial

    gps_folder = os.getcwd() + "/Process Data/GPS Data/" + trial
    gps_edited_folder = os.getcwd() + "/Process Data/GPS Data/" + trial + " Edited"

    travel_logs = os.getcwd() + "/Process Data/Travel Logs/" + trial
    confusion_matrices = os.getcwd() + "/Process Data/Confusion Matrices/"

    # Fix these fucking things
    activity_var_names = ['Run', 'Walk', 'Drive', 'Bike', 'Sed']
    activity_names = ['running', 'walking', 'driving', 'biking', 'sedentary']
    confusion_matrix_labels = ['running', 'jogging', 'walking', 'strolling', 'driving', 'transit', 'biking', 'standing', 'sitting', 'putting on the table', 'sedentary']
    # Number in travel logs is index+1
    travel_codes = ['Driving', 'Transit', 'Biking', 'Walking', 'Strolling', 'Putting On Table', 'Standing', 'Sitting', 'Jogging', 'Running']
    activity_codes = ['Working', 'Leisure', 'Exercise', 'Eating', 'Shopping', 'Going Home', 'Tour',
                      'Sitting', 'Standing', 'Putting on Table', 'Walking', 'Running', 'Biking',
                      'Driving']

    website_log_overall = os.getcwd() + "/Process Data/Separated Raw Test Data/log-overall.txt"

    website_log_local = os.getcwd() + "/Process Data/Separated Raw Test Data/log-files.txt"
