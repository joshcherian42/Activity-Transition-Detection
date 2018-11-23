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


def init():
    global sampling_rate = 30
    global include_GPS = False

    os.chdir(os.path.dirname(os.getcwd()))

    # Phase 1
    global phase_1_raw = os.getcwd() + "/Data/phase_1/raw"
    global phase_1_features = os.getcwd() + "/Data/phase_1/features"
    global phase_1_output = os.getcwd() + "/Data/phase_1/output"

    # Phase 2
    global phase_2_raw = os.getcwd() + "/Data/phase_2/raw"
    global phase_2_features = os.getcwd() + "/Data/phase_2/features"
    global phase_2_output = os.getcwd() + "/Data/phase_2/output"

    # Fix these fucking things
    global activity_var_names = ['Run', 'Walk', 'Drive', 'Bike', 'Sed']
    global activity_names = ['running', 'walking', 'driving', 'biking', 'sedentary']
    global confusion_matrix_labels = ['running', 'jogging', 'walking', 'strolling', 'driving', 'transit', 'biking', 'standing', 'sitting', 'putting on the table', 'sedentary']
    # Number in travel logs is index+1
    global travel_codes = ['Driving', 'Transit', 'Biking', 'Walking', 'Strolling', 'Putting On Table', 'Standing', 'Sitting', 'Jogging', 'Running']
    global activity_codes = ['Working', 'Leisure', 'Exercise', 'Eating', 'Shopping', 'Going Home', 'Tour',
                      'Sitting', 'Standing', 'Putting on Table', 'Walking', 'Running', 'Biking',
                      'Driving']
