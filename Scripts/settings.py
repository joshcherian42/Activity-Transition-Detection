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


def init(data):

    global dataset
    dataset = data

    global sampling_rate
    sampling_rate = 30
    global include_GPS
    include_GPS = False
    global scripts
    scripts = os.getcwd()

    os.chdir(os.path.dirname(os.getcwd()))

    # Phase 1
    global phase_1_raw
    phase_1_raw = os.path.join(os.getcwd(), "Data", "phase_1", "raw", data)
    global phase_1_processed
    phase_1_processed = os.path.join(os.getcwd(), "Data", "phase_1", "processed", data)
    global phase_1_features
    phase_1_features = os.path.join(os.getcwd(), "Data", "phase_1", "features", data)
    global phase_1_cross_val
    phase_1_cross_val = os.path.join(os.getcwd(), "Data", "phase_1", "cross-validation", data)
    global phase_1_output
    phase_1_output = os.path.join(os.getcwd(), "Data", "phase_1", "output", data)
    global models
    models = os.path.join(os.getcwd(), "Data", "models", data)
    global phase_1_window_size_millis
    phase_1_window_size_millis = 13635

    # Phase 2
    global phase_2_raw
    phase_2_raw = os.path.join(os.getcwd(), "Data", "phase_2", "raw")
    global phase_2_raw_subset
    phase_2_raw_subset = os.path.join(os.getcwd(), "Data", "phase_2", "raw_subset")
    global phase_2_features
    phase_2_features = os.path.join(os.getcwd(), "Data", "phase_2", "features")
    global phase_2_output
    phase_2_output = os.path.join(os.getcwd(), "Data", "phase_2", "output")
    global phase_2_remove_window_size_secs
    phase_2_remove_window_size_secs = 900

    global activity_var_names
    activity_var_names = ['Run', 'Walk', 'Drive', 'Bike', 'Sed']
    global activity_names
    activity_names = ['running', 'walking', 'driving', 'biking', 'sedentary']
    global confusion_matrix_labels
    confusion_matrix_labels = ['running', 'jogging', 'walking', 'strolling', 'driving', 'transit', 'biking', 'standing', 'sitting', 'putting on the table', 'sedentary']
    # Number in travel logs is index+1
    global travel_codes
    travel_codes = ['Driving', 'Transit', 'Biking', 'Walking', 'Strolling', 'Putting On Table', 'Standing', 'Sitting', 'Jogging', 'Running']
    global activity_codes
    activity_codes = ['Working', 'Leisure', 'Exercise', 'Eating', 'Shopping', 'Going Home', 'Tour',
                      'Sitting', 'Standing', 'Putting on Table', 'Walking', 'Running', 'Biking',
                      'Driving']

    global features_header
    global raw_data_cols
    if dataset == 'PAMAP2':
        features_header = [  # Hand
                             'Mean_hand_x', 'Mean_hand_y', 'Mean_hand_z',
                             'Mean_hand_euclidean',
                             'Median_hand_x', 'Median_hand_y', 'Median_hand_z',
                             'Median_hand_euclidean',
                             'Stdev_hand_x', 'Stdev_hand_y', 'Stdev_hand_z',
                             'Stdev_hand_euclidean',
                             'Peak_freq_hand_x', 'Peak_freq_hand_y', 'Peak_freq_hand_z',
                             'Peak_freq_hand_euclidean',
                             'Energy_hand_x', 'Energy_hand_y', 'Energy_hand_z',
                             'Energy_hand_euclidean',
                             'Feature_abs_integral_hand_x', 'Feature_abs_integral_hand_y', 'Feature_abs_integral_hand_z',
                             'Feature_abs_integral_hand_euclidean',
                             'Correlation_hand_x', 'Correlation_hand_y', 'Correlation_hand_z',
                             'Power_ratio_hand_x', 'Power_ratio_hand_y', 'Power_ratio_hand_z',
                             'Power_ratio_hand_euclidean',
                             'Peak_psd_hand_x', 'Peak_psd_hand_y', 'Peak_psd_hand_z',
                             'Peak_psd_hand_euclidean',
                             'Entropy_hand_x', 'Entropy_hand_y', 'Entropy_hand_z',
                             'Entropy_hand_euclidean',

                             # Chest
                             'Mean_chest_x', 'Mean_chest_y', 'Mean_chest_z',
                             'Mean_chest_euclidean',
                             'Median_chest_x', 'Median_chest_y', 'Median_chest_z',
                             'Median_chest_euclidean',
                             'Stdev_chest_x', 'Stdev_chest_y', 'Stdev_chest_z',
                             'Stdev_chest_euclidean',
                             'Peak_freq_chest_x', 'Peak_freq_chest_y', 'Peak_freq_chest_z',
                             'Peak_freq_chest_euclidean',
                             'Energy_chest_x', 'Energy_chest_y', 'Energy_chest_z',
                             'Energy_chest_euclidean',
                             'Feature_abs_integral_chest_x', 'Feature_abs_integral_chest_y', 'Feature_abs_integral_chest_z',
                             'Feature_abs_integral_chest_euclidean',
                             'Correlation_chest_x', 'Correlation_chest_y', 'Correlation_chest_z',
                             'Power_ratio_chest_x', 'Power_ratio_chest_y', 'Power_ratio_chest_z',
                             'Power_ratio_chest_euclidean',
                             'Peak_psd_chest_x', 'Peak_psd_chest_y', 'Peak_psd_chest_z',
                             'Peak_psd_chest_euclidean',
                             'Entropy_chest_x', 'Entropy_chest_y', 'Entropy_chest_z',
                             'Entropy_chest_euclidean',

                             # Ankle
                             'Mean_ankle_x', 'Mean_ankle_y', 'Mean_ankle_z',
                             'Mean_ankle_euclidean',
                             'Median_ankle_x', 'Median_ankle_y', 'Median_ankle_z',
                             'Median_ankle_euclidean',
                             'Stdev_ankle_x', 'Stdev_ankle_y', 'Stdev_ankle_z',
                             'Stdev_ankle_euclidean',
                             'Peak_freq_ankle_x', 'Peak_freq_ankle_y', 'Peak_freq_ankle_z',
                             'Peak_freq_ankle_euclidean',
                             'Energy_ankle_x', 'Energy_ankle_y', 'Energy_ankle_z',
                             'Energy_ankle_euclidean',
                             'Feature_abs_integral_ankle_x', 'Feature_abs_integral_ankle_y', 'Feature_abs_integral_ankle_z',
                             'Feature_abs_integral_ankle_euclidean',
                             'Correlation_ankle_x', 'Correlation_ankle_y', 'Correlation_ankle_z',
                             'Power_ratio_ankle_x', 'Power_ratio_ankle_y', 'Power_ratio_ankle_z',
                             'Power_ratio_ankle_euclidean',
                             'Peak_psd_ankle_x', 'Peak_psd_ankle_y', 'Peak_psd_ankle_z',
                             'Peak_psd_ankle_euclidean',
                             'Entropy_ankle_x', 'Entropy_ankle_y', 'Entropy_ankle_z',
                             'Entropy_ankle_euclidean',

                             # Pairwise
                             'Mean_pairwise_hand_chest', 'Stdev_pairwise_hand_chest', 'Feature_abs_integral_hand_chest', 'Energy_hand_chest',
                             'Mean_pairwise_hand_ankle', 'Stdev_pairwise_hand_ankle', 'Feature_abs_integral_hand_ankle', 'Energy_hand_ankle',
                             'Mean_pairwise_chest_ankle', 'Stdev_pairwise_chest_ankle', 'Feature_abs_integral_chest_ankle', 'Energy_chest_ankle',
                             'Mean_pairwise_hand_chest_ankle', 'Stdev_pairwise_hand_chest_ankle', 'Feature_abs_integral_hand_chest_ankle', 'Energy_hand_chest_ankle',

                             # Heart Rate
                             'Normalized_mean', 'Gradient', 'Activity']

        raw_data_cols = ["Time_s",
                         "gesture",
                         "heart_rate",
                         "hand_accx_16g", "hand_accy_16g", "hand_accz_16g",
                         "chest_accx_16g", "chest_accy_16g", "chest_accz_16g",
                         "ankle_accx_16g", "ankle_accy_16g", "ankle_accz_16g"]
