import settings
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import math


# Gets the modified confusion matrix
# If remove_fp_in_margin is true then the redundant predicted positives within the margin_sec window are not counted
# If it is false, then they are counted as FP
def get_confusion_matrix(y_predict, y_actual, times, margin_sec, remove_fp_in_margin):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    accounted_for = [False for i in y_predict]

    def nearest_predict_true(predict, actual, actual_idx, times, margin_sec, remove_fp_in_margin):
        margin = int(math.floor(margin_sec / (settings.phase_1_window_size_millis / 1000))) # Number of margin windows
        delta = 0
        l_idx = max(0, i - delta)
        r_idx = min(i + delta, len(predict) - 1)
        return_val = -1
        while l_idx > actual_idx - margin - 1 and r_idx < actual_idx + margin + 1:
            if predict[l_idx] == 1 and not accounted_for[l_idx] and abs(times[i] - times[l_idx]) <= margin_sec:
                accounted_for[l_idx] = True
                if not remove_fp_in_margin:
                    return l_idx
                if return_val == -1:
                    return_val = l_idx
            if predict[r_idx] == 1 and not accounted_for[r_idx] and abs(times[i] - times[r_idx]) <= margin_sec:
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
            nearest_pred = nearest_predict_true(y_predict, y_actual, i, times, margin_sec, remove_fp_in_margin)
            if nearest_pred == -1 and not accounted_for[i]: # We did not predict this transition so add a FN
                fn += 1
                accounted_for[i] = True
            else:
                tp += 1

    for i in range(len(y_actual)): # Now count everything else which is either a FP or TN since all TP and FN have been counted
        if not accounted_for[i]:
            if y_predict[i] == 1:
                fp += 1
            else:
                tn += 1
            accounted_for[i] = True

    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def print_confusion_matrix(m):
    print('TP:', m['tp'])
    print('FP:', m['fp'])
    print('FN:', m['fn'])
    print('TN:', m['tn'])
    recall = float(m['tp']) / (m['tp'] + m['fn'])
    precision = float(m['tp']) / (m['tp'] + m['fp'])
    print('Recall:', recall)
    print('Precision:', precision)
    f_score = 2 * precision * recall / (precision + recall)
    print('f-score:', f_score)


# Plots the stats for the modified confusion matrix with a window size of +/- seconds param
def plot_stats(y_predict, y_actual, times, seconds, remove_fp_in_margin):
    recalls = []
    precisions = []
    fscores = []
    x = []
    for i in range(int(math.floor(seconds / (settings.phase_1_window_size_millis / 1000)))):
        accept_margin_seconds = i * (settings.phase_1_window_size_millis / 1000)
        m = get_confusion_matrix(y_predict, y_actual, times, accept_margin_seconds, remove_fp_in_margin)
        if (m['tp'] + m['fn']) == 0:
            recall = 0
        else:
            recall = float(m['tp']) / (m['tp'] + m['fn'])

        if (m['tp'] + m['fp']) == 0:
            precision = 0
        else:
            precision = float(m['tp']) / (m['tp'] + m['fp'])

        if (precision + recall) == 0:
            fscore = 0
        else:
            fscore = 2 * precision*recall / (precision + recall)

        recalls.append(recall)
        precisions.append(precision)
        fscores.append(fscore)
        x.append(i * settings.phase_1_window_size_millis / 1000)

    sns.set_style("ticks")
    plt.plot(x, recalls, label='Recall')
    plt.plot(x, precisions, label='Precision')
    plt.plot(x, fscores, label='F-Score')
    plt.axvline(x=300, color='red', linestyle=':')
    plt.xlabel('Margin to Count as TP (s)')
    plt.legend()
    plt.show()


# Generates the models and plots the modified confusion matrix graph with up to 10 minute margins if should_plot is True
# def generate_models(learn_rate, num_trees, depth, should_plot):
def generate_models(should_plot):
    df = pd.DataFrame()
    for subdir, dirs, files in os.walk(settings.phase_2_features):
        for cur_file in sorted(files, key=settings.natural_keys):
            temp_df = pd.read_csv(os.path.join(subdir, cur_file))
            df = pd.concat([df, temp_df], ignore_index=True)
    times = [int(t) for t in df['time'].values]
    df = df.drop('time', axis=1)
    # df.to_csv('all_features.csv')
    labels = df.drop('output', axis=1).keys().values
    x = df.drop('output', axis=1).values
    y = df['output']
    y_predict_all = [0 for i in range(len(y))]

    #clf = RandomForestClassifier(random_state=42)
    clf = GradientBoostingClassifier(random_state=42, learning_rate=learn_rate, n_estimators=num_trees, max_depth=depth)
    #clf = GradientBoostingClassifier(random_state=42)
    #clf = AdaBoostClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, random_state=42)

    selector = VarianceThreshold(threshold=0.05)
    selector.fit(x)
    x = selector.transform(x)

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)

        print("Predicted", sum(y_predict))
        print("Actual", sum(y_test))

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
                    break
                delta += 1
                l_idx = max(0, i - delta)
                r_idx = min(i + delta, len(y_predict_all) - 1)

    m = get_confusion_matrix(y_predict_all, y, times, 300, True)
    if (should_plot):
        plot_stats(y_predict_all, y, times, 600, False)
        plot_stats(y_predict_all, y, times, 600, True)
        print('Removed FP in window')
        print_confusion_matrix(m)
        print('Kept FP in window')
        m = get_confusion_matrix(y_predict_all, y, times, 300, False)
        print_confusion_matrix(m)

    recall = float(m['tp']) / (m['tp'] + m['fn'])
    precision = float(m['tp']) / (m['tp'] + m['fp'])
    f_score = 2 * precision * recall / (precision + recall)

    return f_score
