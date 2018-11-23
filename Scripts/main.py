import os
import shutil
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import settings
import sys
import copy
import exceptions
import traceback
import generate_test_data
import tierone
import generate_models
import evaluation
import merge_gps_activities

count = 0
best = 0
best_params = {}


def delete_files(folder):
    for cur_file in os.listdir(folder):
        file_path = os.path.join(folder, cur_file)
        print 'Deleting...', file_path
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def validate_data(params):
    global best, count, best_params
    count += 1

    print params
    print ''

    phase_one = params['phase_one_window_size']
    # generate_test_data.main(phase_one, phase_one / 2, 'train')

    print ''
    print 'Generating Output'
    acc = generate_models.train_algorithms(copy.deepcopy(params))

    print params
    print ''

    if acc > best:
        print 'new best:', acc, 'using', params['classifier']['type']
        best = acc
        best_params = params

    print 'iters:', count, ', acc:', acc, 'using', params
    print 'cur best:', best, 'using', best_params
    print ''

    os.remove(settings.training_features_all)
    os.remove(settings.training_features_fastslow)
    delete_files(settings.test_features)
    delete_files(settings.models)
    delete_files(settings.output_src)
    delete_files(settings.output_corrected)
    return {'loss': -acc, 'status': STATUS_OK}


def count_files(trial):
    num_files = 0
    for subdir, dirs, files in os.walk(settings.test_src + "/" + trial):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                num_files += 1
    return num_files


def call_process(start_message, cur_function, cur_args):
    log_overall = open(settings.website_log_overall, "a")
    log_overall.write(start_message + '....')
    log_overall.close()

    getattr(*cur_function)(*cur_args)

    log_overall = open(settings.website_log_overall, "a")
    log_overall.write("Done" + "\n")
    if os.path.isfile(settings.website_log_local):
        os.remove(settings.website_log_local)
    log_overall.close()


if __name__ == "__main__":

    files = sys.argv[1].split(', ')
    paths = []
    trial = ''
    for sel_num, selection in enumerate(files):
        if selection != '':
            path = files[sel_num][files[sel_num].index("/Input-Data") + 12:]
            if "/" in path:
                trial = path[0:path.index("/")]
                paths.append(path[path.index(trial) + len(trial) + 1:])

    settings.init(trial)

    space = {
        'phase_one_window_size': hp.choice('window_size_one', range(5000, 120000)),
        'phase_two_window_size': hp.choice('window_size_two', range(3, 20)),
        'scoring_function': hp.choice('phase_two_scoring_function', ['normal', 'normal_inverted', 'squared', 'log', 'log_inverted']),
        'classifier': hp.choice('classifier_type', [
            {
                'type': 'naive_bayes',
            },
            {
                'type': 'svm',
                'C': hp.lognormal('svm_C', 0, 1),
                'kernel': hp.choice('kernel', ['linear', 'rbf']),
                'gamma': hp.uniform('gamma', 0, 20.0)
            },
            {
                'type': 'dtree',
                'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
                'max_depth': hp.choice('dtree_max_depth', [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
                'min_samples_split': hp.choice('dtree_min_samples_split', [2, 5, 10]),
            },
            {
                'type': 'rf',
                'n_estimators': hp.choice('rf_n_estimators', range(10, 2000)),
                'max_features': hp.choice('rf_max_features', ['auto', 'sqrt', 'log2']),
                'min_samples_split': hp.choice('rf_min_samples_split', [2, 5, 10]),
                'min_samples_leaf': hp.choice('rf_min_samples_leaf', [1, 2, 4]),
            },
            {
                'type': 'knn',
                'n_neighbors': hp.choice('knn_n_neighbors', range(1, 50))
            },
            {
                'type': 'mlp',
                'activation': hp.choice('mlp_activation', ['identity', 'logistic', 'tanh', 'relu']),
                'solver': hp.choice('mlp_solver', ['lbfgs', 'sgd', 'adam']),
                'alpha': hp.uniform('mlp_alpha', 0.00001, 0.01),
                'max_iter': hp.choice('max_iter', range(10, 1000))
            }
        ])
    }

    # {'phase_one_window_size': 102675, 'classifier': {'alpha': 0.006083425428079107, 'activation': 'logistic', 'max_iter': 194, 'type': 'mlp', 'solver': 'adam'}, 'phase_two_window_size': 16, 'scoring_function': 'normal'}
    # print validate_data({'phase_one_window_size': 83026, 'classifier': {'n_neighbors': 29, 'type': 'knn'}, 'phase_two_window_size': 16, 'scoring_function': 'log_inverted'})
    # print validate_data({'phase_one_window_size': 23961, 'classifier': {'alpha': 0.005045453705948127, 'activation': 'logistic', 'max_iter': 72, 'type': 'mlp', 'solver': 'adam'}, 'phase_two_window_size': 15, 'scoring_function': 'normal_inverted'})

    # This is for testing cross-validation
    # generate_models.train_algorithms({'phase_one_window_size': 13635, 'classifier': {'max_features': 'auto', 'min_samples_split': 10, 'min_samples_leaf': 2, 'type': 'rf', 'n_estimators': 623}, 'phase_two_window_size': 8, 'scoring_function': 'log'})

    # best = fmin(validate_data, space, algo=tpe.suggest, max_evals=5, trials=trial)
    # print 'best:', best

    # best_params = space_eval(space, best)
    # print tierone.main(best_params)

    params = {'phase_one_window_size': 13635, 'classifier': {'max_features': 'auto', 'min_samples_split': 10, 'min_samples_leaf': 2, 'type': 'rf', 'n_estimators': 623}, 'phase_two_window_size': 8, 'scoring_function': 'log'}
    algorithm = params['classifier']
    tier_two_size = params['phase_two_window_size']
    scoring_function = params['scoring_function']
    phase_one = params['phase_one_window_size']

    try:
        if os.path.isfile(settings.website_log_overall):
            os.remove(settings.website_log_overall)
        if os.path.isfile(settings.website_log_local):
            os.remove(settings.website_log_local)
        if os.path.isfile(settings.log_file):
            os.remove(settings.log_file)
        if os.path.isfile(settings.log_file_corrected):
            os.remove(settings.log_file_corrected)
        if os.path.isfile(settings.log_file_wear):
            os.remove(settings.log_file_wear)

        num_files = 0
        for path in paths:
            num_files += count_files(path)

        log_overall = open(settings.website_log_overall, "a")

        log_overall.write("Number of Files: " + str(num_files) + "\n")
        log_overall.write("Processing: ")

        files_to_be_processed = ''
        for path in paths:
            files_to_be_processed += trial + "/" + path + ", "
        else:
            log_overall.write(files_to_be_processed[:-2] + "\n")

        log_overall.write("\n")
        log_overall.close()

        # generate_models.train_algorithms(copy.deepcopy(params))

        start_messages = ['Extracting Testing Features',
                          'Classifying Activities',
                          'Correcting Classifications',
                          'Removing Non-Wear Time',
                          'Generating Activity Logs',
                          'Generating Graphs']

        functions = [[generate_test_data, 'main'],
                     [tierone, 'gen_output'],
                     [tierone, 'correct_output'],
                     [tierone, 'correct_wear'],
                     [evaluation, 'generate_user_logs'],
                     [merge_gps_activities, 'add_activities_noGPS']]

        function_args = [[phase_one, phase_one / 2, ' test', trial, paths],
                         [algorithm, trial, paths],
                         [False if settings.trial == 'Team Data' else True, phase_one, tier_two_size, scoring_function, trial, paths],
                         [False if settings.trial == 'Team Data' else True, phase_one, trial, paths],
                         [int(sys.argv[2]), trial, paths],
                         [trial, paths]]

        for function_num, func in enumerate(functions):
            call_process(start_messages[function_num], func, function_args[function_num])

        log_overall = open(settings.website_log_overall, "a")
        log_overall.write("\n")

        for path in paths:
            log_overall.write(trial + "/" + path + ",")
        else:
            if len(paths) > 1:
                log_overall.write(' have been processed!' + '\n')
            else:
                log_overall.write(' has been processed!' + '\n')

        paths_concatenated = ''

        for path in paths:
            paths_concatenated += path
        log_overall.write("The activity log can be found at User Logs/activity_log_" + trial + "_" + paths_concatenated.replace('/', '_') + ".csv" + "\n")
        log_overall.write("Graphs can be found in Graphs/" + trial)
        log_overall.close()

    except exceptions.Exception as e:

        with open(settings.website_log_local, 'a') as f:
            f.write("PYTHON ERRORS:\n")
            f.write(traceback.format_exc())
