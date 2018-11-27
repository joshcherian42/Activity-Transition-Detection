from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import settings
import generate_test_data
import tierone

count = 0
best = 0
best_params = {}


def call_process(start_message, cur_function, cur_args):
    getattr(*cur_function)(*cur_args)


if __name__ == "__main__":

    settings.init()

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

    # generate_models.train_algorithms(copy.deepcopy(params))
    # generate_test_data.process_file(13635, 'Amaryllis_9.8.2017_2017-09-06_Hr_10.csv')
    start_messages = ['Extracting Testing Features',
                      'Classifying Activities']
                      #'Correcting Classifications']

    functions = [[generate_test_data, 'main'],
                 [tierone, 'gen_output']]
                 #[tierone, 'correct_output']]

    function_args = [[phase_one],
                     [algorithm]]
                     #[False if settings.trial == 'Team Data' else True, phase_one, tier_two_size, scoring_function, trial, paths]]

    for function_num, func in enumerate(functions):
        call_process(start_messages[function_num], func, function_args[function_num])

    print 'Done!'
