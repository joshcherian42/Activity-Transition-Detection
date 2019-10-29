import settings
import generate_test_data
import tierone
import tier_two
import sys
import os

count = 0
best = 0
best_params = {}


def call_process(start_message, cur_function, cur_args):
    getattr(*cur_function)(*cur_args)


if __name__ == "__main__":

    is_error = False
    available_datasets = ['PAMAP2', 'HAPT']

    if len(sys.argv) == 1:
        print('')
        print('Error: Dataset not specified')
        print('Usage: python main.py <dataset>')
        print('       Available Datasets:', ", ".join(available_datasets))
        print('')
    elif sys.argv[1] not in available_datasets:
        print('')
        print('Error: Dataset not found')
        print('       Available Datasets:', ", ".join(available_datasets))
        print('')
    else:
        settings.init(sys.argv[1])

        # Features are already provided for HAPT
        # tierone.hapt_tierone()

        print('Processing Training Data')
        tier_two.generate_features(os.path.join(settings.phase_1_output, 'Train'), settings.phase_2_raw, 10, 10, 'Train')
        print('')
        print('Processing Test Data')
        tier_two.generate_features(os.path.join(settings.phase_1_output, 'Test'), settings.phase_2_raw, 10, 10, 'Test')

        tier_two.hapt_tiertwo()
        # PAMAP2
        # params = {'phase_one_window_size': 5.12, 'phase_one_overlap_size': 1, 'classifier': {'max_features': 'auto', 'min_samples_split': 10, 'min_samples_leaf': 2, 'type': 'rf', 'n_estimators': 623}, 'phase_two_window_size': 8, 'scoring_function': 'log'}
        # algorithm = params['classifier']
        # tier_two_size = params['phase_two_window_size']
        # scoring_function = params['scoring_function']
        # phase_one = params['phase_one_window_size']
        # phase_one_overlap = params['phase_one_overlap_size']

        # # generate_models.train_algorithms(copy.deepcopy(params))
        # start_messages = [ # 'Preparing Data'
        #                   'Extracting Testing Features']
        #                   # 'Classifying Activities']
        #                   # 'Correcting Classifications']

        # functions = [# [clean_public_data.py, 'clean_data']
        #              [generate_test_data, 'main']]
        #              # [tierone, 'gen_output']]
        #              #[tierone, 'correct_output']]

        # function_args = [ #[]
        #                  [phase_one, phase_one_overlap]]
        #                  # [algorithm]]
        #                  #[False if settings.trial == 'Team Data' else True, phase_one, tier_two_size, scoring_function, trial, paths]]

        # for function_num, func in enumerate(functions):
        #     call_process(start_messages[function_num], func, function_args[function_num])

        # # tierone.cross_val_setup()

        print('Done!')
