import settings
import os
import pandas as pd
import re


def merge_files():
    """Merges data with labels

    Merges activity data from pocket files with label data from label files

    """

    cols = ["Time",
            "AccelerometerX", "AccelerometerY", "AccelerometerZ",
            "GyroscopeX", "GyroscopeY", "GyroscopeZ",
            "GravityX", "GravityY", "GravityZ",
            "MagneticX", "MagneticY", "MagneticZ",
            "Activity"]

    for subdir, dirs, files in os.walk(settings.phase_1_raw):
        for pocket_file in sorted(files, key=settings.natural_keys):
            if pocket_file.endswith('.csv') and "pocket" in pocket_file:
                print "Merging:", pocket_file

                label_file = re.split("_", pocket_file)[0] + "_labels.csv"

                pocket_df = pd.read_csv(os.path.join(subdir, pocket_file))
                label_df = pd.read_csv(os.path.join(subdir, label_file))

                combined_df = pd.DataFrame()
                pocket_df = pocket_df.drop(['Activity'], axis=1)
                label_df_activities = label_df.drop(label_df.iloc[:, :-1], axis=1)

                label_row_num = 0

                for i, row in enumerate(pocket_df.itertuples(), 0):
                    pocket_time = getattr(row, u'Time')
                    while label_row_num < len(label_df.index) and label_df.iloc[label_row_num][u'Time'] < pocket_time:
                        label_row_num += 1

                    if label_row_num < len(label_df.index):
                        temp_pd = pd.concat([pocket_df.iloc[i], label_df_activities.iloc[label_row_num]])
                        combined_df = combined_df.append(temp_pd, ignore_index=True)
                        combined_df = combined_df[cols]

                combined_df.to_csv(os.path.join(settings.phase_1_processed, re.split("_", pocket_file)[0] + ".csv"), index=False)


def replace_other():
    """Replaces Other labels

    When data is not one of the predefined labels, other is selected and a new activity is typed in. As soon as the other button is clicked, the data is labeled as Other,
    followed by the label entered by the user. This replaces those Others with the label entered by the user.

    """

    for subdir, dirs, files in os.walk(settings.phase_1_processed):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                print "Processing: ", cur_file
                df = pd.read_csv(os.path.join(subdir, cur_file))
                reversed_df = df.iloc[::-1].reset_index(drop=True)

                prev_activity = ''
                for i, row in enumerate(reversed_df.itertuples(), 0):
                    activity = getattr(row, u'Activity')
                    if activity == 'Other' and prev_activity != '':
                        reversed_df.at[i, 'Activity'] = prev_activity
                    else:
                        prev_activity = activity
                df = reversed_df.iloc[::-1].reset_index(drop=True)

                df.to_csv(os.path.join(settings.phase_1_processed, cur_file), index=False)


def remove_nothing():
    """Removes Nothing labels

    Before a label is selected, the data is labeled as Nothing. This file removes the data labeled as Nothing

    """

    for subdir, dirs, files in os.walk(settings.phase_1_processed):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.csv'):
                print "Processing: ", cur_file
                df = pd.read_csv(os.path.join(subdir, cur_file))
                df[df.Activity != 'Nothing'].to_csv(os.path.join(settings.phase_1_processed, cur_file), index=False)


if __name__ == "__main__":
    settings.init()

    print "Merging files..."
    merge_files()

    print "\nRemoving 'Nothing' Labels..."
    remove_nothing()

    print "\nReplacing Other Labels..."
    replace_other()

    print "\nDone"
