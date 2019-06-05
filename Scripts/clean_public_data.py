import os
import settings
import csv


def fog_dataset():
    fog_header = ['Time_ms',
                  'ank_accx', 'ank_accy', 'ank_accz',
                  'thigh_accx', 'thigh_accy', 'thigh_accz',
                  'trunk_accx', 'trunk_accy', 'trunk_accz',
                  'gesture']

    for subdir, dirs, files in os.walk(os.path.join(settings.phase_1_raw, "dataset_fog_release")):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.txt'):
                with open(os.path.join(subdir, cur_file)) as dat_file, open(os.path.join(subdir, cur_file[:-4]) + '.csv', 'w') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(fog_header)
                    for line in dat_file:
                        row = [field.strip() for field in line.split(' ')]

                        if row[10] == '0':
                            row[10] = 'Inactive'
                        elif row[10] == '1':
                            row[10] = 'Activity'
                        elif row[10] == '2':
                            row[10] = 'Freeze'

                        csv_writer.writerow(row)


def pamap_dat_to_csv():

    for subdir, dirs, files in os.walk(settings.phase_1_raw):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.dat'):
                print cur_file
                with open(os.path.join(subdir, cur_file)) as dat_file, open(os.path.join(settings.phase_1_processed, subdir.split('/')[-1], cur_file[:-4]) + '.csv', 'w') as csv_file:

                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(settings.raw_data_cols)

                    cur_heart_rate = 0.0
                    cur_rows = []
                    
                    for line in dat_file:
                        row = [field.strip() for field in line.split(' ')]
                        # For one user data was spliced together due to data collection being aborted. So check timestamps to see if are consecutive
                        # Gesture
                        if row[1] == '0':
                            row[1] = 'Inactive'
                        elif row[1] == '1':
                            row[1] = 'lying'
                        elif row[1] == '2':
                            row[1] = 'sitting'
                        elif row[1] == '3':
                            row[1] = 'standing'
                        elif row[1] == '4':
                            row[1] = 'walking'
                        elif row[1] == '5':
                            row[1] = 'running'
                        elif row[1] == '6':
                            row[1] = 'cycling'
                        elif row[1] == '7':
                            row[1] = 'Nordic walking'
                        elif row[1] == '9':
                            row[1] = 'watching TV'
                        elif row[1] == '10':
                            row[1] = 'computer work'
                        elif row[1] == '11':
                            row[1] = 'car driving'
                        elif row[1] == '12':
                            row[1] = 'ascending stairs'
                        elif row[1] == '13':
                            row[1] = 'descending stairs'
                        elif row[1] == '16':
                            row[1] = 'vacuum cleaning'
                        elif row[1] == '17':
                            row[1] = 'ironing'
                        elif row[1] == '18':
                            row[1] = 'folding laundry'
                        elif row[1] == '19':
                            row[1] = 'house cleaning'
                        elif row[1] == '20':
                            row[1] = 'playing soccer'
                        elif row[1] == '24':
                            row[1] = 'rope jumping'

                        data_indices = [0, 1, 2, 5, 6, 7, 22, 23, 24, 39, 40, 41]
                        row = [row[i] for i in data_indices]

                        if 'NaN' not in row:
                            if cur_heart_rate != 0:
                                for new_row in cur_rows:

                                    new_row[2] = round(cur_heart_rate + ((float(row[2]) - cur_heart_rate) / len(cur_rows)))  # linear interpolation

                                    if 'NaN' not in new_row:
                                        csv_writer.writerow(new_row)

                                cur_rows = []

                            cur_heart_rate = float(row[2])
                            cur_rows.append(row)
                        else:
                            cur_rows.append(row)


def opportunity_dat_to_csv():
    opportunity_header = ['Time_ms',
                          'back_accx', 'back_accy', 'back_accz',
                          'back_gyrox', 'back_gyroy', 'back_gyroz',
                          'back_magx', 'back_magy', 'back_magz',
                          'rua_accx', 'rua_accy', 'rua_accz',
                          'rua_gyrox', 'rua_gyroy', 'rua_gyroz',
                          'rua_magx', 'rua_magy', 'rua_magz',
                          'rla_accx', 'rla_accy', 'rla_accz',
                          'rla_gyrox', 'rla_gyroy', 'rla_gyroz',
                          'rla_magx', 'rla_magy', 'rla_magz',
                          'lua_accx', 'lua_accy', 'lua_accz',
                          'lua_gyrox', 'lua_gyroy', 'lua_gyroz',
                          'lua_magx', 'lua_magy', 'lua_magz',
                          'lla_accx', 'lla_accy', 'lla_accz',
                          'lla_gyrox', 'lla_gyroy', 'lla_gyroz',
                          'lla_magx', 'lla_magy', 'lla_magz',
                          'locomotion', 'gesture']

    for subdir, dirs, files in os.walk(os.path.join(settings.phase_1_raw, "OpportunityChallengeDatasetTaskC")):  # also works for OpportunityChallengeDatasetTasksAB_2011_08_12 and OpportunityChallengeLabeled
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.dat'):
                with open(os.path.join(subdir, cur_file)) as dat_file, open(os.path.join(subdir, cur_file[:-4]) + '.csv', 'w') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(opportunity_header)
                    for line in dat_file:
                        row = [field.strip() for field in line.split(' ')]

                        # Locomotion
                        if row[-2] == '101':
                            row[-2] = 'Stand'
                        elif row[-2] == '102':
                            row[-2] = 'Walk'
                        elif row[-2] == '104':
                            row[-2] = 'Sit'
                        elif row[-2] == '105':
                            row[-2] = 'Lie'
                        else:
                            row[-2] = 'Inactive'

                        # Gesture
                        if row[-1] == '506616':
                            row[-1] = 'Open_Door1'
                        elif row[-1] == '506617':
                            row[-1] = 'Open_Door2'
                        elif row[-1] == '504616':
                            row[-1] = 'Close_Door1'
                        elif row[-1] == '504617':
                            row[-1] = 'Close_Door2'
                        elif row[-1] == '506620':
                            row[-1] = 'Open_Fridge'
                        elif row[-1] == '504620':
                            row[-1] = 'Close_Fridge'
                        elif row[-1] == '506605':
                            row[-1] = 'Open_Dishwasher'
                        elif row[-1] == '504605':
                            row[-1] = 'Close_Dishwasher'
                        elif row[-1] == '506619':
                            row[-1] = 'Open_Drawer1'
                        elif row[-1] == '504619':
                            row[-1] = 'Close_Drawer1'
                        elif row[-1] == '506611':
                            row[-1] = 'Open_Drawer2'
                        elif row[-1] == '504611':
                            row[-1] = 'Close_Drawer2'
                        elif row[-1] == '506608':
                            row[-1] = 'Open_Drawer3'
                        elif row[-1] == '504608':
                            row[-1] = 'Close_Drawer3'
                        elif row[-1] == '508612':
                            row[-1] = 'Clean_Table'
                        elif row[-1] == '507621':
                            row[-1] = 'Drink_Cup'
                        elif row[-1] == '505606':
                            row[-1] = 'Toggle_Switch'
                        else:
                            row[-1] = 'Inactive'

                        csv_writer.writerow(row)


def clean_data():
    if settings.dataset == 'PAMAP2':
        pamap_dat_to_csv()
    elif settings.dataset == 'Opportunity':
        opportunity_dat_to_csv()
    elif settings.dataset == 'FOG':
        fog_dataset()
