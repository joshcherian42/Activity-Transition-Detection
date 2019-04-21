import os
import settings
import csv


def pamap_dat_to_csv():
    pamap_header = ['Time_s',
                    'gesture',
                    'hand_temp',
                    'hand_accx_16g', 'hand_accy_16g', 'hand_accz_16g',
                    'hand_accx_6g', 'hand_accy_6g', 'hand_accz_6g',
                    'hand_gyrox', 'hand_gyroy', 'hand_gyroz',
                    'hand_magx', 'hand_magy', 'hand_magz',
                    'chest_temp',
                    'chest_accx_16g', 'chest_accy_16g', 'chest_accz_16g',
                    'chest_accx_6g', 'chest_accy_6g', 'chest_accz_6g',
                    'chest_gyrox', 'chest_gyroy', 'chest_gyroz',
                    'chest_magx', 'chest_magy', 'chest_magz',
                    'ankle_temp',
                    'ankle_accx_16g', 'ankle_accy_16g', 'ankle_accz_16g',
                    'ankle_accx_6g', 'ankle_accy_6g', 'ankle_accz_6g',
                    'ankle_gyrox', 'ankle_gyroy', 'ankle_gyroz',
                    'ankle_magx', 'ankle_magy', 'ankle_magz']

    for subdir, dirs, files in os.walk(os.path.join(settings.phase_1_raw, "PAMAP2_Dataset")):
        for cur_file in sorted(files, key=settings.natural_keys):
            if cur_file.endswith('.dat'):
                with open(os.path.join(subdir, cur_file)) as dat_file, open(os.path.join(subdir, cur_file[:-4]) + '.csv', 'w') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(pamap_header)
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

                        del_indeces = [2, 16, 17, 18, 19, 33, 34, 35, 36, 50, 51, 52, 53]  # Delete orientation columns, these were not collected. Also heart rate.
                        for index in sorted(del_indeces, reverse=True):
                            del row[index]
                        if 'NaN' not in row:
                            csv_writer.writerow(row)


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

    for subdir, dirs, files in os.walk(os.path.join(settings.phase_1_raw, "OpportunityChallengeDatasetTaskC")): #also works for OpportunityChallengeDatasetTasksAB_2011_08_12 and OpportunityChallengeLabeled
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


if __name__ == "__main__":
    settings.init()
    # opportunity_dat_to_csv()
    pamap_dat_to_csv()
