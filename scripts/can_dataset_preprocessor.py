import csv
import os
from typing import TextIO

import pandas as pd

import time

previous_timestamp: float = 0



def pad_list(input_list) -> list[float]:
    while len(input_list) < 8:
        input_list.append(0)
    return input_list


def parse_can_message(can_message: dict[str, str], timestamp) -> tuple[pd.DataFrame, float]:
    try:
        arbitration_id = int(can_message['arbitration_id'], 16)
        datafield: bytes = bytes.fromhex(can_message['data_field'])  # noqa

        data_fields = [int(byte) for byte in datafield[:8]]

        data_fields = pad_list(data_fields)  # noqa

        current_timestamp = time.time()
        time_interval = 0.0 if timestamp == 0.0 else current_timestamp - timestamp

        message_data = [
            arbitration_id,
            data_fields[0],
            data_fields[1],
            data_fields[2],
            data_fields[3],
            data_fields[4],
            data_fields[5],
            data_fields[6],
            data_fields[7],
            time_interval,
            can_message['attack']
        ]
        message_df = pd.DataFrame([message_data])

        return message_df, current_timestamp
    except Exception as e:
        print(e)

def read_all_folders(csv_folders: [str]) -> None:
    print('start reading all folders...')

    new_csv_file = open('can_full_temp.csv', 'w')
    new_csv_file.write('arbitration_id,df1,df2,df3,df4,df5,df6,df7,df8,time_interval,attack\n') # write header

    for csv_folder in csv_folders:
        read_csvs_in_folder(csv_folder, new_csv_file)

    # close after writing operations is complete
    new_csv_file.close()
    print('process completed...')

def read_csvs_in_folder(csv_folder: str, new_csv: TextIO):
    for csv_file in os.listdir(csv_folder):
        print(csv_file)
        csv_file_path = os.path.join(csv_folder, csv_file)
        if csv_file.endswith('.csv'):
            read_csv_file(csv_file_path, new_csv)

def read_csv_file(csv_file: str, new_csv: TextIO) -> None:
    global previous_timestamp
    """
    1. read csv line by line and parse the record using the can_message parser
    2. then write to the new csv file

    :param csv_file: path where the csv file is stored
    :param new_csv: path where the new csv file is stored
    :return: None
    """

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        counter = 0

        for row in reader:
            # parse can message
            parsed_can_message, current_timestamp = parse_can_message(row, timestamp=previous_timestamp)
            previous_timestamp = current_timestamp

            # then write it to new csv file
            new_csv.write(parsed_can_message.to_csv(index=False, header=False))



def main():
    csv_training_folders = [
        '/home/vangerwuaj/Documents/can-train-and-test/set_01/train_01',
        '/home/vangerwuaj/Documents/can-train-and-test/set_02/train_01',
        '/home/vangerwuaj/Documents/can-train-and-test/set_03/train_01',
        '/home/vangerwuaj/Documents/can-train-and-test/set_04/train_01'
    ]

    test_training_folders = [
        '/home/vangerwuaj/Documents/test_folders/1',
        '/home/vangerwuaj/Documents/test_folders/2',
        '/home/vangerwuaj/Documents/test_folders/3',
        '/home/vangerwuaj/Documents/test_folders/4'
    ]

    read_all_folders(csv_training_folders)




if __name__ == '__main__':
    main()