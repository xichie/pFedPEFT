import sys
sys.path.append('./')
sys.path.append('../../')
import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from session import sliding_window, fixed_window
import shutil
import pickle
from sklearn.utils import shuffle
import json


# tqdm.pandas()
# pd.options.mode.chained_assignment = None  # default='warn'


# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def _count_anomaly(log_path):
    total_size = 0
    normal_size = 0
    with open(log_path, errors='ignore') as f:
        for line in f:
            total_size += 1
            if line.split('')[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


def sample_raw_data(data_file, output_file, sample_window_size, sample_step_size):
    """
    only sample supercomputer dataset such as bgl
    """
    sample_data = []
    labels = []
    idx = 0

    with open(data_file, 'r', errors='ignore') as f:
        for line in f:
            labels.append(line.split()[0] != '-')
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print(f"Process {round(idx / sample_window_size * 100, 4)} % raw data", end='\r')

    with open(output_file, "w") as f:
        f.writelines(sample_data)
    print("Sampling done")


def _file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def process_dataset(data_dir, output_dir, log_file, dataset_name, window_type, window_size, step_size, train_size,
                    random_sample=False, session_type="entry"):
    """
    creating log sequences by sliding window
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param window_size:
    :param step_size:
    :param train_size:
    :return:
    """
    ########
    # count anomaly
    ########
    # _count_anomaly(data_dir + log_file)

    ##################
    # Transformation #
    ##################
    print("Loading", f'{data_dir}{log_file}_structured.csv')
    df = pd.read_csv(f'{data_dir}{log_file}_structured.csv')
    
    # build log sequences
    if window_type == "sliding":
        # data preprocess
        if 'bgl' in dataset_name:
            df["datetime"] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
        else:
            df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')

        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)
        n_train = int(len(df) * train_size)
        if session_type == "entry":
            sliding = fixed_window
        else:
            sliding = sliding_window
            window_size = float(window_size) * 60
            step_size = float(step_size) * 60
        print(random_sample)
        if random_sample:
            print("???")
            window_df = sliding(df[["timestamp", "Label", "EventId", "deltaT", "EventTemplate", "Content"]],
                                       para={"window_size": window_size,
                                             "step_size": step_size})
            window_df = shuffle(window_df).reset_index(drop=True)
            n_train = int(len(window_df) * train_size)
            train_window = window_df.iloc[:n_train, :].to_dict("records")
            test_window = window_df.iloc[n_train:, :].to_dict("records")
        else:
            train_window = sliding(
                df[["timestamp", "Label", "EventId", "deltaT", "EventTemplate", "Content"]].iloc[:n_train, :],
                para={"window_size": window_size,
                      "step_size": step_size}).to_dict("records")
            test_window = sliding(
                df[["timestamp", "Label", "EventId", "deltaT", "EventTemplate", "Content"]].iloc[n_train:, :].reset_index(
                    drop=True),
                para={"window_size": window_size, "step_size": step_size}).to_dict("records")

    elif window_type == "session":
        # only for hdfs
        if dataset_name == "hdfs":
          pass
    else:
        raise NotImplementedError(f"{window_type} is not implemented")

    if not os.path.exists(output_dir):
        print(f"creating {output_dir}")
        os.mkdir(output_dir)
    # save pickle file
    # print(train_window_df.head())
    # print(test_window_df.head())
    # train_window = train_window_df.to_dict("records")
    # test_window = test_window_df.to_dict("records")
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)


def _file_generator2(filename, df):
    if "train" in filename:
        is_duplicate = {}
        with open(filename, 'w') as f:
            for _, seq in enumerate(df):
                seq = " ".join(seq)
                if seq not in is_duplicate.keys():
                    f.write(seq + "\n")
                    is_duplicate[seq] = 1
    else:
        with open(filename, 'w') as f:
            for _, seq in enumerate(df):
                seq = " ".join(seq)
                f.write(seq + "\n")

if __name__ == '__main__':
    process_dataset(data_dir="data/logad/", output_dir="data/logad/", log_file="BGL.log", dataset_name="bgl",
                    window_type="sliding", window_size=50, step_size=50, train_size=0.8, random_sample=False,
                    session_type="entry")