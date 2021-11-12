import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='cpp_input/profile_kernel')
args = parser.parse_args()

if os.path.isdir(args.input):
    profile_df = pd.DataFrame()
    for profile_log in os.listdir(args.input):
        if 'num_threads' not in profile_log:
            continue
        filepath = os.path.join(args.input, profile_log)
        profile = open(filepath, 'r').readlines()
        num_threads = int(profile_log.split('.')[0].split('_')[2])

        operation_list = []
        time_list = []
        occurence = []
        for line in profile:
            if 'time' in line:
                line_list = line.split(' ')
                operation = line_list[0]
                if 'Encrypt' in operation:
                    time = float(line_list[4])
                else:
                    time = float(line_list[3])

                if operation in operation_list:
                    i = operation_list.index(operation)
                    time_list[i] += time
                    occurence[i] += 1
                else:
                    operation_list.append(operation)
                    time_list.append(time)
                    occurence.append(1)
        time_list = np.array(time_list) / np.array(occurence)
        profile_df = profile_df.append(pd.DataFrame(
            {'operation': operation_list, 'execution time': time_list, 'num_threads': num_threads}), ignore_index=True)

    profile_df.to_csv('profile_kernel.csv', index=False)
