import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='cpp_input/MNIST_order3/pred/')
args = parser.parse_args()

if os.path.isdir(args.input):
    result = pd.DataFrame()
    columns = ['error_{}'.format(i) for i in range(64)]
    for inputfile in tqdm(os.listdir(args.input)):
        if 'log' not in inputfile:
            continue
        logfile = open(os.path.join(args.input, inputfile), 'r')
        lines = logfile.readlines()
        sample_id = inputfile.split('_')[1]

        step = 0
        error_list = []
        step_list = []
        for linenum, line in enumerate(lines):
            if 'hidden ciphertext' in line:
                hidden_ciphertext = lines[linenum+3].split(':')[1]
                hidden_ciphertext = np.fromstring(hidden_ciphertext, sep=',')[:-1]
                hidden_plaintext = np.fromstring(lines[linenum+7], sep=',')[:-1]
                error = hidden_plaintext - hidden_ciphertext
                step += 1
                error_list.append(error)
                step_list.append(step)

        error_list = np.array(error_list)
        if np.max(error_list) > pow(10, 100):
            continue
        error_df = pd.DataFrame(error_list, columns=columns)
        error_df['step'] = step_list
        error_df['sample_id'] = sample_id
        result = result.append(error_df, ignore_index=True)

    print(result.describe())
    result.to_csv(os.path.join(args.input, 'error.csv'), index=False)

else:
    logfile = open(args.input, 'r')
    lines = logfile.readlines()

    mse = []
    fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, figsize=(12, 10))
    count = 0
    for linenum, line in enumerate(lines):
        if 'hidden ciphertext' in line:
            print(line)
            hidden_ciphertext = lines[linenum+3].split(':')[1]
            hidden_ciphertext = np.fromstring(hidden_ciphertext, sep=',')
            hidden_plaintext = np.fromstring(lines[linenum+7], sep=',')
            error = hidden_plaintext - hidden_ciphertext
            sns.distplot(error, ax=axes[int(count / 6)][int(count % 6)])
            count += 1

            # print(hidden_ciphertext)
            # print(hidden_plaintext)
            mse.append(mean_squared_error(hidden_plaintext, hidden_ciphertext))

    print(mse)
    plt.tight_layout()
    plt.savefig(args.input+'.png')
