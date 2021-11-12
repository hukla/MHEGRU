import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='cpp_input/MNIST_order3')
args = parser.parse_args()

error = pd.read_csv(os.path.join(args.input, 'pred/error.csv'), index_col=-1)
columns = ['error_{}'.format(i) for i in range(64)]
error_array = error.query('step==27')[columns]

mean = error_array.mean()
std = error_array.std()

fc_weight = np.loadtxt(os.path.join(
    args.input, 'weights/fc_W.csv'), delimiter=',')
fc_bias = np.loadtxt(os.path.join(
    args.input, 'weights/fc_b.csv'), delimiter=',')

ctx_scores = []
for repeat in tqdm(range(10)):
    target_list = []
    output_list = []
    mhegru_list = []
    for i in range(10000):
        input_path = os.path.join(args.input, 'input_{}'.format(i))
        last_hidden = np.loadtxt(os.path.join(
            input_path, 'hidden_27.csv'), delimiter=',')
        perturbed_hidden = last_hidden + np.random.normal(mean, std)

        target = int(np.loadtxt(os.path.join(
            input_path, 'target.csv'), delimiter=','))
        output = np.matmul(fc_weight, last_hidden) + fc_bias
        perturbed_output = np.matmul(fc_weight, perturbed_hidden) + fc_bias

        target_list.append(target)
        output_list.append(np.argmax(output))
        mhegru_list.append(np.argmax(perturbed_output))

    # result = pd.DataFrame({'target':target_list, 'plx_output':output_list, 'ctx_output':mhegru_list})
    ctx_scores.append(accuracy_score(target_list, mhegru_list))

print(np.mean(ctx_scores), np.std(ctx_scores))
pd.DataFrame({'ctx_scores': ctx_scores}).to_csv(
    os.path.join(args.input, 'ctx_score.csv'))
