import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from customFunctions import ApproxSigmoid, ApproxTanh

SEED = 2021
torch.random.manual_seed(SEED)
np.random.seed(SEED)

log_interval = 50
epochs = 30
bptt = 6
best_loss = 100
lr = .01
activation = 'heaan'
if_train = False 
if_export = False
debug = False
save = 'cpp_input/addingProblem_6'

##################### APPROX ACTIVATIONS ########################
if activation == 'heaan':
    sigmoid = ApproxSigmoid.apply
    tanh = ApproxTanh.apply
elif activation == 'square':
    class Square(torch.autograd.Function):
        @ staticmethod
        def forward(ctx, input):
            y = torch.pow(input, 2)
            ctx.save_for_backward(input)
            return y
        
        @ staticmethod
        def backward(ctx, grad_output):
            return grad_output
    sigmoid = torch.square
    tanh = torch.square
else:
    sigmoid = torch.sigmoid
    tanh = torch.tanh
##################### APPROX ACTIVATIONS ########################

# x_train, y_train = adding_problem_generator(100)


def adding_problem_generator(N, seq_len=6, high=1):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:

     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    :param N: the number of the entries.
    :param seq_len: the length of a single sequence.
    :param p: the probability of 1 in generated mask
    :param high: the random data is sampled from a [0, high] uniform distribution.
    :return: (X, Y), X the data, Y the label.
    """
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in range(N):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y


class AddingProblemDataset(Dataset):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:

     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    """

    def __init__(self, N, seq_len=6, high=1):
        """
        Args:
            :param N: the number of the entries.
            :param seq_len: the length of a single sequence.
            :param p: the probability of 1 in generated mask
            :param high: the random data is sampled from a [0, high] uniform distribution.
            :return: (X, Y), X the data, Y the label.
        """
        # X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
        X_num = torch.rand(size=(N, seq_len, 1))
        X_mask = torch.zeros((N, seq_len, 1))
        self.Y = torch.ones((N, 1))
        for i in range(N):
            # Default uniform distribution on position sampling
            positions = np.random.choice(seq_len, size=2, replace=False)
            X_mask[i, positions] = 1
            self.Y[i, 0] = torch.sum(X_num[i, positions])
        self.X = torch.cat((X_num, X_mask), 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'data': self.X[idx], 'target': self.Y[idx]}


class GRUCell(nn.Module):
    """
    An implementation of GRUCell
    """

    def __init__(self, input_size, hidden_size, device, bias=False):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias).to(device)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size,
                             bias=bias).to(device)
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        """

        :param x: (batch_size, seq_len, emsize)
        :param hidden: (batch_size, seq_len, hidden_dim)
        :return: (batch_size, seq_len, hidden_dim)
        """
        gate_x = self.x2h(x)  # xU
        gate_h = self.h2h(hidden)  # hW

        i_r, i_z, i_h = gate_x.chunk(chunks=3, dim=1)
        h_r, h_z, h_h = gate_h.chunk(chunks=3, dim=1)
        b_r, b_z, b_h = self.bias.chunk(chunks=3)

        reset_gate = sigmoid(i_r + h_r + b_r)
        update_gate = sigmoid(i_z + h_z + b_z)
        new_gate = tanh(i_h + (reset_gate * h_h) + b_h)

        hy = new_gate + update_gate * (hidden - new_gate)

        return hy


class GRUModel(nn.Module):
    def __init__(self, batch_size=100,
                 input_size=2,
                 hidden_size=64,
                 num_classes=1,
                 bptt=6,
                 device='cuda:0'):
        super(GRUModel, self).__init__()
        self.device = device
        self.bptt = bptt
        self.hidden_dim = hidden_size

        self.gru_cell = GRUCell(
            input_size=input_size, hidden_size=hidden_size, device=self.device).to(device)
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

        # self.hidden = self.init_hidden(seq_len=self.bptt, bsz=batch_size)

    def init_hidden(self, seq_len=-1, bsz=-1):
        if seq_len == -1:
            seq_len = self.bptt
        if bsz == -1:
            bsz = self.batch_size
        return torch.zeros(bsz, self.hidden_dim).to(self.device)

    def forward(self, sequence):
        batch_size, seq_len, _ = sequence.size()

        # seq_len = min(self.bptt, seq_len)

        self.hidden = self.init_hidden(seq_len=seq_len, bsz=batch_size)
        for t in range(seq_len):
            self.hidden = self.gru_cell(sequence[:, t, :], self.hidden)

        logits = self.fc(self.hidden)

        return logits

    def export(self, sequence, save_path):
        batch_size, seq_len, _ = sequence.size()

        for i in range(batch_size):
            with open(os.path.join(save_path, 'input.csv'), 'w') as f:
                np.savetxt(f, sequence[0].cpu().numpy(), delimiter=',')

        # rnn_out = torch.zeros(batch_size, seq_len, self.hidden_dim)
        self.hidden = self.init_hidden(seq_len=seq_len, bsz=batch_size)

        for seq in range(seq_len):
            # hidden(layer_dim, batch_size, hidden_dim)
            self.hidden = self.gru_cell(sequence[:, seq, :], self.hidden)
            with open(os.path.join(save_path, 'hidden_{}.csv'.format(seq)), 'w') as f:
                np.savetxt(f, self.hidden.clone(
                ).detach().cpu().numpy(), delimiter=',')

        output = self.fc(self.hidden)
        with open(os.path.join(save_path, 'output.csv'), 'w') as f:
            np.savetxt(f, output.clone().detach().cpu().numpy(), delimiter=',')

        return output


class torchModel(nn.Module):
    def __init__(self, device):
        super(torchModel, self).__init__()
        self.device = device

        self.rnn = nn.GRU(input_size=2, hidden_size=64,
                          num_layers=1, batch_first=True).to(self.device)
        self.fc = nn.Linear(in_features=64, out_features=1).to(self.device)

    def forward(self, sequence):
        batch_size, seq_len, _ = sequence.size()
        h0 = torch.randn(1, batch_size, 64).to(self.device)
        output, hn = self.rnn(sequence, h0)
        output = self.fc(output)

        return output



def train(model, cur_epoch, dataloader):
    global lr, best_loss
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch_idx, sampled_batch in enumerate(dataloader):
        optimizer.zero_grad()
        model.zero_grad()

        data = sampled_batch['data'].to(device)
        target = sampled_batch['target'].to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        # for p in model.parameters():
        #     # print(p.data.shape)
        # p.data.add_(-lr, p.grad.data)
        # p.data.add_(-args.decay, p.data)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(
                      cur_epoch, batch_idx, len(dataloader), lr,
                      elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

        # print(output)


def evaluate(dataloader):
    model.eval()
    total_loss = 0.

    criterion = nn.MSELoss()

    for batch_idx, sampled_batch in enumerate(dataloader):
        data = sampled_batch['data'].to(device)
        target = sampled_batch['target'].to(device)
        output = model(data)
        prediction = output
        total_loss += criterion(prediction, target).item()

    return total_loss / (len(dataloader) - 1)

def accumulate_activations(model, data_loader):
    # Test the model
    model.eval()

    _activation = {}
    def get_activation(name):
        def hook(model, input, output):
            _activation[name] = output.detach()
        return hook
    
    model.gru_cell.x2h.register_forward_hook(get_activation('x2h'))
    model.gru_cell.h2h.register_forward_hook(get_activation('h2h'))

    total_loss = 0.
    criterion = nn.MSELoss()

    activation = {}

    for batch_idx, sampled_batch in enumerate(data_loader):
        data = sampled_batch['data'].to(device)
        target = sampled_batch['target'].to(device)
        output = model(data)
        prediction = output
        total_loss += criterion(prediction, target).item()

        # activation accumulate
        gate_x = _activation['x2h']
        gate_h = _activation['h2h']

        i_r, i_z, i_h = gate_x.chunk(chunks=3, dim=1)
        h_r, h_z, h_h = gate_h.chunk(chunks=3, dim=1)
        b_r, b_z, b_h = model.gru_cell.bias.chunk(chunks=3)

        _reset_gate = (i_r + h_r + b_r)
        _update_gate = (i_z + h_z + b_z)
        _new_gate = (i_h + (sigmoid(_reset_gate) * h_h) + b_h)

        _reset_gate = _reset_gate.detach().cpu().numpy()
        _update_gate = _update_gate.detach().cpu().numpy()
        _new_gate = _new_gate.detach().cpu().numpy()
        
        if len(activation) == 0:
            activation['reset_gate'] = _reset_gate
            activation['update_gate'] = _update_gate
            activation['new_gate'] = _new_gate
        else:
            activation['reset_gate'] = np.append(activation['reset_gate'], _reset_gate)
            activation['update_gate'] = np.append(activation['update_gate'], _update_gate)
            activation['new_gate'] = np.append(activation['new_gate'], _new_gate)

    print('Prediction MSE: {:.4f}'.format(total_loss / len(data_loader.dataset)))
    
    return activation

# debug
if debug:
    print('debug')
    model = torch.load(os.path.join(save, 'checkpoint.pt'))
    state_dict = model.state_dict()
    model.eval()

    input_file = np.loadtxt(
        os.path.join(save, 'input_0/input.csv'), delimiter=",", dtype=np.float32)
    operands = []
    for line in input_file:
        if line[1] == 1:
            operands.append(line[0])
    print(operands)
    data = torch.tensor(input_file).view(1, -1, 2).to(device)
    output = model(data)
    print(output)

device = 'cuda:0'
model = GRUModel(bptt=bptt, batch_size=100)
best_val_loss = None

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# train
if if_train:
    print('generating dataset')
    train_dataset = AddingProblemDataset(10000, seq_len=bptt)
    train_loader = DataLoader(
        train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_dataset = AddingProblemDataset(1000, seq_len=bptt)
    val_loader = DataLoader(val_dataset, batch_size=100,
                            shuffle=True, num_workers=4)
    test_dataset = AddingProblemDataset(1000, seq_len=bptt)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=True, num_workers=4)
    print('generating dataset done')
    for cur_epoch in range(epochs):
        epoch_start_time = time.time()
        train(model, cur_epoch, train_loader)
        val_loss = evaluate(val_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} '.format(cur_epoch, (time.time() - epoch_start_time),
                                                                                   val_loss))
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(save, 'checkpoint.pt'), 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss

    test_loss = evaluate(test_loader)
    print('=' * 89)
    print('| End of training | test loss {:5.4f} | test ppl {:8.4f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
else:
    train_dataset = AddingProblemDataset(10000, seq_len=bptt)
    train_loader = DataLoader(
        train_dataset, batch_size=100, shuffle=True, num_workers=4)

    model.load_state_dict(torch.load(os.path.join(save, 'checkpoint.pt')))

    activation = accumulate_activations(model, train_loader)

    print(activation)
    for gate in activation.keys():
        print(gate, np.mean(activation[gate]), np.std(activation[gate]), np.min(activation[gate]), np.max(activation[gate]))
        np.save(os.path.join(save, f'{gate}.npy'),activation[gate])
    

# export
if if_export:
    model = torch.load(os.path.join(save, 'checkpoint.pt'))
    state_dict = model.state_dict()
    model.eval()

    if not os.path.isdir(os.path.join(save, 'weights')):
        os.mkdir(os.path.join(save, 'weights'))

    with torch.no_grad():
        # gru_1_W_* (r, z, h)
        w_r, w_z, w_h = model.gru_cell.x2h.weight.chunk(3, 0)
        np.savetxt(os.path.join(save, 'weights/gru_Wr.csv'),
                   w_r.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/gru_Wz.csv'),
                   w_z.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/gru_Wh.csv'),
                   w_h.cpu().numpy(), delimiter=',')

        # gru_1_U_* (r, z, h)
        u_r, u_z, u_h = model.gru_cell.h2h.weight.chunk(3, 0)
        np.savetxt(os.path.join(save, 'weights/gru_Ur.csv'),
                   u_r.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/gru_Uz.csv'),
                   u_z.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/gru_Uh.csv'),
                   u_h.cpu().numpy(), delimiter=',')

        # gru_1_Wb_* (r, z, h)
        b_r, b_z, b_h = model.gru_cell.bias.chunk(3)
        np.savetxt(os.path.join(save, 'weights/gru_br.csv'),
                   b_r.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/gru_bz.csv'),
                   b_z.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/gru_bh.csv'),
                   b_h.cpu().numpy(), delimiter=',')

        # fc_1_* (w, b)
        np.savetxt(os.path.join(save, 'weights/fc_W.csv'),
                   state_dict['fc.weight'].cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(save, 'weights/fc_b.csv'),
                   state_dict['fc.bias'].cpu().numpy(), delimiter=',')

        # export
        for batch_idx, sampled_batch in enumerate(test_loader):
            export_path = os.path.join(save, 'input_{}'.format(batch_idx))
            
            if not os.path.isdir(export_path):
                os.mkdir(export_path)

            data = sampled_batch['data'].to(device)
            target = sampled_batch['target'].to(device)
            model.export(data, export_path)

            with open(os.path.join(export_path, 'target.csv'), 'w') as f:
                np.savetxt(f, target.cpu().numpy(), delimiter=',')
