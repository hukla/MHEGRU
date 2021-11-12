import argparse
import datetime
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import customFunctions

# Train configuration
parser = argparse.ArgumentParser()
parser.add_argument('--if_train', type=bool, default=False)
parser.add_argument('--if_export', type=bool, default=False)
parser.add_argument('--output_path', type=str,
                    default='cpp_input/MNIST')
parser.add_argument('--reshape', type=bool, default=True)
parser.add_argument('--activation', type=int, default=0,
                    help='0: original, 1: approx, 2: wide approx (default: approx)')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--device', type=str, default='')
parser.add_argument('--checkpoint', type=str,
                    default='cpp_input/MNIST/model_h64')
args = parser.parse_args()
if_train = args.if_train

# Device configuration
if args.device == '':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = args.hidden_size
num_layers = 1
num_classes = 10

if if_train:
    batch_size = 100
else:
    batch_size = 1

num_epochs = args.epochs

# Activation functions
if args.activation == 1:
    # LS approximation
    sigmoid = customFunctions.ApproxSigmoid.apply
    tanh = customFunctions.ApproxTanh.apply
elif args.activation == 2:
    # wide LS approximation
    sigmoid = customFunctions.WideApproxSigmoid.apply
    tanh = customFunctions.WideApproxTanh.apply
else:
    # original activation
    sigmoid = torch.sigmoid
    tanh = torch.tanh

# create tensorboard runfile
if if_train:
    logdir = os.path.join(args.output_path, 'checkpoints')
    if args.checkpoint == '':
        dt = datetime.datetime.now()
        runfile = 'mnist_{}'.format(dt.strftime('%y%m%d%H%M'))
    else:
        runfile = args.checkpoint

    runfile_path = os.path.join('runs', runfile)
    if os.path.isdir(runfile_path):
        shutil.rmtree(runfile_path)
    writer = SummaryWriter(runfile_path)
    print('tensorboard runfile at runs/{}'.format(runfile))


global_step = 0

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),  # convert image to tensor
                                               # normalize inputs
                                               transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.Compose([
                                               transforms.ToTensor(),  # convert image to tensor
                                               # normalize inputs
                                               transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                          ]))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.gru = nn.GRU(input_size, hidden_size,
        #   num_layers, batch_first=True)
        self.gru = GRUCell(input_size=input_size,
                           hidden_size=hidden_size, device=device, bias=True, activation=args.activation)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # out, _ = self.gru(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class GRUCell(nn.Module):
    """
    An implementation of GRUCell
    """

    def __init__(self, input_size, hidden_size, device, bias=False, activation=0):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias).to(device)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size,
                             bias=bias).to(device)
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        self.reset_parameters()

        # Activation functions
        if activation == 1:
            # LS approximation
            self.sigmoid = customFunctions.ApproxSigmoid.apply
            self.tanh = customFunctions.ApproxTanh.apply
        elif activation == 2:
            # wide LS approximation
            self.sigmoid = customFunctions.WideApproxSigmoid.apply
            self.tanh = customFunctions.WideApproxTanh.apply
        else:
            # original activation
            self.sigmoid = torch.sigmoid
            self.tanh = torch.tanh

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

        _reset_gate = (i_r + h_r + b_r)
        _update_gate = (i_z + h_z + b_z)

        reset_gate = self.sigmoid(_reset_gate)
        update_gate = self.sigmoid(_update_gate)

        _new_gate = (i_h + (reset_gate * h_h) + b_h)
        new_gate = self.tanh(_new_gate)

        hy = new_gate + update_gate * (hidden - new_gate)

        # return hy
        return hy, _reset_gate, _update_gate, _new_gate


class MHERNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, activation):
        super(MHERNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = GRUCell(
            input_size=input_size, hidden_size=hidden_size, device=device, bias=False, activation=activation)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)

        for t in range(seq_len):
            hidden, _reset_gate, _update_gate, _new_gate = self.gru(
                x[:, t, :], hidden)

        if if_train:
            writer.add_histogram('train/_reset_gate', _reset_gate, global_step)
            writer.add_histogram('train/_update_gate', _update_gate, global_step)
            writer.add_histogram('train/_new_gate', _new_gate, global_step)

        logit = self.fc(hidden)

        return logit

    def export(self, sequence, export_path):
        batch_size, seq_len, _ = sequence.size()

        for i in range(batch_size):
            with open(os.path.join(export_path, 'input.csv'), 'w') as f:
                np.savetxt(f, sequence[0].cpu().numpy(), delimiter=',')

        # rnn_out = torch.zeros(batch_size, seq_len, self.hidden_dim)
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        # self.hidden = self.init_hidden(seq_len=seq_len, bsz=batch_size)

        for seq in range(seq_len):
            # hidden(layer_dim, batch_size, hidden_dim)
            hidden, _, _, _ = self.gru(sequence[:, seq, :], hidden)
            with open(os.path.join(export_path, 'hidden_{}.csv'.format(seq)), 'w') as f:
                np.savetxt(f, hidden.detach().cpu().numpy(), delimiter=',')

        output = self.fc(hidden)
        with open(os.path.join(export_path, 'output.csv'), 'w') as f:
            np.savetxt(f, output.clone().detach().cpu().numpy(), delimiter=',')

        return output


def test(model):
    # Test the model
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        activation = {}
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if if_train:
            logger.write('Test Accuracy of the model on the 10000 test images: {} %\n'.format(
                100 * correct / total))
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(
            100 * correct / total))
    
    return activation

def accumulate_activations(model, data_loader):
    # Test the model
    model.eval()

    _activation = {}
    def get_activation(name):
        def hook(model, input, output):
            _activation[name] = output.detach()
        return hook
    
    ctx_model.gru.x2h.register_forward_hook(get_activation('x2h'))
    ctx_model.gru.h2h.register_forward_hook(get_activation('h2h'))

    with torch.no_grad():
        correct = 0
        total = 0
        activation = {}
        for images, labels in data_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # activation accumulate
            gate_x = _activation['x2h']
            gate_h = _activation['h2h']

            i_r, i_z, i_h = gate_x.chunk(chunks=3, dim=1)
            h_r, h_z, h_h = gate_h.chunk(chunks=3, dim=1)
            b_r, b_z, b_h = model.gru.bias.chunk(chunks=3)

            _reset_gate = (i_r + h_r + b_r)
            _update_gate = (i_z + h_z + b_z)
            _new_gate = (i_h + (model.gru.sigmoid(_reset_gate) * h_h) + b_h)

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

        print('Test Accuracy of the model on the {} images: {} %'.format(total,
            100 * correct / total))
    
    return activation


def reshape(input_arr, res_size: tuple):
    res = np.zeros(res_size)
    if res_size[1] == 1:
        for i in range(len(input_arr)):
            res[i] = input_arr[i]
    else:
        for i in range(len(input_arr)):
            res[i][:len(input_arr[i])] = input_arr[i]
    return res


plx_model = MHERNN(input_size, hidden_size, num_layers,
                   num_classes, 0).to(device)  # activation = torch
ctx_model = MHERNN(input_size, hidden_size, num_layers,
                   num_classes, args.activation).to(device)  # activation = poly

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(plx_model.parameters(
), lr=args.learning_rate, weight_decay=args.weight_decay)

if if_train:
    # create folder for current experiments
    checkpoint_path = os.path.join(logdir, runfile)
    if os.path.isdir(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.mkdir(checkpoint_path)
    shutil.copy2('MNIST.py', checkpoint_path)
    logger = open(os.path.join(checkpoint_path, 'log.txt'), 'w+')
    logger.write('activation={}, hidden_size={}, learning_rate={}, weight_decay={}\n\n'.format(
        args.activation, args.epochs, args.hidden_size, args.weight_decay))

    # Train the model
    total_step = len(train_loader)
    best_loss = 100
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = plx_model(images)
            global_step += 1
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logger.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            if loss.item() < best_loss:
                best_loss = loss.item()
                # Save the model checkpoint
                torch.save(plx_model.state_dict(), args.checkpoint)
                # print(best_loss)

        test(plx_model)
        ctx_model.load_state_dict(torch.load(args.checkpoint))
        test(ctx_model)

    # Save the model checkpoint
    # torch.save(plx_model.state_dict(), os.path.join(
        # args.output_path, args.checkpoint))
else:
    ctx_model.load_state_dict(torch.load(args.checkpoint))

    activation = accumulate_activations(ctx_model, train_loader)

    print(activation)
    for gate in activation.keys():
        print(gate, np.mean(activation[gate]), np.std(activation[gate]), np.min(activation[gate]), np.max(activation[gate]))
        np.save(os.path.join(args.output_path, f'{gate}.npy'),activation[gate])
    

if args.if_export:
    ctx_model.load_state_dict(torch.load(args.checkpoint))
    ctx_model.eval()

    # Export the model weight
    with torch.no_grad():
        # gru_1_W_* (r, z, h)
        w_r, w_z, w_h = ctx_model.gru.x2h.weight.chunk(3, 0)

        w_r = w_r.cpu().numpy()
        w_z = w_z.cpu().numpy()
        w_h = w_h.cpu().numpy()

        if args.reshape:
            w_r = reshape(w_r, (hidden_size, 32))
            w_z = reshape(w_z, (hidden_size, 32))
            w_h = reshape(w_h, (hidden_size, 32))

        np.savetxt(os.path.join(args.output_path, 'weights/gru_Wr.csv'),
                   w_r, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/gru_Wz.csv'),
                   w_z, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/gru_Wh.csv'),
                   w_h, delimiter=',')

        # gru_1_U_* (r, z, h)
        u_r, u_z, u_h = ctx_model.gru.h2h.weight.chunk(3, 0)

        u_r = u_r.cpu().numpy()
        u_z = u_z.cpu().numpy()
        u_h = u_h.cpu().numpy()

        np.savetxt(os.path.join(args.output_path, 'weights/gru_Ur.csv'),
                   u_r, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/gru_Uz.csv'),
                   u_z, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/gru_Uh.csv'),
                   u_h, delimiter=',')

        # gru_1_Wb_* (r, z, h)
        b_r, b_z, b_h = ctx_model.gru.bias.chunk(3)

        b_r = b_r.cpu().numpy()
        b_z = b_z.cpu().numpy()
        b_h = b_h.cpu().numpy()

        np.savetxt(os.path.join(args.output_path, 'weights/gru_br.csv'),
                   b_r, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/gru_bz.csv'),
                   b_z, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/gru_bh.csv'),
                   b_h, delimiter=',')

        # fc_1_* (w, b)
        fc_w = ctx_model.state_dict()['fc.weight'].cpu().numpy()
        fc_b = ctx_model.state_dict()['fc.bias'].cpu().numpy()

        if args.reshape:
            fc_w = reshape(fc_w, (16, hidden_size))
            fc_b = reshape(fc_b, (16, 1))

        np.savetxt(os.path.join(args.output_path, 'weights/fc_W.csv'),
                   fc_w, delimiter=',')
        np.savetxt(os.path.join(args.output_path, 'weights/fc_b.csv'),
                   fc_b, delimiter=',')

        # export
        batch_idx = 0
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            data = images.reshape(-1, sequence_length, input_size).to(device)
            target = labels.to(device)

            export_path = os.path.join(
                args.output_path, 'input_{}'.format(batch_idx))
            if not os.path.isdir(export_path):
                os.mkdir(export_path)

            output = ctx_model.export(data, export_path)

            pred = torch.argmax(output)
            total += target.size(0)
            correct += (pred == target).sum().item()

            with open(os.path.join(export_path, 'target.csv'), 'w') as f:
                np.savetxt(f, target.cpu().numpy(), delimiter=',')

            batch_idx += 1

        print(100 * correct / total)
