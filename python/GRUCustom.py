import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import customFunctions

# torch.manual_seed(1)

# hyperparameters
batch_size = 1
sequence_len = 4
hidden_dim = 256
vocab_size = 10
embedding_dim = 16
learning_rate = 0.1
num_epoch = 1
filename = 'gru_copy.pt'
if_train = False
if_save = False
print_state = False

sigmoid = customFunctions.ApproxSigmoid.apply
tanh = customFunctions.ApproxTanh.apply

# sigmoid = torch.sigmoid
# tanh = torch.tanh


def tensor_to_str(tensor):
    target = tensor.squeeze().numpy()
    result = ""
    for element in target:
        result += str(element)
    return result


def export_state_dict(state_dict):
    np.savetxt('weights/word_embeddings.csv',
               state_dict['word_embeddings.weight'].numpy(), delimiter=',')

    # gru_1_W_* (r, z, h)
    print(state_dict['gru_cell.x2h.weight'].size())
    w_r, w_z, w_h = state_dict['gru_cell.x2h.weight'].chunk(3, 0)
    np.savetxt('weights/gru_1_W_r.csv', w_r.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_W_z.csv', w_z.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_W_h.csv', w_h.numpy(), delimiter=',')

    # gru_1_U_* (r, z, h)
    u_r, u_z, u_h = state_dict['gru_cell.h2h.weight'].chunk(3, 0)
    np.savetxt('weights/gru_1_U_r.csv', u_r.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_U_z.csv', u_z.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_U_h.csv', u_h.numpy(), delimiter=',')

    # gru_1_Wb_* (r, z, h)
    print(state_dict['gru_cell.x2h.bias'].size())
    wb_r, wb_z, wb_h = state_dict['gru_cell.x2h.bias'].chunk(3, 0)
    np.savetxt('weights/gru_1_Wb_r.csv', wb_r.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_Wb_z.csv', wb_z.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_Wb_h.csv', wb_h.numpy(), delimiter=',')

    # gru_1_Wb_* (r, z, h)
    ub_r, ub_z, ub_h = state_dict['gru_cell.h2h.bias'].chunk(3, 0)
    np.savetxt('weights/gru_1_Ub_r.csv', ub_r.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_Ub_z.csv', ub_z.numpy(), delimiter=',')
    np.savetxt('weights/gru_1_Ub_h.csv', ub_h.numpy(), delimiter=',')

    # fc_1_* (w, b)
    np.savetxt('weights/fc_1_w.csv',
               state_dict['fc.weight'].numpy(), delimiter=',')
    np.savetxt('weights/fc_1_b.csv',
               state_dict['fc.bias'].numpy(), delimiter=',')

    print('Weight export done')


class GRUCell(nn.Module):
    """
    An implementation of GRUCell
    """

    def __init__(self, input_size, hidden_size, device, bias=False):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
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

    def __init__(self, ntokens, emsize, nhid, batch_size, bptt, device, nlayers=1, dropout=0, tied=0):
        super(GRUModel, self).__init__()
        self.device = device
        self.hidden_dim = nhid
        self.batch_size = batch_size
        self.bptt = bptt
        self.word_embeddings = nn.Embedding(
            num_embeddings=ntokens, embedding_dim=emsize, max_norm=2).to(device)

        self.gru_cell = GRUCell(
            input_size=emsize, hidden_size=nhid, device=device, bias=False)

        # self.gru_cell_1 = GRUCell(input_size=emsize, hidden_size=nhid, device=device).to(device)
        # self.gru_cell_2 = GRUCell(input_size=nhid, hidden_size=nhid, device=device).to(device)
        self.fc = nn.Linear(nhid, ntokens).to(device)

        self.hidden = self.init_hidden(bptt, batch_size)
        # self.hidden_1 = self.init_hidden(bptt, batch_size)

        self.embedded_inputs = []

    def init_hidden(self, seq_len=-1, bsz=-1):
        if seq_len == -1:
            seq_len = self.bptt
        if bsz == -1:
            bsz = self.batch_size
        return torch.zeros(bsz, self.hidden_dim).to(self.device)

    def forward(self, sequence):
        """

        :param sequence: (seq_len, batch_size)
        :return:
        """
        batch_size, seq_len = sequence.size()
        self.hidden = self.init_hidden(seq_len=seq_len, bsz=batch_size)
        rnn_out = torch.zeros(batch_size, seq_len,
                              self.hidden_dim).to(self.device)

        embeds = self.word_embeddings(sequence)
        for t in range(seq_len):
            self.hidden = self.gru_cell(embeds[:, t, :], self.hidden)
            rnn_out[:, t, :] = self.hidden

        # hidden_1 = self.gru_cell_1(embeds, self.hidden_1)
        # self.hidden = self.gru_cell_2(hidden_1, self.hidden)
        # rnn_out = self.hidden
        # for seq in range(seq_len):
        #     self.hidden = self.gru_cell(embeds[seq, :, :], self.hidden)
        #     rnn_out[seq, :, :] = self.hidden

        logits = self.fc(rnn_out)
        # prediction = F.log_softmax(logits, dim=1)

        # return prediction

        return logits

    def export(self, sequence):
        sequence_str = tensor_to_str(sequence)

        with open('encoded_input/%s.csv' % sequence_str, 'w') as f:
            embeds = self.word_embeddings(sequence)
            np.savetxt(f, embeds[0].numpy(), delimiter=',')

        rnn_out = torch.zeros(batch_size, self.bptt, hidden_dim)

        hidden_file = open('hidden_state/%s.csv' % sequence_str, 'w')
        for seq in range(sequence.size(1)):
            # hidden(layer_dim, batch_size, hidden_dim)
            self.hidden = self.gru_cell(embeds[:, seq, :], self.hidden)
            np.savetxt(hidden_file, self.hidden.numpy(), delimiter=',')
            rnn_out[:, seq, :] = self.hidden
        hidden_file.close()

        logits = self.fc(rnn_out)
        prediction = F.log_softmax(logits, dim=1)

        return prediction

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        print(state_dict)
        own_state['word_embeddings.weight'].copy_(
            state_dict['word_embeddings.weight'])
        own_state['gru_cell.x2h.weight'].copy_(
            state_dict['gru_cell.x2h.weight'])
        own_state['gru_cell.h2h.weight'].copy_(
            state_dict['gru_cell.h2h.weight'])
        own_state['gru_cell.x2h.bias'].copy_(state_dict['gru_cell.x2h.bias'])
        own_state['gru_cell.h2h.bias'].copy_(state_dict['gru_cell.h2h.bias'])
        own_state['fc.weight'].copy_(state_dict['fc.weight'])
        own_state['fc.bias'].copy_(state_dict['fc.bias'])

        export_state_dict(state_dict)


# Data generation
def training_set(sequence_len):
    sequence_ = torch.tensor(range(10), dtype=torch.long)
    sequence = sequence_.clone().expand(100, 10)
    sequence = torch.reshape(sequence, (-1,))
    x_ = sequence.view(-1, sequence_len)
    x = x_.clone().detach()
    y_ = torch.cat((sequence, sequence), 0)
    y_ = y_[1:100 * 10 + 1]
    y_ = y_.view(-1, sequence_len)
    y = y_.clone().detach()
    return x, y


def train(data, label):
    # initialize
    optimizer.zero_grad()
    loss = 0

    # detach old hidden state
    model.hidden = model.init_hidden()

    prediction = model(data)

    for s in range(sequence_len):
        loss += criterion(prediction[:, s, :], label[:, s])

    loss.backward()
    optimizer.step()

    return loss / batch_size


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def test_model():
    with torch.no_grad():
        x_test = x[0:5]
        for data in x_test:
            print('Test input: ', data)
            pred = model(data.view(1, -1))
            # pred = model.export(data.view(1, -1))
            print('Test output:', pred.argmax(dim=2))


if __name__ == "__main__":
    model = GRUModel(embedding_dim, hidden_dim, vocab_size)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start = time.time()
    all_losses = []
    loss_avg = 0

    x, y = training_set(sequence_len)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if if_train:
        try:
            for epoch in range(num_epoch):
                for data, label in loader:
                    loss = train(data, label)

                    print("[%s (%d %d%%) %.4f]" % (time_since(start),
                                                   epoch, epoch / num_epoch * 100, loss))

            with torch.no_grad():
                test_model()
            if print_state:
                print("model's state dict:")
                for param_tensor in model.state_dict():
                    print(param_tensor, "\t", model.state_dict()[param_tensor])
        except KeyboardInterrupt:
            print("Stop training")

        finally:
            if if_save:
                print("Saving...")
                torch.save(model.state_dict(), filename)
                print("Saved")
    else:
        model.load_my_state_dict(torch.load(filename))

        test_model()
