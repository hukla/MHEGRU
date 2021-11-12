# coding: utf-8
import argparse
import datetime
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import shutil

import datautil
from GRUCustom import GRUModel

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='custom.pt',
                    help='path to save the final model')  # collect activation if ''
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--decay', type=float, default=1e-2)
parser.add_argument('--logfile', type=str, default='')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = datautil.Corpus(args.data)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    """

    :param data: tockenized data. 1-d LongTensor
    :param bsz: batch_size. int
    :return: batchified data. (*, batch_size) LongTensor
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
# if args.model == 'Transformer': model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
# args.nlayers, args.dropout).to(device) else: model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
# args.nlayers, args.dropout, args.tied).to(device)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if args.model == 'Custom':
    model = GRUModel(ntokens=ntokens, emsize=args.emsize, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout,
                     batch_size=args.batch_size, tied=args.tied, bptt=args.bptt, device=device).to(device)

    if args.save == '':
        model.gru_cell.x2h.register_forward_hook(get_activation('gru_cell.x2h'))
        model.gru_cell.x2h.register_forward_hook(get_activation('gru_cell.h2h'))
else:
    import model

    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(
        device)

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    """
    get batch from batchified data. read sequentially
    :param source: batchfied data. (*, batch_size) LongTensor
    :param i: batch idx. int
    :return data: (seq_len, batch_size)
    :return target: (seq_len * batch_size, 1)
    """
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    # if seq_len < args.bptt:
    #     eos = corpus.dictionary.word2idx['<eos>']
    #     data = F.pad(data, (0, 0, 0, args.bptt - seq_len), "constant", eos)
    #     target = F.pad(target, (0, 0, 0, args.bptt - seq_len), "constant", eos)
    #     padding = torch.zeros(args.bptt-seq_len, device=device, dtype=data.dtype)
    #     data = torch.cat([data, padding], dim=0)
    return data.t().contiguous(), target.t().contiguous().view(-1)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(bsz=eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            elif args.model == 'Custom':
                output = model(data)
                hidden = repackage_hidden(model.hidden)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train(cur_epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(bsz=args.batch_size)

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.decay)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=args.decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=args.decay)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        optimizer.zero_grad()
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
        elif args.model == 'Custom':
            # if batch == 0:
            # seq_len = min(args.bptt, data.shape[0])
            model.hidden = repackage_hidden(model.hidden)
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     # print(p.data.shape)
        #     p.data.add_(-lr, p.grad.data)
        #     p.data.add_(-args.decay, p.data)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

            # tensorboard
            writer.add_scalar('train/loss', cur_loss, cur_epoch * len(train_data) + i)
            writer.add_scalar('train/ppl', math.exp(cur_loss), cur_epoch * len(train_data) + i)

            for name, param in model.named_parameters():
                writer.add_histogram('param/' + name, param.clone().cpu().data.numpy(), cur_epoch * len(train_data) + i)
                if args.model == 'Custom' and args.save == '':
                    writer.add_histogram('act/' + 'x2h', activation['gru_cell.x2h'], cur_epoch * len(train_data) + i)
                    writer.add_histogram('act/' + 'h2h', activation['gru_cell.h2h'], cur_epoch * len(train_data) + i)
            # print(torch.cuda.memory_allocated())

            writer.flush()

        del loss, output


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# Writer will output to ./runs/ directory by default
if args.logfile == '':
    runfile = os.path.join('runs', 'ptb_{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
else:
    runfile = args.logfile

if os.path.isdir(runfile):
    shutil.rmtree(runfile)
writer = SummaryWriter(runfile)

# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.load:
        model = torch.load(args.save)
        # model.eval()
        print('model loaded: %s' % args.save)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        # tensorboard
        writer.add_scalar('valid/loss', val_loss, epoch * len(train_data))
        writer.add_scalar('valid/ppl', math.exp(val_loss), epoch * len(train_data))
        writer.add_scalar('train/lr', lr, epoch * len(train_data))

        # writer.add_histogram('train/weight', model.parameters())

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if args.save != '':
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    for var_name in model.state_dict():
                        print(var_name, '\t', model.state_dict()[var_name].shape)
                    print('model saved')
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
if args.save != '':
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

writer.close()


def stringify(encoded_input):
    output_string = ""
    for word in encoded_input:
        output_string += (corpus.dictionary.idx2word[word] + " ")

    return output_string


data, targets = get_batch(test_data, 1)
print(data.shape, targets.shape)
print('data')
for d in data:
    print(stringify(d))
print('\ntarget')
for t in targets.reshape(args.bptt, eval_batch_size):
    print(stringify(t))
output = model(data)
print(output.shape)
pred = output.argmax(dim=2)
print(pred.shape)
print('\npred')
for p in pred:
    print(stringify(p))
output_flat = output.view(-1, ntokens)
loss = len(data) * criterion(output_flat, targets).item()
print('\nloss:{:.4f} ppl:{:8.2f}'.format(loss, math.exp(loss)))
