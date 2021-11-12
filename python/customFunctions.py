#%%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

_ORDER = 7
# sigmoid3 = [0.5, 0.1424534, -0.0013186]
# tanh5 = [0, 0.6667391, -0.0440927, 0.0010599]

# ARCHIVE
sigmoid5 = [0.5, 0.19131, -0.0045963, 0.0000412332]
sigmoid5 = [0.5, 1.73496, -4.19407, 5.43402, -2.50739]
tanh5 = [1, -8.49814, 11.99804, -6.49478]

# TANH COEFFICIENTS in [-4, 4]
tanh5 = [0.00038260975296624476, 0.7652194684902834, -0.07353682621097166, 0.002702731463794033]  # RMSE: 0.0543
tanh7 = [0.00043379132689107155, 0.8675825874601593, -0.13111610042441557,
         0.010619884719547454, -0.0003063185603617004]  # RMSE: 0.0252

# TANH COEFFICIENTS in [-8, 8]
# tanh5 = [0.00023538462067648527, 0.470769237014534, -0.013706627379808406, 0.00013709227249412444]  # RMSE: 0.1439
# tanh7 = [0.00029254520669058187, 0.5850904016152296, -0.029783042533337054,
        #  0.0006897190966442012, -5.345349316223101e-06]  # RMSE: 0.0952

# SIGMOID COEFFICIENTS
sigmoid3 = [0.5, 0.1424534, -0.0013186]  # x in [-6, 6]
sigmoid5 = [-0.5, 0.19131, -0.0045963, 0.0000412332]  # x in [-4, 4]
sigmoid7 = [0.5, 0.216884, -0.00819276,
            0.000165861, -0.00000119581]  # x in [-7, 7]

# SIGMOID COEFFICIENTS in [-4, 4]
sigmoid5 = [0.5, 0.2395387323043425, -0.013292314184825743, 0.0003761779618728925]  # RMSE: 0.0036
sigmoid7 = [0.5, 0.246915068639199, -0.01744150479946735,
            0.0009466918912718388, -2.2073465028959364e-05]  # RMSE: 0.0009

# SIGMOID COEFFICIENTS in [-8, 8]
sigmoid5 = [0.5, 0.19130488174364327, -0.004596051850950526, 4.223017442715702e-05]  # RMSE: 0.0273
sigmoid7 = [0.5, 0.21689567455156572, -0.008194757398825834,
            0.00016593568955483007, -1.1965564496759948e-06]  # RMSE: 0.0126

tanh3 = [0.00010830490193428313*2, 0.2166098003754834*2, -0.006604707645655612*2]

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)


def check_input(inp, bound: int = 8):
    minval = torch.min(inp)
    maxval = torch.max(inp)
    if minval < (-1) * bound or maxval > bound:
        print(minval, maxval)


def approx_sigmoid(input, order: int = 5):
    if order == 3:
        y = sigmoid3[0]
        for i in range(1, len(sigmoid3)):
            y += sigmoid3[i] * torch.pow(input, 2 * i - 1)
    elif order == 5:
        y = sigmoid5[0]
        for i in range(1, len(sigmoid5)):
            y += sigmoid5[i] * torch.pow(input, 2 * i - 1)
    else:
        y = sigmoid7[0]
        for i in range(1, len(sigmoid7)):
            y += sigmoid7[i] * torch.pow(input, 2 * i - 1)

    return y


def approx_tanh(input, order: int = 5):
    if order == 3:
        y = tanh3[0]
        for i in range(1, len(tanh3)):
            y += tanh3[i] * torch.pow(input, 2 * i - 1)
    elif order == 5:
        y = tanh5[0]
        for i in range(1, len(tanh5)):
            y += tanh5[i] * torch.pow(input, 2 * i - 1)
    else:
        y = tanh7[0]
        for i in range(1, len(tanh7)):
            y += tanh7[i] * torch.pow(input, 2 * i - 1)

    return y


def wide_approx_thunder(x, n, M=4.0, L=2.0, f=approx_tanh):
    y = x / (M * (L ** (n - 1)))
    for i in range(n - 1):
        y = f(y) * L
    return f(y) * M


def wide_approx_tanh(x):
    return approx_tanh(wide_approx_thunder(x, n=3, M=2, L=2, f=approx_tanh))


def wide_approx_sigmoid(x):
    return approx_sigmoid(wide_approx_thunder(x, n=3, M=4, L=2, f=approx_tanh))


class ApproxSigmoid(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, input):
        # check_input(input, 8)
        y = approx_sigmoid(input, order=_ORDER)
        ctx.save_for_backward(input)
        return y

    @ staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad = torch.sigmoid(i) * (1 - torch.sigmoid(i))
        return grad_output * grad


class WideApproxSigmoid(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, input):
        # check_input(input)
        y = wide_approx_sigmoid(input)
        ctx.save_for_backward(input)
        return y

    @ staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad = torch.sigmoid(i) * (1 - torch.sigmoid(i))
        return grad_output * grad


class ApproxTanh(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, input):
        # check_input(input, 4)
        y = approx_tanh(input, order=_ORDER)
        ctx.save_for_backward(input)
        if torch.isnan(y).any():
            print('overflow!')
            # print('overflow!:', input)
            # exit(1)
        return y

    @ staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad = (1 - torch.tanh(i)) * (1 + torch.tanh(i))
        return grad_output * grad


class WideApproxTanh(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, input):
        y = wide_approx_tanh(input)
        ctx.save_for_backward(input)
        if torch.isnan(y).any():
            print('overflow!')
            # print('overflow!:', input)
            # exit(1)
        return y

    @ staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad = (1 - torch.tanh(i)) * (1 + torch.tanh(i))
        return grad_output * grad


class CustomLoss(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, y_pred, target):
        # logits : log_softmax / target : indices
        ctx.save_for_backward(y_pred, target)

        loss = 0

        for i, y in enumerate(target):
            loss -= y_pred[i][y]

        return loss

    @ staticmethod
    def backward(ctx, grad_output):
        y_pred, target = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.zeros(y_pred.size())
        for i, y in enumerate(target):
            grad_input[i][y] = -1

        return grad_input, None


def test_activation():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.rnn = torch.nn.GRU(
                input_size=1, hidden_size=1, batch_first=True)

        def forward(self, x):
            return self.rnn(x)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.plxmodel(), lr=0.01)

    dtype = torch.float
    # device = torch.device("cpu")
    # custom = ApproxTanh.apply

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # N, D_in, H, D_out = 1, 20, 10, 20

    # Create random Tensors to hold input and outputs.

    input = torch.tensor([[i * (-2) for i in range(-10, 10)]], dtype=dtype)
    x = Variable(torch.rand(1, 1, 1), requires_grad=True)
    # hidden = torch.zeros(1, 1, 1)
    net.zero_grad()
    optimizer.zero_grad()

    y = net(x)
    loss = criterion(y, torch.zeros(1))
    # print(loss.data)

    # print(x)
    # print(y)

    y.register_hook(lambda grad: print(grad))
    y.backward()
    # print(y.grad)
    # y.register_hook(save_grad('y'))

    # x = torch.randn(N, D_in, device=device, dtype=dtype)
    # y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Create random Tensors for weights.
    # w1 = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=True)
    # print(x.mm(w1))
    # w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    # y = torch.tanh(x.mm(w1))
    # y_pred = custom(x.mm(w1))

    # loss = (y_pred - y).pow(2).sum()
    # print(loss.item())

    # print("w1:", w1[0])
    loss.backward()
    # print("w1 grad:", w1.grad)


if __name__ == '__main__':
    x = torch.arange(-16, 16, 0.05, dtype=torch.float32)
    # approx = wide_approx_tanh(x.clone())
    torch_y = torch.tanh(x.clone())
    _approx_tanh = ApproxTanh.apply
    _approx_sigmoid = ApproxSigmoid.apply
    _wide_approx_tanh = WideApproxTanh.apply
    _wide_approx_sigmoid = WideApproxSigmoid.apply

    # x = np.arange(-16, 16, 0.05)
    approx_tanh_ = _approx_tanh(x)
    approx_sigmoid_ = _approx_sigmoid(x)
    wide_approx_tanh_ = _wide_approx_tanh(x)
    wide_approx_sigmoid_ = _wide_approx_sigmoid(x)
    torch_tanh_ = torch.tanh(x)
    torch_sigmoid_ = 1 / (1 + torch.exp((-1) * x))

    plt.figure()
    plt.plot(x, torch_sigmoid_, label='numpy')
    plt.plot(x, approx_sigmoid_, label='approx_sigmoid')
    plt.plot(x, wide_approx_sigmoid_, label='wide_approx_sigmoid')
    plt.xlim(-4, 4)
    plt.ylim(0, 1.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, torch_tanh_, label='numpy')
    plt.plot(x, approx_tanh_, label='approx_tanh')
    plt.plot(x, wide_approx_tanh_, label='wide_approx_tanh')
    plt.xlim(-16, 16)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.show()
