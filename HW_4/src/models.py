import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayer(nn.Module):

    def __init__(self, init='uniform'):
        super().__init__()
        self.layer1 = nn.Linear(784, 300, bias=False)
        self.layer2 = nn.Linear(300, 10, bias=False)
        if init == 'uniform':
            torch.nn.init.uniform_(self.layer1.weight, a=-1.0, b=1.0)
            torch.nn.init.uniform_(self.layer2.weight, a=-1.0, b=1.0)
        elif init == 'zero':
            torch.nn.init.zeros_(self.layer1.weight)
            torch.nn.init.zeros_(self.layer2.weight)
        else:
            raise ValueError("Invalid initialization")

    def forward(self, x):
        out = self.layer1(x)
        out = F.sigmoid(out)
        out = self.layer2(out)
        return out


class ThreeLayerScratch:

    def __init__(self, init='uniform'):

        if init == 'uniform':
            self.layer1 = 2*torch.rand(300, 784) - 1
            self.layer2 = 2*torch.rand(10, 300) - 1
        elif init == 'zero':
            self.layer1 = torch.zeros(300, 784)
            self.layer2 = torch.zeros(10, 300)
        else:
            raise ValueError('Invalid weight initialization')

        self.grad1 = torch.zeros_like(self.layer1)
        self.grad2 = torch.zeros_like(self.layer2)
        self.layer1_out = torch.zeros(300)
        self.activations1 = torch.zeros(300)
        self.layer2_out = torch.zeros(10)

    def forward(self, x):
        out = self.layer1 @ x
        self.layer1_out = out.clone()
        out = self.sigmoid(out)
        self.activations1 = out.clone()
        out = self.layer2 @ out
        self.layer2_out = out.clone()
        out = self.softmax(out)
        return out

    # Perform backward pass to compute gradients. Let the gradients accumulate so they can
    # be averaged over a batch.
    def backward(self, x, y, y_t):
        loss_grad = self.cross_entropy_loss_grad(y, y_t)
        output_jac = self.softmax_grad(self.layer2_out)
        sigmoid_grad = self.sigmoid_grad(self.layer1_out)

        self.grad2.add_(torch.outer(output_jac.T @ loss_grad, self.activations1))

        A = (output_jac @ self.layer2).T
        b = A @ loss_grad
        v = torch.mul(sigmoid_grad, b)
        self.grad1.add_(torch.outer(v, x))

    # Computes a single gradient descent step
    def step(self, lr=1e-3):
        self.layer1.add_(-self.grad1, alpha=lr)
        self.layer2.add_(-self.grad2, alpha=lr)

    def zero_grad(self):
        self.grad1 = torch.zeros_like(self.layer1)
        self.grad2 = torch.zeros_like(self.layer2)



    @staticmethod
    def sigmoid(t):
        return 1/(1 + torch.exp(-t))

    def sigmoid_grad(self, t):
        z = self.sigmoid(t)
        return z*(1 - z)

    @staticmethod
    def softmax(v):
        Z = torch.sum(torch.exp(v))
        return torch.exp(v)/Z

    def softmax_grad(self, v):
        s = self.softmax(v)
        return torch.diag(s) - torch.outer(s, s)

    @staticmethod
    def cross_entropy_loss(x, y):
        return -torch.dot(torch.log(x), y)

    @staticmethod
    def cross_entropy_loss_grad(x, y):
        return -torch.mul(torch.pow(x, -1), y)


def test1():
    x = torch.ones(784)
    x[700] = 100
    y_t = torch.zeros(10)
    y_t[3] = 1
    net = ThreeLayerScratch()
    for _ in range(100):
        y = net.forward(x)
        loss = net.cross_entropy_loss(y, y_t)
        print(loss.item())
        net.backward(x, y, y_t)
        net.step(lr=1e-2)
        net.zero_grad()


def test2():
    x = torch.randn(784)
    net = ThreeLayer()
    y = net(x)
    print(y)


if __name__ == '__main__':
    test1()
    #test2()
