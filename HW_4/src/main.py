import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from argparse import ArgumentParser
from models import *
from tqdm import tqdm
import os


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model', type=str, default='scratch', choices=['scratch', 'torch'],
                        help='run with autograd or from scratch')
    parser.add_argument('--initialization', type=str, default='uniform', choices=['uniform', 'zero'],
                        help='weight initialization')
    return parser


def build_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    trainset = MNIST(root='../data/mnist', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = MNIST(root='../data/mnist', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    return trainloader, testloader


def build_model(args):
    net = {
        'scratch': ThreeLayerScratch,
        'torch': ThreeLayer
    }[args.model](init=args.initialization)
    return net


def train(args, net, optimizer, loss_fn, data_loader, epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    total = 0
    correct = 0
    m = len(data_loader)

    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
        if args.model == 'scratch':
            assert isinstance(net, ThreeLayerScratch)
            n = len(targets)
            net.zero_grad()
            batch_loss = 0
            batch_total = 0
            batch_correct = 0
            for i in range(n):
                y_t = torch.zeros(10)
                y_t[targets[i]] = 1
                x = inputs[i, :]
                y = net.forward(x)
                batch_loss += (1 / n) * net.cross_entropy_loss(y, y_t).item()
                net.backward(x, y, y_t)
                _, predicted = y.max(0)
                batch_total += 1
                batch_correct += predicted.eq(targets[i])

            train_loss += (1/m)*batch_loss
            total += batch_total
            correct += batch_correct
            # average the gradients
            net.grad1.mul_(1/n)
            net.grad2.mul_(1/n)
            net.step(lr=args.lr)

        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += (1/m)*loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    error = 1 - (correct/total)
    accuracy = 100. * (correct/total)
    print('train error {:.3f}'.format(error))
    print('train accuracy {:.3f}'.format(accuracy))
    print('loss {:.3f}'.format(train_loss))

    return error, train_loss


def test(args, net, data_loader, loss_fn):
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        m = len(data_loader)
        if args.model == 'scratch':
            assert isinstance(net, ThreeLayerScratch)
            n = len(targets)
            batch_loss = 0
            batch_total = 0
            batch_correct = 0
            for i in range(n):
                y_t = torch.zeros(10)
                y_t[targets[i]] = 1
                x = inputs[i, :]
                y = net.forward(x)
                batch_loss += (1 / n) * net.cross_entropy_loss(y, y_t).item()
                _, predicted = y.max(0)
                batch_total += 1
                batch_correct += predicted.eq(targets[i])

            test_loss += (1 / m) * batch_loss
            total += batch_total
            correct += batch_correct

        else:
            with torch.no_grad():
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += (1/m)*loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    error = 1 - (correct / total)
    print('test error {:.3f}'.format(error))
    print('loss {:.3f}'.format(test_loss))

    return error, test_loss


def main():
    parser = get_parser()
    args = parser.parse_args()
    trainloader, testloader = build_dataset()
    net = build_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.model == 'torch':
        optimizer = SGD(net.parameters(), args.lr, momentum=0, weight_decay=0)
    else:
        optimizer = None

    train_errors = []
    test_errors = []
    train_losses = []
    test_losses = []

    for epoch in range(50):
        train_error, train_loss = train(args, net, optimizer, loss_fn, trainloader, epoch)
        test_error, test_loss = test(args, net, testloader, loss_fn)

        train_errors.append(train_error)
        test_errors.append(test_error)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    if not os.path.isdir('curve'):
        os.mkdir('curve')

    save_name = '{}-lr_{}-init_{}'.format(args.model, args.lr, args.initialization)
    #torch.save({'train_err': train_errors, 'test_err': test_errors, 'train_loss': train_losses,
                #'test_loss': test_losses}, os.path.join('curve', save_name))


if __name__ == "__main__":
    main()


