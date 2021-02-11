from __future__ import print_function
from tc_net import NetMnist, NetImageClass
import unittest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from pathlib import Path
import requests


class MyTorch(unittest.TestCase):
    def test_hello(self):
        torch.manual_seed(1)
        x = torch.randn(5, 4, 3, 2)
        y = x.view(120)
        size = x.size()[1:]  # all dimensions except the batch dimension
        n = 1
        for s in size:
            n *= s
        z = x.view(-1, n)
        print(z)

    def test_back(self):
        torch.manual_seed(1)
        x = torch.rand(1, requires_grad=True)
        print("x", x)
        y = x * x
        print("y", y)
        out = y.mean()
        print("out", out)
        out.backward()
        print(x.grad)

    def test_NetMnist(self):
        net = NetMnist()
        print(net)
        params = list(net.parameters())
        for i in range(len(params)):
            print(params[i].size())

        input = torch.randn(1, 1, 32, 32)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = net(input)
        target = torch.randn(10)  # a dummy target, for example
        target = target.view(1, -1)  # make it the same shape as output
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(loss)

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def test_NetImageClassTraining(self):
        net = NetImageClass("./data")
        net.htraining('cifar_net.pth')

    def test_NetImageClassPredict(self):
        path = './data'
        net = NetImageClass(path)
        net.load_state_dict(torch.load(path+'/cifar_net.pth'))
        dataiter = iter(net.testloader)
        images, labels = dataiter.next()
        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % net.classes[labels[j]] for j in range(4)))

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % net.classes[predicted[j]]
                                      for j in range(4)))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in net.testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def test_gpu(self):
        net = NetImageClass('./data')
        net.useGpu()

    def test_nn(self):
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for t in range(500):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            if t % 100 == 99:
                print(t, loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    def test_foo(self):
        DATA_PATH = Path("data")
        PATH = DATA_PATH / "mnist"

        PATH.mkdir(parents=True, exist_ok=True)

        URL = "http://deeplearning.net/data/mnist/"
        FILENAME = "mnist.pkl.gz"

        if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)

    def kwargs(self, **kwargs):
        for key, value in kwargs.items():
            print(key, value)

    def args(self, *args):
        for v in args:
            print(v)

    def test_etc(self):
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        self.kwargs(**cuda_kwargs)
        self.args(*cuda_kwargs)
