from exp5.pyt_model import Model
import pickle
import numpy as np
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

if __name__ == '__main__':
    batch_size = 64
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_batch = DataLoader(train_dataset, batch_size=batch_size)
    test_batch = DataLoader(test_dataset, batch_size=batch_size)
    model = Model()
    opti = SGD(model.parameters(), lr=1e-2)
    cost = CrossEntropyLoss()
    losses = []
    for i in range(3):
        for epoch, (train_x, train_label) in enumerate(train_batch):
            label_np = np.zeros((train_label.shape[0], 10))
            opti.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            losses.append(loss.detach().numpy())
            if epoch % 10 == 0:
                print('epoch: {}, loss: {}'.format(epoch, loss.sum().item()))
            loss.backward()
            opti.step()
    #torch.save(model, 'models.pkl')
    draw_losses(losses)