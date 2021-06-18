import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch

# 加载模型以及数据
batch_size = 1000
model = torch.load('models.pkl')
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
test_batch = DataLoader(test_dataset, batch_size=batch_size)
correct = 0
sum = 0

# 计算准确率
for idx, (test_x, test_label) in enumerate(test_batch):
    predict = model(test_x.float()).detach()
    predict = np.argmax(predict, axis=-1)
    label_np = test_label.numpy()
    accuracy = predict == test_label
    correct += np.sum(accuracy.numpy(), axis=-1)
    sum += accuracy.shape[0]

print('accuracy: {:.2f}'.format(correct / sum))