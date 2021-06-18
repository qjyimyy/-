import pickle
from exp5.layer import Conv
from exp5.layer import ReLU
from exp5.layer import MaxPool
from exp5.layer import FC

# 定义LeNet5网络模型
class LeNet5():

    def __init__(self, weights):
        self.conv1 = Conv(1, 6, 28, 28, 5, 1, 2)  # 输入通道数，输出通道数，输入尺寸，卷积核大小，步长，填充/输出28*28*6
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2, 2)  # 填充大小/输出14*14*6
        self.conv2 = Conv(6, 16, 14, 14, 5, 1, 0)  # 输出10*10*16
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2, 2)  # 输出5*5*16
        self.FC1 = FC(16 * 5 * 5, 120) # 输入神经元数，输出神经元数，相当于全连接，输出1*1*120
        self.ReLU3 = ReLU()
        self.FC2 = FC(120, 84)
        self.ReLU4 = ReLU()
        self.FC3 = FC(84, 10)
        self.p2_shape = None

        if weights == '':
            pass
        else:
            with open(weights, 'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.ReLU1._forward(h1)
        p1 = self.pool1._forward(a1)
        h2 = self.conv2._forward(p1)
        a2 = self.ReLU2._forward(h2)
        p2 = self.pool2._forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0], -1)  # Flatten 转化为列向量
        h3 = self.FC1._forward(fl)
        a3 = self.ReLU3._forward(h3)
        h4 = self.FC2._forward(a3)
        a5 = self.ReLU4._forward(h4)
        h5 = self.FC3._forward(a5)
        return h5

    def backward(self, dout):
        dout = self.FC3._backward(dout)
        dout = self.ReLU4._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU3._backward(dout)
        dout = self.FC1._backward(dout)
        dout = dout.reshape(self.p2_shape)  # reshape
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b,
                self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b,
         self.FC3.W, self.FC3.b] = params