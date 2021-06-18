import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from exp5.model import LeNet5
from exp5.layer import Softmax
import time

# 加载数据
def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

# 将label转换成one-hot类型
def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

# 抽取批数据
def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N - batch_size)
    return X[i:i + batch_size], Y[i:i + batch_size]


def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred * Y_true, axis=1)
    for e in M:
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss / N

# 交叉熵损失
class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax._forward(Y_pred)
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout

# 利用Adam优化
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.l = len(params)
        self.parameters = params
        self.moumentum = []
        self.velocities = []
        self.m_cat = []
        self.v_cat = []
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
            self.moumentum.append(np.zeros(param['val'].shape))
            self.v_cat.append(np.zeros(param['val'].shape))
            self.m_cat.append(np.zeros(param['val'].shape))

    def step(self):
        self.t += 1
        for i in range(self.l):
            g = self.parameters[i]['grad']
            self.moumentum[i] = self.beta1 * self.moumentum[i] + (1 - self.beta1) * g
            self.velocities[i] = self.beta2 * self.velocities[i] + (1 - self.beta2) * g * g
            self.m_cat[i] = self.moumentum[i] / (1 - self.beta1 ** self.t)
            self.v_cat[i] = self.velocities[i] / (1 - self.beta2 ** self.t)
            self.parameters[i]['val'] -= self.lr * self.m_cat[i] / (self.v_cat[i] ** 0.5 + self.epislon)

# 准备数据
model = LeNet5('')

batch_size = 64
D_in = 784
D_out = 10
print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

# 准备数据
X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

losses = []
optim = Adam(model.get_params(), lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8)
Loss = CrossEntropyLoss()

ITER = 1000
t_start = time.time()
for i in range(ITER):
    # 获取批数据，并且转化one-hot编码
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
    Y_batch = MakeOneHot(Y_batch, D_out)

    Y_pred = model.forward(X_batch)
    loss, dout = Loss.get(Y_pred, Y_batch)
    model.backward(dout)
    optim.step()

    if i % 100 == 0:
        print("%s%% iter: %s, loss: %s" % (100*i/ITER, i, loss))
        losses.append(loss)


t_end = time.time()
print('Time cost: ', t_end-t_start)
# 保存参数
# weights = model.get_params()
# with open("weights-1000.pkl", "wb") as f:
#     pickle.dump(weights, f)

draw_losses(losses)