from exp5.model import LeNet5
import pickle
import random
import numpy as np

def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N - batch_size)
    return X[i:i + batch_size], Y[i:i + batch_size]

X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

model = LeNet5('weights-10000.pkl')

test_size = 1000
print("test_size = " + str(test_size))
X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
X_test_min,  Y_test_min  = get_batch(X_test,  Y_test,  test_size)

# TRAIN SET ACC
Y_pred_min = model.forward(X_train_min)
result = np.argmax(Y_pred_min, axis=1) - Y_train_min
result = list(result)
print("训练集测试--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", accuracy=" + str(result.count(0)/X_train_min.shape[0]))

# TEST SET ACC
Y_pred_min = model.forward(X_test_min)
result = np.argmax(Y_pred_min, axis=1) - Y_test_min
result = list(result)
print("测试集测试--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", accuracy=" + str(result.count(0)/X_test_min.shape[0]))