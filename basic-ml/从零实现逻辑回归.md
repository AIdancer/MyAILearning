```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 以两个点为中心生成测试数据
def get_samples():
    features = []
    labels = []
    N = 100
    c0 = np.array([0, 0])
    c1 = np.array([3, 3])
    X0 = c0 + np.random.normal(0, 0.5, size=(N, c0.shape[0]))
    y0 = np.zeros(N, dtype=np.int)
    X1 = c1 + np.random.normal(0, 0.5, size=(N, c1.shape[0]))
    y1 = np.ones(N, dtype=np.int)
    # plt.scatter(X0[:,0], X0[:,1], c='b', marker='+')
    # plt.scatter(X1[:,0], X1[:,1], c='r', marker='x')
    # plt.show()
    features = np.vstack([X0, X1])
    labels = np.hstack([y0, y1])
    return features, labels


# 计算逻辑回归模型参数值（自己实现梯度更新）
def logistic(X, y):
    theta = np.random.normal(0, 1, size=(X.shape[1]))
    b = np.random.normal(0, 1)
    print(X.shape, y.shape)
    learning_rate = 0.5
    for k in range(20):
        sum_loss = 0.0
        for i in range(X.shape[0]):
            tx = np.dot(theta, X[i]) + b
            ex = np.exp(-tx)
            tmp = ex / ((1 + ex) * (1+ex))
            delta = y[i] - 1.0 / (1 + ex)
            sum_loss += np.sqrt(delta * delta)
            grad_theta = -delta * tmp * X[i]
            grad_b = -delta * tmp
            theta = theta - learning_rate * grad_theta
            b = b - learning_rate * grad_b
        print('iteration : ', k+1, sum_loss, theta, b)
    return theta, b

# 预测
def predict(theta, b, X, y):
    yes, no = 0, 0
    for i in range(X.shape[0]):
        tx = np.dot(theta, X[i]) + b
        prob = 1.0 / (1.0 + np.exp(-tx))
        label = None
        if prob > 0.8:
            label = 1
        else:
            label = 0
        if label == y[i]:
            yes += 1
        else:
            no += 1
    print('acc : %.2f%%' % (yes * 100.0 / (yes + no)))


if __name__ == '__main__':
    X, y = get_samples()
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    theta, b = logistic(X_train, y_train)
    print(theta, b)
    predict(theta, b, X_test, y_test)

```
