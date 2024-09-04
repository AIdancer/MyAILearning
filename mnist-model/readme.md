### 加载数据
数据可从kaggle下载csv版本  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datadir = r"D:\moon\data\ml-data\mnist"

def load_mnist_train():
    fpath = datadir + "\mnist_train.csv"
    df = pd.read_csv(fpath)
    vecs = np.array(df)
    X = vecs[:, 1:]
    y = vecs[:, 0].ravel()
    return (X, y)

def load_mnist_test():
    fpath = datadir + "\mnist_test.csv"
    df = pd.read_csv(fpath)
    vecs = np.array(df)
    X = vecs[:, 1:]
    y = vecs[:, 0].ravel()
    return (X, y)

if __name__ == "__main__":
    X_train, y_train = load_mnist_train()
    print(X_train.shape, y_train.shape)
    X_test, y_test = load_mnist_test()
    print(X_test.shape, y_test.shape)
```
