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

### xgboost预测mnist
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from load_data import load_mnist_train, load_mnist_test

if __name__ == "__main__":
    X_train, y_train = load_mnist_train()
    print("loading training data...")
    print(X_train.shape, y_train.shape)
    param_dist = {'objective':'multi:softmax', 'n_estimators':500, 'max_depth':6, 'use_label_encoder':False, 'eval_metric':'mlogloss'}
    xgb_model = XGBClassifier(**param_dist)
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=True)
    print("loading testing data...")
    X_test, y_test = load_mnist_test()
    pred_test = xgb_model.predict(X_test)
    print("test_scroe : {}".format(accuracy_score(pred_test, y_test)))
# test_scroe : 0.9812
```

### mnist-DL model (MLP)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

class MyDataSet(Dataset):
    def __init__(self, _x, _y):
        self.X = torch.tensor(_x, dtype=torch.float32)
        self.y = torch.tensor(_y, dtype=torch.long)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.y.shape[0]
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        ret = self.model(x)
        return ret

if __name__ == "__main__":
    train_X, train_y = load_mnist_train()
    print(train_X.shape, train_y.shape)
    net = Net()
    for p in net.parameters():
        nn.init.normal_(p, mean=0, std=0.02)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
    
    batch_size = 64
    loader = DataLoader(MyDataSet(train_X, train_y), batch_size=batch_size)
    
    for epoch in range(50):
        for xi, yi in loader:
            out = net(xi)
            l = loss(out, yi)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch : {}   loss:{:.5f}'.format(epoch + 1, l.item()))
        
    test_X, test_y = load_mnist_test()
    out = net(torch.tensor(test_X, dtype=torch.float32))
    pred_prob = F.softmax(out, dim=1)
    total, correct = 0, 0
    label = test_y
    indexs = torch.argmax(out, dim=1)
    for i in range(label.shape[0]):
        total += 1
        c = indexs[i].item()
        if c == int(label[i]):
            correct += 1
    print(total, correct, total-correct)
    print('测试集 acc : %.2f%%' % (correct * 100.0 / total))
# 测试集 acc : 98.49%
```
