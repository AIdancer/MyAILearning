### 多分类示例
    tag : softmax pytorch CrossEntropy
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

cores = [[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]]
X = []
y = []
for i in range(len(cores)):
    val = cores[i]
    core = np.array(val)
    noise = np.random.randn(200, 3) * 0.1
    X.append(noise + core)
    y.append(np.zeros(200) + i)
X = np.vstack(X)
mm = np.max(X, axis=0)
mi = np.min(X, axis=0)

X = (X - mi) / (mm - mi)

features = np.hstack([X, np.array(y).reshape(-1, 1)])
print(features.shape)
np.random.shuffle(features)
print(features[0:10])

class MyDataset(Dataset):
    def __init__(self, features):
        self.X = torch.tensor(features[:, 0:3], dtype=torch.float32)
        self.y = torch.tensor(features[:, 3], dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )

    def forward(self, x):
        ret = self.model(x)
        return ret


net = Net()
for p in net.parameters():
    nn.init.normal_(p, mean=0, std=0.01)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(20):
    sum_loss = 0
    for xi, yi in DataLoader(dataset=MyDataset(features), batch_size=10):
        output = net(xi)
        l = loss(output, yi)
        optimizer.zero_grad()
        l.backward()
        sum_loss += l.item()
        optimizer.step()
    print('epoch : {}   loss : {}'.format(epoch, l.item()))

predict = net(torch.tensor(features[:, 0:3], dtype=torch.float32))

total, correct = 0, 0
label = features[:,3]
indexs = torch.argmax(predict, dim=1)
for i in range(label.shape[0]):
    total += 1
    c = indexs[i].item()
    if c == int(label[i]):
        correct += 1
print('acc : %.2f%%' % (correct * 100.0 / total))

print(predict[0:10])
print(torch.argmax(predict[0:10], dim=1))
print(label[0:10])
```
