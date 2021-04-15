import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

feature = np.load('./feature.npy')
X = feature[:,0:-1]
y = feature[:,-1]
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_train[0:20])


class MyDataSet(Dataset):
    def __init__(self, _x, _y):
        self.X = torch.tensor(_x, dtype=torch.float32)
        self.y = torch.tensor(_y, dtype=torch.long)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.y.shape[0]
    
class ResNet(nn.Module):
    def __init__(self, input_size):
        super(ResNet, self).__init__()
        self.residual = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        out = F.relu(self.residual(x))
        return x + out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(205, 1024)
        self.resnet = ResNet(1024)
        self.linear2 = nn.Linear(1024, 4)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.resnet(out)
        out = self.linear2(out)
        return out
      
  
net = Net()
for p in net.parameters():
    nn.init.normal_(p, mean=0, std=0.02)

loss = nn.CrossEntropyLoss()
    
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

batch_size = 128
# loader = DataLoader(MyDataSet(X, y), batch_size=batch_size)
loader = DataLoader(MyDataSet(X_train, y_train), batch_size=batch_size)
n = y.shape[0]
for epoch in range(500):
    for xi, yi in loader:
        out = net(xi)
        l = loss(out, yi)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch : {}   loss:{:.5f}'.format(epoch + 1, l.item()))
    

    
out = net(torch.tensor(X_train, dtype=torch.float32))
pred_prob = F.softmax(out, dim=1)
total, correct = 0, 0
label = y_train
indexs = torch.argmax(out, dim=1)
for i in range(label.shape[0]):
    total += 1
    c = indexs[i].item()
    if c == int(label[i]):
        correct += 1
print('训练集 acc : %.2f%%' % (correct * 100.0 / total))
# 训练集 acc : 99.94%

out = net(torch.tensor(X_test, dtype=torch.float32))
pred_prob = F.softmax(out, dim=1)
total, correct = 0, 0
label = y_test
indexs = torch.argmax(out, dim=1)
for i in range(label.shape[0]):
    total += 1
    c = indexs[i].item()
    if c == int(label[i]):
        correct += 1
print(total, correct, total-correct)
print('测试集 acc : %.2f%%' % (correct * 100.0 / total))
# 测试集 acc : 99.94%
