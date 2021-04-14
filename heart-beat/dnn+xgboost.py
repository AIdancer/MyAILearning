import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

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
      
# v1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(205, 1024),
            nn.ReLU()
        )
        self.decoder = nn.Linear(1024, 4)

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode
    
net = Net()

loss = nn.CrossEntropyLoss()

for p in net.parameters():
    nn.init.normal_(p, mean=0, std=0.01)
    
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

batch_size = 128
loader = DataLoader(MyDataSet(X, y), batch_size=batch_size)
# loader = DataLoader(MyDataSet(X_train, y_train), batch_size=batch_size)
n = y.shape[0]
for epoch in range(300):
    for xi, yi in loader:
        encode, decode = net(xi)
        l = loss(decode, yi)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch : {}   loss:{:.5f}'.format(epoch + 1, l.item()))
    
x_new, _ = net(torch.tensor(feature[:,0:-1], dtype=torch.float32))
y_new = np.array(feature[:,-1], dtype=np.int32)
x_new = x_new.detach().numpy()

print(x_new.shape)
X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

param_dist = {'objective':'multi:softmax', 'n_estimators':600, 'max_depth':6, 'use_label_encoder':False}
xgb_model = XGBClassifier(**param_dist)

xgb_model.fit(x_new, y_new, eval_set=[(x_new, y_new)], eval_metric='mlogloss', verbose=True)

pred = xgb_model.predict(X_train)
pred_test = xgb_model.predict(X_test)
prob = xgb_model.predict_proba(X_test)

print('train score : {:.5f}'.format(accuracy_score(pred, y_train)))
print('test score : {:.5f}'.format(accuracy_score(pred_test, y_test)))

# 识别准确率可以达到100%
# train score : 1.00000
# test score : 1.00000

# 预测testA
df = pd.read_csv('/home/users/liqiaz/data/heart-beat/testA.csv')
aX = []
n = df.shape[0]
for i in range(n):
    signal_str = df.loc[i, 'heartbeat_signals']
    signals = np.array(list(map(float, signal_str.split(','))))
    aX.append(signals)
aX = np.array(aX)

ax_new, _ = net(torch.tensor(aX, dtype=torch.float32))
ax_new = ax_new.detach().numpy()
a_pred = xgb_model.predict(ax_new)
print(a_pred[0:10])

res = []
n = a_pred.shape[0]
for i in range(n):
    aid = df.loc[i,'id']
    values = [0, 0, 0, 0]
    index = a_pred[i]
    values[index] = 1
    res.append([aid] + values)
ans = pd.DataFrame(res, columns=['id', 'label_0', 'label_1', 'label_2', 'label_3'])
ans.to_csv('./ans.csv', index=False)
