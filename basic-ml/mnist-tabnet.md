## 使用tabnet做手写数字识别

### tabnet安装下载
```bash
pip install pytorch-tabnet
```

### 导入相关包
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import lightgbm as lgb
```

### 加载数据
```python
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X = np.array(X)
y = np.array(y, dtype=np.int32)
print(X.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

print("x_train shape : ", x_train.shape)
print("y_train shape : ", y_train.shape)
print("x_test shape : ", x_test.shape)
print("y_test shape : ", y_test.shape)

```

### 创建模型并迭代
```python
clf = TabNetClassifier()
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])
```

### 预测
```python
y_pred = clf.predict(x_train)

y_test_pred = clf.predict(x_test)

print(accuracy_score(y_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

# 输出:
# 0.9875178571428571
# 0.9694285714285714
```

### 对比结果
与[xgboost和LGBM](./mnist算法示例.md)进行对比可以发现，tabnet预测准确率相当不俗，经过适当调优可以击败LightGBM。  
虽然我们测试中不如xgboost，而且训练过程比较慢（因为是CPU模式，GPU模式会更快）；但是可以预估在算力充足，fine-tune更充分的条件下，tabnet有足够的潜力超越xgboost和LGBM。
