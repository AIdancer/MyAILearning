## MNIST数据集算法示例

### 导入包
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb
```

### 加载MNIST数据
```python
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = np.array(X)
y = np.array(y, dtype=np.int32)
print(X.shape)
print(y.shape)
```

### 构建训练集和测试集
```python
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
print("x_train shape : ", x_train.shape)
print("y_train shape : ", y_train.shape)
print("x_test shape : ", x_test.shape)
print("y_test shape : ", y_test.shape)
```

### 构建XGBoost分类器并进行分类
```python
param_dist = {'objective':'multi:softmax', 'n_estimators':50, 'max_depth':5, 'use_label_encoder':False}
xgb_model = XGBClassifier(**param_dist)
xgb_model.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric='mlogloss', verbose=True)

pred_train = xgb_model.predict(x_train)
pred_test = xgb_model.predict(x_test)
print("train score : ", accuracy_score(pred_train, y_train))
print("test score : ", accuracy_score(pred_test, y_test))

# 输出:
# train score :  0.9956607142857142
# test score :  0.9723571428571428
```

### 构建LightGBM分类器并执行分类
```python
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test)

# 二分类
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'l2', 'auc'},
#     'num_leaves': 9,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }

# 多分类
params = {
    'boosting_type': 'gbdt',
    'objective' : 'multiclass',
    'num_class' : 10,
    'metric': 'multi_logloss',
    'num_leaves': 9,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


gbm = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=lgb_eval)

y_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred[0:10])
print(y_train[0:10])
print(accuracy_score(y_train, y_pred))
y_test_pred = gbm.predict(x_test)
y_test_pred = np.argmax(y_test_pred, axis=1)
print(accuracy_score(y_test, y_test_pred))


# 输出:
# [9 8 9 7 0 1 1 5 4 1]
# [9 8 9 7 0 1 1 5 4 1]
# 0.9295178571428572
# 0.9274285714285714
```