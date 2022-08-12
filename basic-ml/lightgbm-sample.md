```python
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

n = 500
class_a = np.random.normal(loc=(1,1), scale=0.5, size=(n, 2))
y_a = np.ones(n, dtype=np.int32)
class_b = np.random.normal(loc=(-1,-1), scale=0.5, size=(n, 2))
y_b = np.zeros(n, dtype=np.int32)
X = np.vstack([class_a, class_b])
y = np.hstack([y_a, y_b])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

plt.scatter(class_a[:,0], class_a[:,1])
plt.scatter(class_b[:,0], class_b[:,1])
# plt.scatter(X[:,0], X[:,1])

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval)

# 预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
result = (y_pred > 0.5).astype(np.int32)
print('acc score:')
print(accuracy_score(y_test, result))
# 评估
print('预估结果的rmse为:')
print(mean_squared_error(y_test, y_pred) ** 0.5)
```
