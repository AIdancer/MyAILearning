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
```
