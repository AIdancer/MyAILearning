import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier

import pickle

feature = np.load('./feature.npy')

X = feature[:,0:-1]
y = feature[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

param_dist = {'objective':'multi:softmax', 'n_estimators':100, 'max_depth':5, 'num_class':4}
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='mlogloss', verbose=True)

pickle.dump(xgb_model, open('xgb.dat', 'wb'))

pred = xgb_model.predict(X_train)
pred_test = xgb_model.predict(X_test)

print('train score : ', accuracy_score(pred, y_train))
print('test score : ', accuracy_score(pred_test, y_test))

# train score :  0.9991142857142857
# test score :  0.9832666666666666

pred = svm_clf.predict(X_train)
pred_test = svm_clf.predict(X_test)
print('train score : ', accuracy_score(pred, y_train))
print('test score : ', accuracy_score(pred_test, y_test))

# train score :  0.9738714285714286
# test score :  0.9695
