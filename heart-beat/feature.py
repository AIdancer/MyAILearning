import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\86133\liqiaz\data\heart-beat\train.csv')
df.head(5)

X, y = [], []
n = df.shape[0]
for i in range(n):
    signal_str, label = df.loc[i, 'heartbeat_signals'], int(df.loc[i, 'label'])
    signals = np.array(list(map(float, signal_str.split(','))))
    X.append(signals)
    y.append(label)
X = np.array(X)
y = np.array(y)
feature = np.hstack([X, y.reshape(-1,1)])
np.save('feature.npy', feature)
