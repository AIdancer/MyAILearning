```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance


stocks = index_components("000905.XSHG")

df = get_factor(stocks,  ["market_cap", "pe_ratio_ttm", "pb_ratio_ttm", 
                                        "inc_revenue_ttm", "return_on_equity_ttm", "gross_profit_margin_ttm"], date="2023-11-01")

pe_mid = df["pe_ratio_ttm"].median()
print("pe_mid:{:.3f}".format(pe_mid))

pb_mid = df["pb_ratio_ttm"].median()
print("pb_mid:{:.3f}".format(pb_mid))

inc_mid = df["inc_revenue_ttm"].median()
print("inc_mid:{:.3f}".format(inc_mid))

roe_mid = df["return_on_equity_ttm"].median()
print("roe_mid:{:.3f}".format(roe_mid))

gross_mid = df["gross_profit_margin_ttm"].median()
print("gross_mid:{:.3f}".format(gross_mid))

df.fillna(value=gross_mid, inplace=True)
df.head(5)



prices = get_price(stocks, start_date="2023-11-01", end_date="2023-12-01", frequency="1d", fields=["close"])
ic_price = get_price("000905.XSHG", start_date="2023-11-01", end_date="2023-12-01", frequency="1d", fields=["close"])


bench_ret = ic_price["close"][-1] / ic_price["close"][0] - 1.0
rk = df.rank()
n = df.shape[0]
index = df.index.values
print(n)

X, y = [], []

for i in range(n):
    code = index[i][0]
    cap, pe, pb, inc, roe, gross = df.iloc[i,:]
    p = prices.loc[code]["close"]
    ret = p[-1] / p[0] - 1.0
    f_cap = 1.0 - rk["market_cap"][i] / 500
    f_pe = pe_mid - pe
    f_pb = pb_mid - pb
    f_inc = inc - inc_mid
    f_roe = roe - roe_mid
    f_gross = gross - gross_mid
    row = [f_cap, f_pe, f_pb, f_inc, f_roe, f_gross]
    X.append(row)
    y.append(ret-bench_ret)
    

mean = np.mean(X, axis=0)
print(mean)
X_ = np.array(X)
y_ = np.array(y).reshape(500, -1)
print(X_.shape, y_.shape)

model = XGBRegressor(n_estimators=100, max_depth=6, subsample=0.7)
model.fit(X_, y_, eval_set=[(X_, y_)], eval_metric='rmse', verbose=True)

pred = model.predict(X_)

print(pred.shape)
print(pred[0:10])
scores = []
for i in range(n):
    scores.append((index[i][0], pred[i]))
scores.sort(key=lambda x: x[1], reverse=True)
print(scores[0:10])

targets = [val[0] for val in scores[0:100]]

print(targets)

plot_importance(model)
```
