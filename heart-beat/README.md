### 心跳信号分类预测
<p>&emsp;&emsp;数据来源是阿里的天池大数据竞赛 https://tianchi.aliyun.com/competition/gameList/activeList</p>

### 模型总结

  1. xgboost : 仅次于DNN+xgboost
  2. 神经网络 : 神经网络得分略低于xgboost，总体准确率相当。
  3. SVM : 效果比前两个准确率总体低2%左右。
  4. DNN + xgboost : 目前取得最好得分的模型
  5. DNN + ResNet + xgboost : 得分效果略差于模型4，原因可能是DNN本身就已经把特征处理的非常好了，残差网络的添加反而造成了过拟合。
  
  
