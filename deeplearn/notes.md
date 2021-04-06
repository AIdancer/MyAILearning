### 最佳网络层数
<p>&emsp;&emsp;在多个分类场景中，在配置网络层数时发现，简单的Linear+ReLU+Softmax层的网络，并非层数越多越好，隐藏层2层的效果反而大幅好于隐藏层3层以上的效果。原因可能是深层网络的梯度传导学习率不合适。</p>
<p>&emsp;&emsp;在使用backblaze磁盘故障预测中，以上现象尤其明显。尽管在训练集中，中间隐藏层使用128个神经元的Linear + ReLU + Linear产生的效果可以好于SVM，但是在测试集上则表现不佳，而且增加层数之后出现无法
  收敛的现象，后续会使用残差网络进行优化。</p>

### 变分自编码器VAE资料
  1. https://blog.csdn.net/deephub/article/details/107012873
  2. https://blog.csdn.net/weixin_36815313/article/details/107728274
