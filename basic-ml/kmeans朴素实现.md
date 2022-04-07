```python
import random
import numpy as np
import matplotlib.pyplot as plt


# 生成聚类中心
def get_centers(k):
    return np.random.rand(k, 2) * 30.0


# 根据中心生成样本点
def generate_samples(centers, n):
    points = []
    for i in range(centers.shape[0]):
        noise = np.random.randn(n, 2)
        points.append(centers[i] + noise)
    points = np.concatenate(points)
    return points

# kmeans-聚类算法
def kmeans_cluster(samples, k):
    n = samples.shape[0]
    index = list(range(n))
    random.shuffle(index)
    c = index[0:k]
    # 随机确定聚类初始点
    centers = samples[c]
    iteration_limit = 20

    # 迭代
    for t in range(iteration_limit):
        dist = {}
        # 计算聚类中心距离所有样本点距离
        for i in range(k):
            temp = centers[i] - samples
            dist[i] = np.sqrt(np.sum(temp * temp, axis=1))
        # 寻找每个样本点距离中心的距离，并归为距离最小的中心类别
        labels = np.zeros(n, dtype=np.int)
        mmin = dist[0]
        for i in range(1, k):
            for j in range(n):
                if dist[i][j] < mmin[j]:
                    mmin[j] = dist[i][j]
                    labels[j] = i
        # 生成新的聚类中心
        for i in range(k):
            tc = np.zeros(2, dtype=np.float)
            cnt = 0
            for j in range(n):
                if labels[j] == i:
                    tc = tc + samples[j]
                    cnt += 1
            centers[i] = tc / cnt
        print('iteration : ', t+1)
        print(centers)


if __name__ == '__main__':
    centers = get_centers(3)
    samples = generate_samples(centers, 100)
    print('origin centers : ')
    print(centers)
    print('- - - - - - - - -')
    kmeans_cluster(samples, 3)
    x = samples[:, 0]
    y = samples[:, 1]
    plt.scatter(x, y)
    plt.show()
```
