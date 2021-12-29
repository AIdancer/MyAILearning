```Python
#丐版word2vec

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data

dtype = torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dtype)
print(device)

sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
vocab = list(set(word_sequence))
word2idx = {w : i for i, w in enumerate(vocab)}
idx2word = {i : w for i, w in enumerate(vocab)}
print(word2idx)
print(idx2word)

batch_size = 8
embedding_size = 2
C = 2
voc_size = len(vocab)

skip_grams = []
for idx in range(C, len(word_sequence) - C):
    center = word2idx[word_sequence[idx]]
    context_idx = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
    context = [word2idx[word_sequence[i]] for i in context_idx]
    for w in context:
        skip_grams.append([center, w])
print(skip_grams)

def make_data(skip_grams):
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        input_data.append(np.eye(voc_size)[skip_grams[i][0]])
        output_data.append(skip_grams[i][1])
    return input_data, output_data
    
input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)

class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(voc_size, embedding_size).type(dtype))
        self.V = nn.Parameter(torch.randn(embedding_size, voc_size).type(dtype))
    
    def forward(self, X):
        hidden_layer = torch.matmul(X, self.W)
        output_layer = torch.matmul(hidden_layer, self.V)
        return output_layer

model = Word2Vec().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(2000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if (epoch + 1) % 200 == 0:
            print(epoch + 1, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
for i, label in enumerate(vocab):
    W, WT = model.parameters()
    print(W[i])
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    
```
