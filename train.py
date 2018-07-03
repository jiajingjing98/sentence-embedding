
W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"

from encoder import Encoder
import torch
import numpy as np
import torch.nn as nn
from torch import optim

f = Encoder()
sentences = ['He had a knife in his hand', 'I hit his hand with a brick', 'Today is sunny']
f.set_w2v_path(W2V_PATH)
f.build_vocab(sentences, True)
embeddings = f.encode(sentences, bsize=300, tokenize=False, verbose=True)
c1 = np.inner(embeddings[0], embeddings[1])
c2 = np.inner(embeddings[0], embeddings[2])
print(c1)
print(c2)

weight = torch.FloatTensor(2).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

optim_fn = optim.Adam
optimizer = optim_fn(f.parameters(), lr=0.0005)

for param in f.parameters():
    print(param)