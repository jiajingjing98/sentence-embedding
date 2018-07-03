
W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"

from encoder import Encoder
import torch
import numpy as np
import torch.nn as nn
from torch import optim

f = Encoder()
sentences = ['Memories of childhood are unforgettable.', 'I was four years old when my grandfather died.',
             'I clearly remember how everybody in the house was weeping.']
f.set_w2v_path(W2V_PATH)
f.build_vocab(sentences, True)
embeddings = f.encode(sentences, bsize=400, tokenize=False, verbose=True)

print(embeddings)
scores = np.matmul(embeddings,embeddings)
print(scores)
scores_sum = np.sum(scores, axis=1, keepdims=True)

scores = scores/scores_sum
np.fill_diagonal(scores, 0)

print(scores)

targets = np.zeros((400,400))
context_size = 1

ctxt_sent_pos = list(range(-context_size, context_size+1))
ctxt_sent_pos.remove(0)
for ctxt in ctxt_sent_pos:
    targets += np.eye(400, k=ctxt)
targets_sum = np.sum(targets,axis=1, keepdims=True)
targets = targets / targets_sum
print(targets)




weight = torch.FloatTensor(2).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

optim_fn = optim.Adam
optimizer = optim_fn(f.parameters(), lr=0.0005)

