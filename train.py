
W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"

from newencoder import Encoder
import torch
import numpy as np
import torch.nn as nn
from torch import optim

f = Encoder()
sentences = ['Memories of childhood are unforgettable.', 'I was four years old when my grandfather died.',
             'I clearly remember how everybody in the house was weeping.']

f.zero_grad()

f.set_w2v_path(W2V_PATH)
f.build_vocab(sentences, True)
scores = f(sentences, bsize=400, tokenize=False, verbose=True)
print(scores.requires_grad)



targets = np.zeros((3,3))
context_size = 1

ctxt_sent_pos = list(range(-context_size, context_size+1))
ctxt_sent_pos.remove(0)
for ctxt in ctxt_sent_pos:
    targets += np.eye(3, k=ctxt)
targets_sum = np.sum(targets,axis=1, keepdims=True)
targets = targets / targets_sum

targets = torch.from_numpy(targets)



def xentropy_cost(pred, target):

    logged = torch.log(pred)
    a = target.float()*logged
    print(a)
    cost = -torch.sum(a)
    print(cost)
    return cost

weight = torch.FloatTensor(3).fill_(1)
loss_fn = nn.functional.binary_cross_entropy_with_logits

optimizer = optim.Adam(f.parameters(), lr=0.0005)
loss = loss_fn(scores, targets.float(), weight=weight)
print(loss)
loss.backward()
optimizer.step()
