W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"

from encoder import Encoder
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F


f = Encoder()
sentences = ['The Moon is filled wit craters.', 'It has no light of its own.', 'It gets its light from the Sun.']

data = [['Memories of childhood are unforgettable.', 'I was four years old when my grandfather died.',
             'I clearly remember how everybody in the house was weeping.'], ['Today is sunny', 'We should go out for a picnic.', 'Love the weather.']]

f.zero_grad()

f.set_w2v_path(W2V_PATH)

targets = np.zeros((3,3))
context_size = 1

ctxt_sent_pos = list(range(-context_size, context_size+1))
ctxt_sent_pos.remove(0)
for ctxt in ctxt_sent_pos:
    targets += np.eye(3, k=ctxt)
targets_sum = np.sum(targets,axis=1, keepdims=True)
targets = targets / targets_sum

print(targets)

targets = torch.from_numpy(targets)

optimizer = optim.Adam(f.parameters(), lr=0.0005)


def loss_fn(pred, target):
    m = nn.Softmax(-1)
    s_pred = m(pred)
    losses = F.binary_cross_entropy_with_logits(s_pred, target)
    return losses

with torch.no_grad():
    f.build_vocab(sentences, True)
    embeddings = f.encode(sentences, 3)
    loss = loss_fn(embeddings, targets.float())
    print("loss before training: ", loss)

for epoch in range(20):
    for instance in data:
        optimizer.zero_grad()
        if epoch==0:
            f.build_vocab(instance, True)
        scores = f.encode(instance, 3)
        loss = loss_fn(scores, targets.float())
        print(loss)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    f.build_vocab(sentences, True)
    embeddings = f.encode(sentences, 3)
    loss = loss_fn(embeddings, targets.float())
    print("loss after training: ", loss)