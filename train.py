
W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"

from encoder import Encoder
from givenencoder import Gr
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from nltk.tokenize import sent_tokenize as ST


f = Encoder()


sentences = ['The Moon is filled wit craters.', 'It has no light of its own.', 'It gets its light from the Sun.']
g = open('/home/jingjing/Desktop/big.txt', 'r')
data = g.read()
splat = data.split("\n\n")
data = []
for p in splat:
    data.append(ST(p))



f.set_w2v_path(W2V_PATH)
f.build_vocab(sentences)

def make_target(context_size, dim):
    targets = np.zeros((dim, dim))
    ctxt_sent_pos = list(range(-context_size, context_size+1))
    ctxt_sent_pos.remove(0)
    for ctxt in ctxt_sent_pos:
        targets += np.eye(dim, k=ctxt)
    targets_sum = np.sum(targets,axis=1, keepdims=True)
    targets = targets / targets_sum
    targets = torch.from_numpy(targets)
    return targets

targets = make_target(1, len(sentences))

def loss_fn(pred, target):
    mask = 1 - torch.diag(torch.ones(pred.size(1)))
    npred = pred * mask
    s_pred = F.softmax(npred, -1)
    losses = F.binary_cross_entropy_with_logits(s_pred, target, size_average=False)
    return losses

optimizer = optim.Adam(f.parameters(), lr=0.0005)

for ins in data[:13]:
    optimizer.zero_grad()
    f.build_vocab(ins, True)
    a = f.encode(ins, 400, True, False)
    targets = make_target(1, len(ins))
    loss = loss_fn(a, targets.float())
    print(loss)
    loss.backward()
    optimizer.step()

optimizer.zero_grad()
d = data[13]
f.build_vocab(d, True)
print(len(d))
targets = make_target(1, len(data[13]))