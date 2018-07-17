
W2V_PATH = "./glove.840B.300d.txt"

from model import QTModel
#from givenencoder import Gr
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from nltk.tokenize import sent_tokenize as ST
import nltk

nltk.download('punkt')

f = QTModel()


sentences = ['The Moon is filled wit craters.', 'It has no light of its own.', 'It gets its light from the Sun.']
with open('./try.txt', 'r') as in_file:
    text = in_file.read()
    sents = ST(text)

data = sents


f.set_w2v_path(W2V_PATH)


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

def nln(pred, target):
    mask = 1 - torch.diag(torch.ones(pred.size(1)))
    npred = pred * mask
    s_pred = F.softmax(npred, -1)
    product = targets.float() * torch.log(s_pred)
    loss = torch.mean(-torch.sum(product,1))
    return loss

with torch.no_grad():
    f.build_vocab(sentences, True)
    embs = f(sentences, 400, True, True)
    #print(embs)
    targets = make_target(1, len(sentences))
    loss = nln(embs, targets.float())
    
    print("loss before training: ", loss)

optimizer = optim.Adam(f.parameters(), lr=0.0005)
torch.nn.utils.clip_grad_norm_(f.parameters(), 5)

for epoch in range(1):
    optimizer.zero_grad()
    if epoch == 0:
        f.build_vocab(data, True)
    targets = make_target(1, len(data))
    scores = f(data, 400)
    loss = nln(scores, targets.float())
    print(loss)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    #f.build_vocab(sentences, True)
    embeddings = f(sentences, 400)
    targets = make_target(1, len(sentences))
    loss = nln(embeddings, targets.float())
    print("loss after training: ", loss)