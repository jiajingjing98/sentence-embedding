
W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"

from encoder import Encoder
from givenencoder import Gr
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

p = Gr()
q = Encoder()
sentences = ['Memories of childhood are unforgettable.', 'I was four years old when my grandfather died.',
             'I clearly remember how everybody in the house was weeping.']

p.set_w2v_path(W2V_PATH)
q.set_w2v_path(W2V_PATH)
p.build_vocab(sentences)
q.build_vocab(sentences)

a = p.encode(sentences, 400, True, False)
b = q.encode(sentences, 400, True, False)

print(a, b)