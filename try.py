import torch
import numpy as np
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = [("我 不 知 道".split(), "CHINESE"),
        ("Give it to me".split(), "ENGLISH"),
        ("把 它 给 我".split(), "CHINESE"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("失 去 了".split(), "CHINESE"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2



class Bow(nn.Module):
	def __init__(self, num_labels, vocab_size):
		super(Bow, self).__init__()
		self.linear = nn.Linear(vocab_size, num_labels)

	def forward(self, bow_vec):
		return F.log_softmax(self.linear(bow_vec), dim=1)

def make_bow_vector(sentence, word_to_ix):
	vec = torch.zeros(len(word_to_ix))
	for word in sentence:
		vec[word_to_ix[word]] += 1
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])


model = Bow(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
    print(param.grad)

with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)



label_to_ix = {"CHINESE": 0, "ENGLISH": 1}

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)



loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        model.zero_grad()

        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)
        

        log_probs = model(bow_vec)

        loss = loss_function(log_probs, target)
        
        loss.backward()
        optimizer.step()
        for param in model.parameters():
            print(param.grad)

print("testing")
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        lst = log_probs.tolist()[0]
        if lst[0] > lst[1]:
        	print(" ".join(instance) + " is Chinese.")
        else:
        	print(" ".join(instance) + " is English.")








