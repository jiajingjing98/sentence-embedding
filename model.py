

W2V_PATH = "/home/jingjing/Desktop/InferSent-master/dataset/GloVe/glove.840B.300d.txt"


import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn


class QTModel(nn.Module):
    def __init__(self):
        super(QTModel, self).__init__()
        self.f = Encoder()
        self.g = Encoder()

    def forward(self, sentences, bsize, tokenize=True, verbose=False):
        e1 = self.f.encode(sentences, bsize, tokenize, verbose)
        e2 = self.g.encode(sentences, bsize, tokenize, verbose)
        embs = torch.mm(e1, torch.transpose(e2, 0, 1))
        return embs

    def build_vocab(self, sentences, tokenize=True):
        self.f.build_vocab(sentences, tokenize)
        self.g.build_vocab(sentences, tokenize)

    def set_w2v_path(self, w2v_path):
        self.f.w2v_path = w2v_path
        self.g.w2v_path = w2v_path


class Encoder(nn.Module):

    def __init__(self):
        #torch.manual_seed(0)
        super(Encoder, self).__init__()
        self.bsize = 400
        self.word_emb_dim = 300
        self.enc_gru_dim = 1200
        self.version = 1
        self.word_vec = {}

        self.enc_gru = nn.GRU(self.word_emb_dim, self.enc_gru_dim, 1,
                                bidirectional=True)
        self.init_gru = Variable(torch.FloatTensor(2, self.bsize,
                                                    self.enc_gru_dim).zero_())

        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False




    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        bsize = sent.size(1)

        self.init_gru = self.init_gru if bsize == self.init_gru.size(1) else \
            Variable(torch.FloatTensor(2, bsize, self.enc_gru_dim).zero_())

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.LongTensor(idx_sort)))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        _, hn = self.enc_gru(sent_packed, self.init_gru)
        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)

        emb = emb.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        return emb


    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break

        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = torch.FloatTensor()
        for stidx in range(0, len(sentences), bsize):
            with torch.no_grad():
                batch = Variable(self.get_batch(
                        sentences[stidx:stidx + bsize]))
            batch = self.forward(
                (batch, lengths[stidx:stidx + bsize]))
            embeddings = torch.cat((embeddings, batch))

        # unsort
        idx_unsort = torch.from_numpy(np.argsort(idx_sort))
        embeddings = embeddings.index_select(0, Variable(idx_unsort))

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                     'cpu', bsize))


        #embeddings = torch.mm(embeddings, torch.transpose(embeddings, 0, 1))

        return embeddings


