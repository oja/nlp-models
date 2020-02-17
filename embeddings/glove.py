import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class GloVe(nn.Module):
    def __init__(self, coo_matrix, embedding_dim, vocab_size, batch_size):
        super(GloVe, self).__init__()
        
        self.coo_matrix = coo_matrix

        self.batch_size = batch_size
        
        # empirically found values from the paper
        self.alpha = 0.75
        self.cutoff = 100
        
        # laplacian smoothing
        self.coo_matrix += 1
        
        self.vocab_size = coo_matrix.shape[0]
        self.embedding_i = nn.Embedding(vocab_size, embedding_dim)
        self.bias_i = nn.Embedding(vocab_size, 1)
        
        self.embedding_j = nn.Embedding(vocab_size, embedding_dim)
        self.bias_j = nn.Embedding(vocab_size, 1)

    def forward(self, word_i, word_j):
        # word_i: [batch_size], long
        # word_j: [batch_size], long
        coos = Variable(torch.from_numpy(np.array([self.coo_matrix[word_i[x], word_j[x]] for x in range(self.batch_size)]))).to(torch.long) # [batch_size]
        weighting = Variable(torch.from_numpy(np.array([self._get_weighting(x) for x in coos]))).to(torch.long) # [batch_size]

        embed_i_out = self.embedding_i(word_i)
        bias_i_out = self.bias_i(word_i)
        embed_j_out = self.embedding_j(word_j)
        bias_j_out = self.bias_j(word_j)

        return embed_i_out, embed_j_out, bias_i_out, bias_j_out, coos, weighting
    
    def _get_weighting(self, occur):
        return 1.0 if occur > self.cutoff else (occur / self.cutoff) ** self.alpha