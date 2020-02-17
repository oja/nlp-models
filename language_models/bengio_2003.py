import torch
import torch.nn as nn
import torch.nn.functional as F

class Bengio2003(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, ngram_size):
        super(Bengio2003, self).__init__()
        self.C = nn.Embedding(vocab_size, embedding_dim)
        self.g = nn.Linear(embedding_dim * context_size, vocab_size)
        self.softmax = nn.Softmax()
        self.ngram_size = ngram_size
        
    def forward(self, inputs):
        out = self.C(inputs)
        out = torch.cat(tuple([out[i] for i in range(0, self.ngram_size)]), 1)
        out = self.g(torch.tanh(out))
        return F.log_softmax(out, dim=1)