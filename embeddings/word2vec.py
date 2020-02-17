import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        out = torch.sum(self.embeddings(inputs), dim=0)
        out = self.linear(out)
        out = F.log_softmax(out, dim=0)
        return out

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = self.linear(out)
        out = F.log_softmax(out, dim=0)
        return out