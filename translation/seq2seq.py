import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2)
        
    def forward(self, inputs):
        out = self.embedding(inputs)
        out, (hidden, cell) = self.lstm(out)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2)
        self.prediction_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, inputs, hidden, cell):
        out = inputs.unsqueeze(0)
        out = self.embedding(out)
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
        prediction = self.prediction_layer(out.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):
        batch_size = target.shape[1]
        max_length = target.shape[0]
        target_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(max_length, batch_size, target_vocab_size)
        hidden, cell = self.encoder(source)
        inputs = target[0,:]
        
        for t in range(1, max_length):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < 0.5
            top1 = output.max(1)[1]
            inputs = (target[t] if teacher_force else top1)
            
        return outputs