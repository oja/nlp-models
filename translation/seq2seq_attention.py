import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, decoder_hidden_dim):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
        self.fully_connected = nn.Linear(hidden_dim * 2, decoder_hidden_dim)
    
    def forward(self, inputs):
        out = self.embedding(inputs)
        out, hidden = self.gru(out)
        hidden = torch.tanh(self.fully_connected(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return out, hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, hidden_dim):
        super(Attention, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        
        self.attn = nn.Linear((encoder_hidden_dim * 2) + hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, hidden, outputs):
        batch_size = outputs.shape[1]
        source_len = outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, source_len, 1)
        outputs = outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, outputs), dim = 2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, hidden_dim, attention):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, hidden_dim)        
        self.out = nn.Linear((encoder_hidden_dim * 2) + hidden_dim + embedding_dim, output_dim)
        
    def forward(self, inputs, hidden, encoder_outputs):
        inputs = inputs.unsqueeze(0)
        embedded = self.embedding(inputs)
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)    
        weighted = weighted.permute(1, 0, 2)
        output, hidden = self.rnn(torch.cat((embedded, weighted), dim=2), hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim=1))        
        return output, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        encoder_outputs, hidden = self.encoder(src)
                
        output = trg[0,:]
        
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < 0.5
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs