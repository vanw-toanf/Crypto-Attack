# models.py
import torch
import torch.nn as nn
import config

class Generator(nn.Module):
    def __init__(self, vocab_size):
        super(Generator, self).__init__()
        self.seq_length = config.SEQ_LENGTH
        
        self.dense_in = nn.Linear(config.LATENT_DIM, 128 * config.SEQ_LENGTH)
        # SỬA Ở ĐÂY: Dùng LeakyReLU để ổn định hơn
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.lstm = nn.LSTM(128, 128, batch_first=True, num_layers=2)
        self.layernorm = nn.LayerNorm(128)
        self.dense_out = nn.Linear(128, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, z):
        x = self.dense_in(z)
        x = self.leaky_relu(x) # Áp dụng LeakyReLU
        x = x.view(-1, self.seq_length, 128)
        
        lstm_out, _ = self.lstm(x)
        lstm_out_norm = self.layernorm(lstm_out)
        
        x = self.dense_out(lstm_out_norm)
        x = self.softmax(x)
        return x

class Critic(nn.Module):
    def __init__(self, vocab_size):
        super(Critic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        # SỬA Ở ĐÂY: Giảm Critic xuống còn 1 lớp LSTM
        self.lstm = nn.LSTM(config.EMBEDDING_DIM, 128, batch_first=True, num_layers=1)
        self.layernorm = nn.LayerNorm(128)
        self.dense = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        return self.forward_from_embeddings(x)

    def forward_from_embeddings(self, x_embedded):
        x, _ = self.lstm(x_embedded)
        last_step_out = x[:, -1, :]
        last_step_out_norm = self.layernorm(last_step_out)
        x = self.dense(last_step_out_norm)
        return x