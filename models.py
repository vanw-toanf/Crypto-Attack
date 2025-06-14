import torch
import torch.nn as nn
import config

class CharRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            config.NUM_LAYERS,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(config.HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden