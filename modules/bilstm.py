import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, n_class, n_hidden, device):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, n_hidden)
        self.n_hidden = n_hidden
        self.device = device

    def forward(self, X):
        batch_size = X.shape[0]
        input = X.transpose(0, 1)
        hidden_state = torch.randn(1 * 2, batch_size, self.n_hidden).to(self.device)
        cell_state = torch.randn(1 * 2, batch_size, self.n_hidden).to(self.device)
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = self.fc(outputs)
        return outputs
