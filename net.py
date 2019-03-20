import torch
import torch.nn as nn


class ActionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 100, 1) # LSTM(input_size, hidden_layer, LSTM_layers)
        # self.lstm = nn.LSTM(24, 512, 1)
        self.fc1 = nn.Linear(100, 100)
        self.sigmoid = nn.Softsign()
        self.fc2 = nn.Linear(100, 3)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        out = input.transpose(0, 1)
        out, (ht, ct) = self.lstm(out)
        out = self.fc1(ht[0])
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out
