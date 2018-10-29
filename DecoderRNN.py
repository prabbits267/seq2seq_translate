import torch
from torch import nn
from torch.autograd import Variable


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, input, hidden):
        embedding = self.embedding(input)
        out, (hidden_state, cell_state) = self.lstm(embedding, hidden)
        out = self.softmax(self.out(out))
        return out, (hidden_state, cell_state)

    def init_hidden(self):
        hidden_state = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))
        cell_state = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))
        return hidden_state, cell_state