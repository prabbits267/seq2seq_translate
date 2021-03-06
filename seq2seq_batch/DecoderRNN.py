import torch

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers=1, batch_size=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, self.input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden_state):
        batch_size, seq_len, _ = input_seq.size()
        output, (hidden_state, cell_state) = self.lstm(input_seq, hidden_state)

        output = self.softmax(output)
        output = output.view(batch_size, seq_len, _)
        return output, (hidden_state, cell_state)

    def init_hidden(self):
        hidden_state = Variable(torch.zeros([self.n_layers, self.batch_size, self.hidden_size]))
        cell_state = Variable(torch.zeros([self.n_layers, self.batch_size, self.hidden_size]))
        return hidden_state, cell_state

