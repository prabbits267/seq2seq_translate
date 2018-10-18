import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers=1, batch_size=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.lstm = nn.LSTM(
            input_size= self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, input_len):
        batch_size, seq_len, _ = input_seq.size()
        packed_input =  pack_padded_sequence(input_seq, input_len, batch_first=True)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input)
        unpacked_output, unpacked_len = pad_packed_sequence(packed_output)
        output = unpacked_output.view(batch_size * seq_len, -1)
        output = self.out(output)
        output = self.softmax(output)
        output = output.view(batch_size, seq_len, _)
        return output, (hidden_state, cell_state)

    def init_hidden(self):
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        cell_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        return hidden_state, cell_state





