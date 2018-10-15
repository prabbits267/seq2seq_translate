from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.lstm = nn.LSTM(
            input_size= self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True
        )

    def forward(self, input_seq, input_len):
        batch_size, seq_len, _ = input_seq.size()
        packed_input =  pack_padded_sequence(input_seq, input_len)
        output, (hidden_state, cell_state) = self.lstm(packed_input)
        unpacked_output, unpacked_len = pad_packed_sequence(output)

        return unpacked_output, (hidden_state, cell_state)





