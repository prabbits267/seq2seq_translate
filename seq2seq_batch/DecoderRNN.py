from torch import nn


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, self.input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

