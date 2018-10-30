import torch
from torch import nn
from EncoderRNN import *
from attn import Attn


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoder, self).__init__()

        self.attn_model = Attn('general', 64)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_p
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, encoder_outputs):
        embedding = self.embedding(input)
        # (1, 1, hidden) (1, time_step, hidden_size)
        attn_weights = self.attn(hidden, encoder_outputs)

        # context weight = context weight * encoder output
        # attention weight : [max_len] | encoder_output: [1, max_len, hidden_size]
        encoder_outputs = encoder_outputs.squeeze(0)
        context = torch.mm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedding, context), 0)
        output, hidden = self.lstm(rnn_input, hidden)
        output = self.out(self.softmax(output))
        return output, hidden





