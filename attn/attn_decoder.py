import torch

from torch import nn
import torch.nn.functional as F
from attn.attn import Attn


class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, attn_method, output_size,n_layers=1):
        super(AttnDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.attn = Attn(attn_method, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    # input: tensor([<1>]) hidden: (1,1,64), encoder_outputs: (1, 10, 64)
    def forward(self, input, hidden, encoder_outputs):
        embedding = self.embeddings(input)
        output, (hidden_state, cell_state) = self.lstm(input, hidden)

        # size: ([time_step])
        attn_weights = self.attn(hidden_state, encoder_outputs)
        encoder_outputs = encoder_outputs.squeeze()
        context = attn_weights.mm(encoder_outputs)

        concat_input = torch.cat((context, hidden_state),1)
        concat_output = F.tanh(self.concat(concat_input))
        output = F.softmax(self.output(concat_output), dim=1)

        return output, hidden_state, cell_state

attn = AttnDecoder()