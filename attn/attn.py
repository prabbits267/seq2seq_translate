import torch

from torch import nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))


    # hidden (1,1,hidden)
    # encoder_output (1, max_len, hidden)
    def forward(self, hidden_state, encoder_output):
        hidden_state = hidden_state.squeeze(0)
        encoder_output = encoder_output.squeeze(0)
        max_len = encoder_output.size(0)
        attn_energy = torch.zeros([1, max_len])
        for i in range(max_len):
            # hidden[0] (hidden) encoder_output[i]: (hidden)
            attn_energy[:, i] = self.score(hidden_state[0], encoder_output[i])
        return F.softmax(attn_energy, dim=1)

    def score(self, hidden_state, encoder_output):
        if self.method == 'dot':
            return hidden_state.dot(encoder_output)
        elif self.method == 'general':
            attn_energy = self.attn(encoder_output)
            return hidden_state.dot(attn_energy)
        elif self.method == 'concat':
            attn_energy = self.attn(torch.cat((hidden_state, encoder_output), 0))
            return self.v.dot(attn_energy)

attn = Attn('dot', 64)
a = torch.randn([1,1,64])
b = torch.randn([1,10,64])
c = attn(a,b)
print(c)