import torch
from torch import nn
from torch.autograd import Variable

from DecoderRNN import DecoderRNN
from EncoderRNN import EncoderRNN
from PrepareData import *
data = PrepareData()

SOS = 0
use_cuda = torch.cuda.is_available()

class Translate():
    def __init__(self):
        self.data = PrepareData()
        self.lang_1 = data.lang_1
        self.lang_2 = data.lang_2
        self.char2index = data.char2index
        self.index2char = data.index2char

        self.input_size = 100
        self.hidden_size = 64
        self.output_size = 100
        self.learning_rate = 0.01
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_cuda else 'cpu'

        self.encoder = EncoderRNN(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder = DecoderRNN(output_size=self.output_size, hidden_size=self.hidden_size)

        if use_cuda:
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

        self.loss_function = nn.NLLLoss()

    def create_variable(self, tensor):
        return Variable(tensor.to(self.device))

    def convert2ind(self, sent):
        inds = [self.char2index[w] for w in sent]
        return self.create_variable(torch.LongTensor([inds]))

    def train(self, input_tensor, target_tensor):
        loss = 0
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        target_length = target_tensor.size(0)

        encoder_output, encoder_hidden = self.encoder(input_tensor)

        decoder_hidden = encoder_hidden

        decoder_input = torch.LongTensor([[SOS]]).to(self.device)

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            loss += self.loss_function(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

        self.loss_function.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()


trans = Translate()
target_tensor = trans.convert2ind("J'ai compris.")
input_tensor = trans.convert2ind("I get it.")

input, _ = trans.encoder(input_tensor)
output = trans.decoder(target_tensor,_)

print(output)






