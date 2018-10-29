import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DecoderRNN import DecoderRNN
from EncoderRNN import EncoderRNN
from PrepareData import *
from seq2seq_dataset import Seq2SeqDataset

data = PrepareData()

SOS = 0
use_cuda = torch.cuda.is_available()

class Translate():
    def __init__(self):
        self.data = PrepareData()
        self.dataset = Seq2SeqDataset()
        self.data_loader = DataLoader(dataset=self.dataset,
                                      batch_size=1,
                                      shuffle=True)
        self.lang_1 = data.lang_1
        self.lang_2 = data.lang_2
        self.char2index = data.char2index
        self.index2char = data.index2char

        self.input_size = 100
        self.hidden_size = 64
        self.output_size = 100
        self.learning_rate = 0.01
        self.num_epoch = 500
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_cuda else 'cpu'

        self.encoder = EncoderRNN(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder = DecoderRNN(output_size=self.output_size, hidden_size=self.hidden_size)

        if use_cuda:
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

        self.loss_function = nn.NLLLoss()

    def create_variable(self, tensor):
        return Variable(tensor.to(self.device))

    def convert2ind(self, sent):
        inds = [self.char2index[w] for w in sent]
        return self.create_variable(torch.LongTensor([inds]))

    # train on single sentence -- return loss + decoder_outputs
    def step(self, input_sent, target_tensor):


        input_tensor = self.convert2ind(list(input_sent[0]))
        target_tensor = self.convert2ind(list(target_tensor[0]))
        target_tensor = target_tensor.squeeze(0)
        clip = 5.0
        loss = 0

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size()[1]
        target_length = target_tensor.size()[0]

        encoder_hidden = self.encoder(input_tensor)

        decoder_input = self.create_variable(torch.LongTensor([[SOS]])).to(self.device)
        decoder_hidden = encoder_hidden

        decoder_outputs = []

        for i in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            decoder_input = torch.max(decoder_output, 1)[1].unsqueeze(0)
            target = target_tensor[i].unsqueeze(0)
            loss += self.loss_function(decoder_output, target)
            decoder_outputs.append(decoder_input)

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        return loss.data[0], decoder_outputs

    def train(self):
        for i in range(self.num_epoch):
            for i, (x_data, y_data) in enumerate(self.data_loader):
                loss, result = self.step(x_data, y_data)
            print(loss)



trans = Translate()
trans.train()







