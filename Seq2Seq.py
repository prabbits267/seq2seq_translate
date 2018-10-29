import torch
from torch import nn
from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from DecoderRNN import DecoderRNN
from EncoderRNN import EncoderRNN
from seq2seq_dataset import Seq2SeqDataset


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        full_text, tokens = self.read_data()
        dataset = Seq2SeqDataset()
        self.dataloader = DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=True)
        self.SOS = '_'
        self.char2index = {}
        self.char2index[self.SOS] = 0
        token_idx = {w:i for i, w in enumerate(tokens)}
        self.char2index.update(token_idx)
        self.index2char = {w[1]:w[0] for w in self.char2index.items()}

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

        self.input_size = len(self.char2index)
        self.hidden_size = 64
        self.n_layers = 1
        self.learning_rate = 0.01
        self.n_epoch = 1000

        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = DecoderRNN(self.hidden_size, len(tokens), self.n_layers)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def forward(self, input, target):
        input_tensor = self.convert_to_tensor(input)
        target_tensor = self.convert_to_tensor(target)
        decoder_input = target_tensor

        # use teacher forcing. use sheduled sampling. leave it behind
        encoder_hidden = self.encoder(input_tensor)
        decoder_output, (hidden_state, cell_state) = self.decoder(decoder_input, encoder_hidden)

        return decoder_output


    # set configuration for single sentence
    def convert_to_tensor(self, sent):
        sent = self.SOS + sent[0] + self.SOS
        seq = self.create_variable(torch.LongTensor([[self.char2index[w] for w in list(sent)]]))
        return seq

    def create_variable(self, seq):
        return seq.to(self.device)

    def read_data(self):
        with open('data/data.txt', 'rt', encoding='utf-8') as file_reader:
            full_text = file_reader.read()
        tokens = sorted(set(full_text))
        return full_text, tokens

    def train(self, model, optimizer, loss_func):
        epoch_loss = 0
        optimizer.zero_grad()
        for i, (x_data, y_data) in enumerate(self.dataloader):
            output = model(x_data, y_data)
            output = output.squeeze(0)
            target_tensor = self.convert_to_tensor(y_data).squeeze(0)
            loss = loss_func(output, target_tensor)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        return epoch_loss

    def epoch_train(self):
        model = Seq2Seq()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_func = NLLLoss()

        for i in range(self.n_epoch):
            loss = self.train(model, optimizer, loss_func)
            a = self.eval('G')
            print(loss)

    def eval(self, sent):
        seq = self.convert_to_tensor(sent)
        decoder_hidden = self.encoder(seq)
        decoder_input = torch.LongTensor([[self.char2index[self.SOS]]]).to(self.device)
        output_seq, (hidden_state, cell_state) = self.decoder(decoder_input, decoder_hidden)

        for i in range(13):
            output_seq, (hidden_state, cell_state) = self.decoder(decoder_input, decoder_hidden)
            decoder_input = torch.max(output_seq, 2)[1]
            print(decoder_input)
        return output_seq, hidden_state, cell_state





seq = Seq2Seq()
seq.epoch_train()


