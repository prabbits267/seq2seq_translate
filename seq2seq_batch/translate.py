from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from EncoderRNN import *
from DecoderRNN import *
from seq2seq_batch.TranslateDataset import TranslateDataset


class Translate():
    def __init__(self, input_size, hidden_size, n_layers, batch_size, learning_rate):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"

        dataset = TranslateDataset()
        self.dataloader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        self.char2index, self.index2char = dataset.char2index, dataset.index2char
        self.vocab_size = dataset.len

        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.vocab_size, self.n_layers, self.batch_size)
        self.decoder = DecoderRNN(self.input_size, self.hidden_size, self.vocab_size, self.n_layers, self.batch_size)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.loss = NLLLoss()

        self.encoder_optim = torch.optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        self.deoder_optim = torch.optim.SGD(self.encoder.parameters(), lr=self.learning_rate)


    def create_variable(self, tensor):
        return tensor.to(self.device)

    def sent_to_seq(self, input):
        return torch.Tensor([self.char2index[w] for w in input])

    def create_batch(self, input, target):
        input_seq = [list(w) for w in input]
        output_seq = [list(w) for w in target]

        seq_pair = sorted(zip(input_seq, output_seq), key=lambda p:len(p[0]), reverse=True)
        input_seq, target_seq = zip(*seq_pair)

        input_seq, input_len = self.pad_sequence(input_seq)
        target_seq, target_len = self.pad_sequence(target_seq)

        return self.create_variable(input_seq), self.create_variable(target_seq), self.create_variable(input_len)

    def create_variable(self, tensor):
        return tensor.to(self.device)

    def pad_sequence(self, input_list):
        max_len = len(input_list[0])
        batch_size = len(input_list)

        pad_tensor = torch.zeros(batch_size, max_len)
        input_len = list()

        for i, chr in enumerate(input_list):
            input_len.append(len(chr))
            pad_tensor[i, :len(chr)] = self.sent_to_seq(chr)

        return pad_tensor, input_len




trans = Translate(100, 6, 1, 5, 0.01)
encoder = trans.encoder

for i, (lang_1, lang_2) in enumerate(trans.dataloader):
    a, b, _ = trans.create_batch(lang_1, lang_2)
    print(a)
    print(b)




