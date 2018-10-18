from torch.utils.data import DataLoader

from EncoderRNN import *
from DecoderRNN import *
from seq2seq_batch.TranslateDataset import TranslateDataset


class Translate():
    def __init__(self, input_size, hidden_size, vocab_size, n_layers, batch_size):
        self.batch_size = batch_size


        dataset = TranslateDataset()
        self.dataloader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)


        self.encoder = EncoderRNN()