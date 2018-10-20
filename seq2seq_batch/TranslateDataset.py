from torch.utils.data import Dataset, DataLoader


class TranslateDataset(Dataset):
    def __init__(self):
        self.path = '../data/data.txt'
        self.lang_1, self.lang_2, self.full_text, self.len = self.read_file()
        self.char2index, self.index2char = self.generate_dict()
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.lang_1[index], self.lang_2[index]

    def read_file(self):
        with open(self.path, 'rt', encoding='utf-8') as file_reader:
            raw_text = file_reader.read()

        full_text = raw_text.splitlines()
        lang_1, lang_2 = list(), list()

        for text in full_text:
            pair = text.split('\t')
            if(len(pair) == 2):
                lang_1.append(pair[0])
                lang_2.append(pair[1])
        return lang_1, lang_2, raw_text, len(lang_1)

    def generate_dict(self):
        char_list = list(sorted(set(self.full_text)))
        char2index = {w:i for i,w in enumerate(char_list)}
        index2char = {w[1]:w[0] for w in char2index.items()}
        return char2index, index2char