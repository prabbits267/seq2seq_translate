class PrepareData():
    def __init__(self):
        self.path = 'data/data.txt'
        self.lang_1, self.lang_2, self.tokens = self.read_data()
        self.char2index = {}
        self.index2char = {}
        self.char2index['_'] = 0
        self.index2char[0] = '_'
        self.char2index = {w:i for i, w in enumerate(self.tokens)}
        self.index2char = {w[1]:w[0] for w in self.char2index.items()}

    def read_data(self):
        with open(self.path, 'rt', encoding='utf-8') as file_reader:
            full_text = file_reader.read()
        lang_1 = list()
        lang_2 = list()
        tokens = sorted(set(full_text))
        for pair in full_text.splitlines():
            source, target = pair.split('\t')
            lang_1.append(source)
            lang_2.append(target)
        return lang_1, lang_2, tokens