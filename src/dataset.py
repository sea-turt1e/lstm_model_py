from collections import defaultdict
import pickle

bos = '<BOS>'
eos = '<EOS>'

class Dataset:
    def __init__(self, dataPath=None):
        # path: data path
        self.data = []
        self.idData = []
        self.id2word = {}
        self.word2id = {}

        if dataPath:
            self.setData(dataPath)
            self.setDict()
            self.setIdData()

    def setData(self, path):
        # pathの内容を1行ずつ読み、分割とパディングを行う
        self.data = [[bos]+line.strip().split(' ')+[eos] for line in open(path)]

    def setDict(self):
        # 頻度辞書を作る
        wordCountDict = defaultdict(lambda:0)
        for line in self.data:
            for w in line:
                wordCountDict[w] += 1

        # 頻度辞書を降順にしてid化
        for k,v in reversed(sorted(wordCountDict.items(), key=lambda x:x[1])):
            self.word2id[k] = len(self.word2id)
            self.id2word[self.word2id[k]] = k

    def setIdData(self):
        # id辞書を使って単語ごとにidにしていく
        self.idData = [[self.word2id[w] for w in line] for line in self.data]

    def save(self, dictPath):
        pickle.dump((self.id2word, self.word2id), open(dictPath,'wb'))

    def load(self, dictPath):
        self.id2word, self.word2id = pickle.load(open(dictPath,'rb'))

if __name__ == '__main__':
    path = '../data/ptb.train.txt'
    ds = Dataset(path)

    # print(ds.id2word)