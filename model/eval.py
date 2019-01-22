import torch
import lm as L
import dataset

epoch = 0

dictPath = '../model/ds.pickle'
testDataPath = '../data/ptb.test.txt'
modelPath = '../model/lm_%d.model'%epoch

ds = dataset.Dataset()

ds.load('../model/dicts.pickle')
ds.setData(testDataPath)
ds.setIdData()

lm = L.LM(len(ds.word2id))
lm = torch.load(modelPath % epoch)
lm.eval() # 評価モードにする

H = 0
W = 0
for idLine in ds.idData:
    H += lm.getSentenceLogProb(idLine)
    W += len(idLine) - 1
H /= W
print('entropy:', H)
print('PPL:', 2**H)