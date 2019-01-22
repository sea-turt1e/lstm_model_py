import torch
import torch.optim as optim
import lm as L
# 作成したlm.py
import dataset
# 作成したdataset.py
maxEpoch= 10 # 最大学習回数
dictPath= '../model/dicts.pickle'
modelPath= '../model/lm_%d.model'
trainDataPath= '../data/ptb.train.txt'

ds = dataset.Dataset(trainDataPath)
ds.save(dictPath)

lm = L.LM(vocSize=len(ds.word2id))
lm.train()

opt = optim.Adam(lm.parameters())

for ep in range(maxEpoch):
    accLoss = 0
    for idLine in ds.idData:
        opt.zero_grad()
        loss = lm.getLoss(idLine)
        loss.backward()
        opt.step()
        accLoss += loss
    
    print('epoch:', ep)
    print('loss:', loss)
    torch.save(lm, '../model/lm_%d.model'%(ep+1))

print('FINISH TRAINING')