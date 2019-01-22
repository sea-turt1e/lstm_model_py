import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

embedSize = 100
lstmHid = 200
lstmDepth = 1

class LM(nn.Module): # nn.Moduleを継承
    def __init__(self, vocSize): # 語彙数を引数にする
        super(LM, self).__init__() # 親クラスの内容を読む
        # ここに必要なレイヤーを追加
        self.embed = nn.Embedding(vocSize, embedSize)
        self.lstm = nn.LSTM(input_size=embedSize, hidden_size=lstmHid, num_layers=1, dropout=0.5)
        self.linear = nn.Linear(lstmHid, vocSize)
        self.criterion = nn.CrossEntropyLoss() # ロス関数

    def forward(self, idLine):
        # embedding -> LSTM -> linear -> log_softmax
        ems = self.embed(torch.LongTensor(idLine))
        hid = (torch.zeros(lstmDepth, 1, lstmHid), 
                torch.zeros(lstmDepth, 1, lstmHid))
        ys, _ = self.lstm(ems.view(len(idLine), 1, -1), hid)
        ys = ys.view(len(idLine), -1)   # Linearで扱えるように変形
        zs = self.linear(ys)
        _ = F.log_softmax(zs, dim=1)
        return zs

    def getLoss(self, idLine):
        ys = self.forward(idLine[:-1])  # 入力
        ts = torch.LongTensor(idLine[1:])   # 教師
        loss = self.criterion(ys, ts)
        return loss

    def getSentenceLogProb(self, idLine):
        inp = idLine[:-1]
        zs = self.forward(inp) # log_softmaxの結果が返ってくる
        # i番目の確率分布における
        # i+1単語に対応する確率を取り，足し合わせる H =0
        for i in range(len(idLine)-1):
            H += -zs[i][idLine[i+1]].data[0]
        return H


if __name__ == '__main__':
    lm = LM(10)
    idLine = [0,1,2,3,0]
    # print(lm.getLoss([0,1,2,3,4,5,0]))

    # forwardを試す
    print(lm.forward(idLine))
    # getLossを試す
    print(lm.getLoss(idLine))
