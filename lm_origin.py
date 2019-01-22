import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

embedSize = 100
lstmHid = 200
lstmDepth = 1

class LM(nn.Module):
    def __init__(self, vocSize):
        super(LM,self).__init__()
        self.embed = ## embedding初期化 ##
        self.lstm = ## lstm初期化 ##

        self.linear = ## linear初期化 ##
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, idLine):
        ems = ## embedding (len(idLine)*embedSize) ##
        hid = ## LSTMのhidStateをzero-initialize ##
              # ((lstmDepth, 1, hidSize),(lstmDepth, 1, hidSize))
        
        # LSTMはコピペでOK #
        ys, _ = self.lstm(ems.view(len(idLine),1,-1), hid)
        ys = ys.view(len(idLine),-1) # Linearで扱えるように変形
        
        zs = ## ysをlinearに通す ##
        zs_log_softmax = ## zsをlog_softmaxに通す ##
        return zs   
    # getLossはコピペでOK
    def getLoss(self, idLine):
        zs = self.forward(idLine[:-1])
        ts = Variable(torch.LongTensor(idLine[1:]))
        loss = self.criterion(zs, ts)
        return loss

if __name__ == '__main__':
    lm = LM(10)
    print(lm.getLoss([0,1,2,3,4,5,0]))
