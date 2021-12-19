import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F


gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')#0.76左右
# gcn_reduce = fn.mean(msg='m', out='h')#0.7左右
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x

class DR_GCN(nn.Module):
    def __init__(self):
        super(DR_GCN, self).__init__()
        self.layer1 = GCNLayer(1433, 128)
        self.layer2 = GCNLayer(128, 7)
        self.layer3 = GCNLayer(7, 1433)


    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        pre_label = x
        x = F.relu(x)
        x = self.layer3(g, x)
        return pre_label,x