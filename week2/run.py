import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from model import *
from utils import *
import time
import numpy as np

# net = Net()
# print(net)

net = DR_GCN()
print(net)



g, features, labels, train_mask, test_mask = load_cora_data()
# Add edges between each node and itself to preserve old node representations
g.add_edges(g.nodes(), g.nodes())
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
best_acc = 0
best_index = 0
for epoch in range(50):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    # logits = net(g, features)
    logits,x_hat = net(g, features)
    logp = F.log_softmax(logits, 1)
    # loss = F.nll_loss(logp[train_mask], labels[train_mask])
    regulizer = nn.MSELoss(reduction='mean')
    loss = F.nll_loss(logp[train_mask], labels[train_mask])+regulizer(x_hat[train_mask],features[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(net, g, features, labels, test_mask)
    if acc > best_acc:
        best_acc = acc
        best_index = epoch
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc, np.mean(dur)))
print(f'{best_index}th epoch best: {best_acc}')

