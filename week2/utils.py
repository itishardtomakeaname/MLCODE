import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import CoraGraphDataset
def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0] #2078 nodes,10556 edges
    features = g.ndata['feat'] #feature_dim 1433
    labels = g.ndata['label'] #max(label)=6,totally 7 classes
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    return g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)