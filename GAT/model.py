import torch
import torch.nn as nn 
import torch.nn.functional as F
from layer import GraphConvolutionLayer, GraphAttentionLayer, SparseGraphConvolutionLayer, SparseGraphAttentionLayer

# TODO step 1.
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(nfeat, nhid,dropout)
        self.layer2 = nn.ReLU()
        self.layer3 = GraphConvolutionLayer(nhid, nclass,dropout)
    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x)
        x = self.layer3(x, adj)
        return x
        
    
# TODO step 2.
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.layer1 = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat= True) for _ in range(nheads)])
        self.dropout2 = nn.Dropout(dropout)
        self.layer2 = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat = False)
    def forward(self, x, adj):
        x = self.dropout1(x)
        x = torch.cat([layer(x, adj) for layer in self.layer1], dim = 1)
        x = self.dropout2(x)
        x = self.layer2(x, adj)
        return x
    


# TODO step 3.
class SpGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SpGCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(nfeat, nhid,dropout)
        self.layer2 = nn.ReLU()
        self.layer3 = GraphConvolutionLayer(nhid, nclass,dropout)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x)
        x = self.layer3(x, adj)
        return x

class SpGAT(nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        pass

    def forward(self, x, adj):
        pass