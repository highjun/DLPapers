
import numpy as np 
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter

# TODO step 1. 
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolutionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        a = 1/ np.sqrt(self.W.size(1))
        self.W.data.uniform_(-a, a)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj):
        x = torch.mm(adj, input)
        x = torch.mm(x, self.W)
        x = self.dropout(x)
        return x


# TODO step 2. 
class GraphAttentionLayer(nn.Module):
    """multihead attention """ 
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        u = 1/ np.sqrt(self.W.size(1))
        self.W.data.uniform_(-u, u)

        self.a = Parameter(torch.FloatTensor(2*out_features, 1))
        u = 1/ np.sqrt(self.a.size(1))
        self.a.data.uniform_(-u, u)

        self.lrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, input, adj):
        x = torch.mm(input, self.W)
        
        num_node = x.size(0)
        x_1 = x.repeat_interleave(num_node, dim = 0)
        x_2 = x.repeat(num_node, 1)
        concated = torch.cat([x_1, x_2],dim = 1).view(num_node, num_node, 2*self.out_features)
        alpha = self.lrelu(torch.matmul(concated, self.a).squeeze(2))
        mask = -9e15*torch.ones_like(alpha)
        attn = torch.where(adj> 0 , alpha, mask)
        attn = torch.softmax(attn, dim = 1)
        attn = self.dropout(attn)

        new_h = torch.mm(attn, x)
        return self.elu(new_h)



# TODO step 3.
class SparsemmFunction(torch.autograd.Function):
    """ for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a , b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class Sparsemm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SparsemmFunction.apply(indices, values, shape, b)


class SparseGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(SparseGraphConvolutionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        a = 1/ np.sqrt(self.W.size(1))
        self.W.data.uniform_(-a, a)
        self.dropout = nn.Dropout(dropout)
        self.spmm = Sparsemm() 
    def forward(self, input, adj):
        x = torch.mm(input, self.W)
        x = self.spmm(adj, x)
        x = self.dropout(x)
        return x

class SparseGraphAttentionLayer(nn.Module):
    """multihead attention """ 
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        pass 
    
    def forward(self, input, adj):
        pass