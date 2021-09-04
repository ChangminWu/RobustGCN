import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.utils import dense_to_sparse, add_self_loops, remove_self_loops, to_undirected
from torch_geometric.utils import erdos_renyi_graph

from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter_add
import math


class MLP(torch.nn.Module):
    def __init__(self, indim, hiddim):
        super(MLP, self).__init__()
        self.lin = torch.nn.Linear(indim, hiddim)

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, data=None, merged=False):
        return self.lin(x)


class GCNRobustConv(MessagePassing):
    def __init__(self, indim, outdim, 
                 epsilon=1.0, 
                 learnable_eps=False,
                 add_identity=True,
                 normalize=True,
                 use_rf=False,
                 rf_norm=True,
                 use_gat=False):
        super(GCNRobustConv, self).__init__(aggr='add')

        self.init_eps = epsilon
        self.learnable_eps = learnable_eps
        self.add_identity = add_identity
        self.normalize= normalize
        self.use_rf = use_rf
        self.use_gat = use_gat
        self.rf_norm = rf_norm
        self.indim, self.outdim = indim, outdim

        if self.use_rf:
            if self.rf_norm:
                self.register_buffer('weight', torch.randn(indim, outdim) / torch.Tensor([max(indim, outdim)]).sqrt())
            else:
                self.register_buffer('weight', torch.randn(indim, outdim))
        else:
            self.linear = torch.nn.Linear(indim, outdim)

        if self.learnable_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([epsilon]))
        else:
            self.register_buffer("eps", torch.Tensor([epsilon]))

        if self.use_gat:
            if outdim % 8 == 0:
                self.conv = GATConv(indim, outdim//8, heads=8, dropout=0.5)
            elif outdim % 8 != 0:
                self.conv = GATConv(indim, outdim, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index, data, merged=True):
        if self.use_rf:
            x = torch.matmul(x, self.weight.to(x.device))
        else:
            x = self.linear(x)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.use_gat:
            out = self.eps * self.conv(x, edge_index)
        else:
            out = self.eps * (self.propagate(edge_index, x=x, norm=norm) + F.relu(x) * 1. / deg.view(-1, 1))

        if not merged and hasattr(data, "edge_index_random"):
            edge_index_random = data.edge_index_random
            edge_index_random, _ = remove_self_loops(edge_index_random, edge_attr=None)
            if self.add_identity:
                edge_index_random, _ = add_self_loops(edge_index_random)

            if self.normalize:
                row_random, col_random = edge_index_random
                deg_random = degree(row_random, x.size(0), dtype=x.dtype)
                deg_random_inv_sqrt = deg_random.pow(-0.5)
                deg_random_inv_sqrt[deg_random_inv_sqrt == float('inf')] = 0
                norm_random = deg_random_inv_sqrt[row_random] * deg_random_inv_sqrt[col_random]
            else:
                norm_random = None

            out += (1 - self.eps) * self.propagate(edge_index_random, x=x, norm=norm_random)

        if hasattr(data, "edge_index_kernel"):
            edge_index_kernel, edge_attr_kernel = data.edge_index_kernel, data.edge_attr_kernel

            if self.normalize:
                row_kernel, col_kernel = edge_index_kernel
                deg_kernel = torch.zeros((x.size(0), ), dtype=x.dtype, device=edge_index_kernel.device)
                deg_kernel = deg_kernel.scatter_add_(0, row_kernel, edge_attr_kernel.squeeze())
                deg_kernel = torch.clamp(deg_kernel, min=0.0)
                deg_kernel_inv_sqrt = deg_kernel.pow(-0.5)
                deg_kernel_inv_sqrt[deg_kernel_inv_sqrt == float('inf')] = 0
                norm_kernel = deg_kernel_inv_sqrt[row_kernel] * deg_kernel_inv_sqrt[col_kernel]
            else:
                norm_kernel = None

            out += (1 - self.eps) * self.propagate(edge_index_kernel, x=x,
                                                   edge_attr=edge_attr_kernel, norm=norm_kernel)

        return out

    def message(self, x_j, edge_attr=None, norm=None):
        if edge_attr is not None:
            out = F.relu(x_j * edge_attr)
        else:
            out = F.relu(x_j)
        if norm is not None:
            out = norm.view(-1, 1) * out
        return out

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        if not self.use_rf:
            glorot(self.linear.weight)
            zeros(self.linear.bias)
        else:
            self.weight = torch.randn(self.indim, self.outdim)
            if self.rf_norm:
                self.weight /= torch.Tensor([max(self.indim, self.outdim)]).sqrt()

        if self.learnable_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([self.init_eps]))

        if self.use_gat:
            self.conv.reset_parameters()