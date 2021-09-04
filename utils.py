import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce, transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes

from typing import Optional, Union, Tuple
from torch import Tensor

import random
import numpy as np

from scipy.linalg import svd
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, euclidean_distances


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


# modified from https://github.com/giannisnik/cnn-graph-classification/blob/master/kcnn/nystrom.py
class Nystrom():
    def __init__(self, kernel_type="rbf", kernel_params=None, n_components=100, random_state=None):
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
    
    def fit(self, X):
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]
		
        # get basis vectors
        n_components = min(n_samples, self.n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis= X[basis_inds, :]
        
        if self.kernel_type == "rbf":
            self.kernel = rbf_kernel
        elif self.kernel_type == "linear":
            self.kernel = linear_kernel
        elif self.kernel == "distance":
            self.kernel = euclidean_distances
        
        basis_kernel = self.kernel(basis, basis, **self._get_kernel_params())
        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = inds
        return self
        
    def transform(self, X):
        embedded = self.kernel(X, self.components_, **self._get_kernel_params())
        print("embedd shape, ", embedded.shape)
        return np.dot(embedded, self.normalization_.T)

    def transform_full(self, X):
        embedded = self.kernel(X, self.components_, **self._get_kernel_params())
        basis_kernel = self.kernel(self.components_, self.components_, **self._get_kernel_params())
        out = embedded @ np.linalg.pinv(basis_kernel) @ embedded.T
        #out = embedded @ self.normalization_.T @ self.normalization_ @ embedded.T
        out /= np.outer(np.sqrt(np.diag(out)), np.sqrt(np.diag(out)))
        out = np.nan_to_num(out)
        return out

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        return params


def spectral_norm(num_nodes, edge_index, edge_attr=None):
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)

    size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
    adj = torch.sparse_coo_tensor(edge_index, edge_attr, size)
    adj = adj.to_dense().squeeze()
    sigma = torch.linalg.norm(adj, ord=2)
    return sigma


def to_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                  num_nodes: Optional[int] = None,
                  reduce: str = "add") -> Union[Tensor, Tuple[Tensor, Tensor]]:

    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=1)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                    num_nodes, reduce)
    if edge_attr is None:
        return edge_index
    else:
        return edge_index, edge_attr