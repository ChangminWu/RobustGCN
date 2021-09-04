import torch
import torch.distributions as tdist
from torch_geometric.utils import remove_self_loops, degree, add_self_loops, negative_sampling
from torch_geometric.utils import erdos_renyi_graph, stochastic_blockmodel_graph, barabasi_albert_graph
from torch_sparse import coalesce

import numpy as np
import networkx as nx

from sklearn.preprocessing import StandardScaler

from utils import Nystrom, sample, filter_adj, spectral_norm, to_undirected
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, euclidean_distances

EPS = 1e-15


class GraphRandomNoise(object):
    # add noise (based on random graph) to the original graph: Erdos-Renyi, Stochastic Block Model and Barabasi-Albert
    def __init__(self, noise_type="erdos-renyi", **kwargs):
        self.noise_type = noise_type
        self.attr_dict = kwargs

    def __call__(self, data):
        device, num_edges = data.edge_index.device, data.edge_index.size(1)
        num_nodes = data.x.size(0)
        assert num_nodes == data.num_nodes
        num_noisy_edges = int(self.attr_dict["noise_ratio"] * num_edges)

        if self.noise_type == "erdos-renyi":
            edge_prob =  num_noisy_edges / (num_nodes * (num_nodes - 1))
            print("edge_probability is {}\n".format(edge_prob))
            edge_index_random = erdos_renyi_graph(num_nodes, edge_prob=edge_prob)

        elif self.random_type == "barabasi-albert":
            num_new_edges = int((num_nodes - np.sqrt(num_nodes**2 - 4*num_noisy_edges)) / 2.)
            edge_index_random = barabasi_albert_graph(num_nodes, num_edges=num_new_edges)

        elif self.random_type == "sbm":
            num_blocks = len(self.attr_dict["edge_probs"])
            num_nodes_per_block = num_nodes // num_blocks
            block_sizes = [ num_nodes - num_nodes_per_block * (num_blocks - 1) if i==0 else num_nodes_per_block
                            for i in range(num_blocks) ]
            edge_index_random = stochastic_blockmodel_graph(block_sizes=block_sizes, **self.attr_dict)
            
        data.edge_index_random = edge_index_random
        return data


# code adpated from negative sampling implementation in Pytorch Geometric
class GraphInsertionNoise(object):
    # sample from negative edges of the original graph as noise
    def __init__(self, noise_ratio=1.0, method="dense"):
        self.noise_ratio = noise_ratio
        self.method = method

    def __call__(self, data):
        edge_index, num_edges = data.edge_index, data.edge_index.size(1)
        num_nodes = data.x.size(0)
        assert num_nodes == data.num_nodes

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index

        # edge_index_random = negative_sampling(edge_index, num_nodes=num_nodes,
        # num_neg_samples=num_noisy_edges, method="sparse", force_undirected=False)

        num_noisy_edges = min(num_nodes*num_nodes-edge_index.size(1), self.noise_ratio*num_edges)
        num_noisy_edges = max(1, int(num_noisy_edges//2))  ### add at least 1 edge

        size = (num_nodes * (num_nodes + 1)) // 2
        mask = row <= col
        row, col = row[mask], col[mask]
        idx = row * num_nodes + col - row * (row + 1) // 2

        alpha = abs(1 / (1 - 1.1 * (edge_index.size(1) / size)))

        if self.method == 'dense':
            mask = edge_index.new_ones(size, dtype=torch.bool)
            mask[idx] = False
            mask = mask.view(-1)

            perm = sample(size, int(alpha * num_noisy_edges), device=edge_index.device)
            perm = perm[mask[perm]][:num_noisy_edges]

        else:
            perm = sample(size, int(alpha * num_noisy_edges))
            mask = torch.from_numpy(np.isin(perm,  idx.to('cpu'))).to(torch.bool)
            perm = perm[~mask][:num_noisy_edges] #.to(edge_index.device).to(torch.long)

        perm = perm.numpy()
        row = ((2. * num_nodes + 1) - np.sqrt((2. * num_nodes + 1)**2 - 8. * perm)) // 2
        col = perm - row * (2 * num_nodes - row -1) // 2

        row, col = torch.LongTensor(row), torch.LongTensor(col)

        edge_index_random = torch.stack([row, col], dim=0).to(edge_index.device).long()
        edge_index_random = to_undirected(edge_index_random, num_nodes, reduce="add")
        data.edge_index_random = edge_index_random
        return data


# code adpated from structure negative sampling implementation in Pytorch Geometric
class GraphStructuralInsertionNoise(object):
    # sample from negative edges of the original graph as noise; 
    # structure means one edge sampled for each existing edge of every node 
    def __call__(self, data):
        edge_index, num_nodes = data.edge_index, data.x.size(0)
        i, j = edge_index.to('cpu')
        idx_1 = i * num_nodes + j

        k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
        idx_2 = i * num_nodes + k

        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        rest = mask.nonzero(as_tuple=False).view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
            idx_2 = i[rest] * num_nodes + tmp
            mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
            k[rest] = tmp
            rest = rest[mask.nonzero(as_tuple=False).view(-1)]
        
        edge_index_random = torch.stack([edge_index[0], k.to(edge_index.device)], dim=0)
        edge_index_random = to_undirected(edge_index_random)
        data.edge_index_random, _ = remove_self_loops(edge_index_random, edge_attr=None)
        return data


class GraphDeletionNoise(object):
    # randomly remove edges from the original graph
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def __call__(self, data):
        edge_index, num_nodes = data.edge_index, data.x.size(0)
        row, col = edge_index

        row, col, _ = filter_adj(row, col, None, row < col)

        mask = edge_index.new_full((row.size(0),), self.noise_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask).to(torch.bool)

        row, col, _ = filter_adj(row, col, None, mask)
        edge_index = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
        data.edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        data.edge_index = edge_index
        return data


class NoiseMerge(object):
    # merge noise with original graph and store in edge_index_random
    def __call__(self, data):
        if hasattr(data, "edge_index_random"):
            edge_index, edge_index_random = data.edge_index, data.edge_index_random
            num_nodes = data.num_nodes
            edge_index = torch.cat([edge_index, edge_index_random], dim=1)
            data.edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        return data


class NodeFeatNoise(object):
    def __init__(self, noise_ratio, keep_origin=True):
        self.noise_ratio = noise_ratio
        self.keep_origin=True

    def __call__(self, data):
        x, num_nodes = data.x, data.num_nodes
        std = torch.std(x, dim=0)
        noise_dist = tdist.Normal(loc=0.0, scale=std*self.noise_ratio)
        noise = noise_dist.sample((num_nodes,))

        if self.keep_origin:
            data.x += noise
        else:
            data.x = noise
        return data


class NodeFeatKernel(object):
    # compute kernel value based on node embeddings between every pair of nodes
    def __init__(self, kernel="rbf", add_identity=True, approximate=True, standarize=True, centerize=True):
        self.kernel = kernel
        self.add_identity = add_identity
        self.approximate = approximate
        self.standarize = standarize
        self.centerize = centerize

    def __call__(self, data):
        x, device = data.x, data.edge_index.device
        num_nodes = data.num_nodes

        row = torch.arange(num_nodes, dtype=torch.long, device=device)
        col = torch.arange(num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        edge_index_kernel = torch.stack([row, col], dim=0)

        edge_index_kernel, _ = remove_self_loops(edge_index_kernel, edge_attr=None)
        
        if self.add_identity:
            edge_index_kernel, _ = add_self_loops(edge_index_kernel)
        
        data.edge_index_kernel = edge_index_kernel
        
        feat = x.numpy()
        if self.standarize:
            scaler = StandardScaler()
            scaler.fit(feat)
            feat = scaler.transform(feat)

        if self.approximate:
            model = Nystrom(kernel_type=self.kernel, n_components=500)
            model.fit(feat)
            Q = model.transform_full(feat)
        else:
            if self.kernel == "rbf":
                kernel_func = rbf_kernel
            elif self.kernel == "linear":
                kernel_func = linear_kernel
            elif self.kernel == "distance":
                kernel_func = euclidean_distances
            else:
                raise ValueError("{} not implemented".format(self.kernel))

            Q = kernel_func(feat, feat)

        if not self.add_identity:
            np.fill_diagonal(Q, 0)

        if self.centerize:
            P = np.eye(Q.shape[0]) - np.ones([Q.shape[0], Q.shape[0]])/Q.shape[0]
            Q = P @ Q @ P

        Q = Q.reshape(-1, 1)
        Q = torch.FloatTensor(Q)
        data.edge_attr_kernel = Q
        return data


class NodeFeatKernelSparsification(object):
    # sparsify kernel matrix to reduce computational cost
    def __init__(self, sparsifier="origin", ratio=1.0, add_identity=True, normalize='none'):
        '''
        sparsifier: sparsification method
        ratio: density of the sparsified kernel comparing to the original graph
        add_identity: whether to add identity (self-loops) on kernels
        normalize: whether to perform spectral normalization on kernel values
        '''
        self.sparsifier = sparsifier
        self.ratio = ratio
        self.add_identity = add_identity
        self.normalize = normalize

    def __call__(self, data):
        edge_index, device = data.edge_index, data.edge_index.device
        num_edges, num_nodes = edge_index.size(1), data.x.size(0)

        if self.normalize != 'none':
            sigma = spectral_norm(num_nodes, edge_index)
            print("Spectral norm of Adj is {}".format(sigma))

            if self.normalize == "preprocess":
                edge_index_norm, edge_attr_norm = data.edge_index_kernel, data.edge_attr_kernel
                if not self.add_identity:
                    edge_index_norm, _ = add_self_loops(edge_index_norm, _)
                sigma_norm = spectral_norm(num_nodes, edge_index_norm, edge_attr_norm)
                print("Spectral norm of Complete Kernel Matrix is {}".format(sigma_norm))

                data.edge_attr_kernel = data.edge_attr_kernel * (sigma + EPS) / (sigma_norm + EPS)

        Q = data.edge_attr_kernel.view(num_nodes, num_nodes).numpy()

        if self.sparsifier == "origin":
            edge_index_kernel = edge_index
            row_kernel, col_kernel = edge_index

        elif self.sparsifier == "random":
            edge_prob = min(1.0, self.ratio*num_edges/(num_nodes*(num_nodes-1)))
            print("edge_probability is {}\n".format(edge_prob))

            edge_index_kernel = erdos_renyi_graph(num_nodes, edge_prob=edge_prob)

        elif self.sparsifier == "topk":
            k = max(int(2. * self.ratio * num_edges / num_nodes), 1)
            print("Average degree is approximately {}\n".format(k))

            ind = np.argsort(Q, axis=0)[::-1][:k]
            row_kernel = []
            col_kernel = []
            for i in range(ind.shape[1]):
                col_kernel.extend([i] * k)
                row_kernel.extend(list(ind[:, i]))
            row_kernel = torch.LongTensor(row_kernel, device=device)
            col_kernel = torch.LongTensor(col_kernel, device=device)
            edge_index_kernel = torch.stack([row_kernel, col_kernel], dim=0)
            edge_index_kernel = to_undirected(edge_index_kernel)

        elif self.sparsifier == "threshold":
            sorted_edges = np.sort(Q.flatten())[::-1]
            left = sorted_edges[max(int(2. * self.ratio * num_edges) - 1, 0)]
            right = sorted_edges[int(2. * self.ratio * num_edges)]
            eps = (left + right) / 2
            print("Threshold value is {}\n".format(eps))

            row_kernel, col_kernel = (Q >= eps).nonzero()
            row_kernel = torch.LongTensor(row_kernel, device=device)
            col_kernel = torch.LongTensor(col_kernel, device=device)
            edge_index_kernel = torch.stack([row_kernel, col_kernel], dim=0)
            edge_index_kernel = to_undirected(edge_index_kernel)

        edge_index_kernel, _ = remove_self_loops(edge_index_kernel, edge_attr=None)
        if self.add_identity:
            edge_index_kernel, _ = add_self_loops(edge_index_kernel)

        row_kernel, col_kernel = edge_index_kernel
        Q = Q[row_kernel, col_kernel]

        Q = Q.reshape(-1, 1)
        Q = torch.FloatTensor(Q)

        data.edge_index_kernel, data.edge_attr_kernel = edge_index_kernel, Q

        if self.normalize == "postprocess":
            edge_index_norm, edge_attr_norm = data.edge_index_kernel, data.edge_attr_kernel
            sigma_norm = spectral_norm(num_nodes, edge_index_norm, edge_attr_norm)
            print("Spectral norm of Sparsified Kernel Matrix is {}".format(sigma_norm))

            data.edge_attr_kernel = data.edge_attr_kernel * (sigma + EPS) / (sigma_norm + EPS)
        return data


class NodeFeatKernelDirect(object):
    # directly apply kernel sparsification on large graphs such as ogbn-arxiv
    def __init__(self, ratio=1.0, add_identity=True):
        '''
        ratio: density of the sparsified kernel comparing to the original graph
        add_identity: whether to add identity (self-loops) on kernels
        '''
        self.ratio = ratio
        self.add_identity = add_identity

    def __call__(self, data):
        edge_index, device = data.edge_index, data.edge_index.device
        num_edges, num_nodes = edge_index.size(1), data.x.size(0)

        edge_prob = min(1.0, self.ratio*num_edges/(num_nodes*(num_nodes-1)))
        print("edge_probability is {}\n".format(edge_prob))

        edge_index_kernel = edge_index

        edge_index_kernel, _ = remove_self_loops(edge_index_kernel, edge_attr=None)
        if self.add_identity:
            edge_index_kernel, _ = add_self_loops(edge_index_kernel)

        row_kernel, col_kernel = edge_index_kernel
        # inner product kernel
        Q = torch.bmm(data.x[row_kernel].unsqueeze(-2), data.x[col_kernel].unsqueeze(-1)).squeeze().view(-1, 1)
        data.edge_index_kernel, data.edge_attr_kernel = edge_index_kernel, Q
        return data