import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull


class LoadData(nn.Module):
    def __init__(self, root, name, pre_transform, transform):
        super(LoadData, self).__init__()
        self.root = root
        self.name = name.lower()
        self.pre_transform = pre_transform
        self.transform = transform

    def load(self):
        if self.name == "cora":
            self.dataset = Planetoid(root=self.root, name="Cora",
                                     pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "citeseer":
            self.dataset = Planetoid(root=self.root, name="CiteSeer",
                                     pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "pubmed":
            self.dataset = Planetoid(root=self.root, name="PubMed",
                                     pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "corafull":
            self.dataset = CoraFull(root=self.root,
                                    pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "computers":
            self.dataset = Amazon(root=self.root, name="Computers",
                                  pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "photo":
            self.dataset = Amazon(root=self.root, name="Photo",
                                  pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "cs":
            self.dataset = Coauthor(root=self.root, name="CS",
                                    pre_transform=self.pre_transform, transform=self.transform)
        elif self.name == "physics":
            self.dataset = Coauthor(root=self.root, name="Physics",
                                    pre_transform=self.pre_transform, transform=self.transform)
        else:
            raise ValueError("{} dataset is not included".format(self.name))

    def split(self, split_type="random", num_train_per_class=20, num_val=500, num_test=1000):
        data = self.dataset.get(0)
        if split_type=="public" and hasattr(data, "train_mask"):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else:
            train_mask = torch.zeros_like(data.y, dtype=torch.bool)
            val_mask = torch.zeros_like(data.y, dtype=torch.bool)
            test_mask = torch.zeros_like(data.y, dtype=torch.bool)

            for c in range(self.dataset.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                train_mask[idx] = True

            remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            val_mask[remaining[:num_val]] = True
            test_mask[remaining[num_val:num_val + num_test]] = True
        return (train_mask, val_mask, test_mask)