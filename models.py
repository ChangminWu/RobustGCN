import torch
import torch.nn as nn
import torch.nn.functional as F

from convs.gcn_conv import MLP, GCNRobustConv


class GCNNet(torch.nn.Module):
    def __init__(self, indim, hiddim, num_classes, num_layers=1, readout='mlp',
                 exp_type="random-feature", merged=False, epsilon=1.0, learnable_eps=False,
                 add_identity=True, normalize=True, rf_norm=True):
        super(GCNNet, self).__init__()
    
        self.exp_type = exp_type
        self.merged = merged
        self.num_layers = num_layers

        if self.exp_type == "mlp":
            self.conv = MLP
            params = {}

        elif self.exp_type == "random-feature":
            self.conv = GCNRobustConv
            params = {'use_rf': True,
                      'rf_norm': rf_norm,
                      'epsilon': epsilon,
                      'learnable_eps': learnable_eps,
                      'add_identity': add_identity,
                      'normalize': normalize}

        elif self.exp_type == "gat":
            self.conv = GCNRobustConv
            params = {'use_gat': True,
                      'epsilon': epsilon,
                      'learnable_eps': learnable_eps,
                      'add_identity': add_identity,
                      'normalize': normalize}

        else:
            self.conv = GCNRobustConv
            params = {'epsilon': epsilon,
                      'learnable_eps': learnable_eps,
                      'add_identity': add_identity,
                      'normalize': normalize}

        if readout == 'mlp':
            self.readout = MLP(hiddim, num_classes)

        elif readout == 'conv':
            self.readout = self.conv(hiddim, num_classes, **params)

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(self.conv(indim, hiddim, **params))
        for _ in range(self.num_layers-1):
            self.conv_layers.append(self.conv(hiddim, hiddim, **params))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.conv_layers):
            if i==0:
                out = F.relu(conv(x, edge_index, data, merged=self.merged))
            else:
                out = F.relu(conv(out, edge_index, data, merged=self.merged))
            out = F.dropout(out, training=self.training)
        out = self.readout(out, edge_index, data, merged=self.merged)
        return F.log_softmax(out, dim=1)

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        self.readout.reset_parameters()