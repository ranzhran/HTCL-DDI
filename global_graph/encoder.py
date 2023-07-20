import torch
import torch.nn as nn
from argparse import Namespace
from .graph_conv import AttnGraphConvolution, Attntopo

class GAT(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False, nheads=8, sparse: bool = True):
        super(GAT, self).__init__()
        self.input_dim = num_features
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse

        GC = AttnGraphConvolution
        self.attentions = [GC(in_features=num_features, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GC(args.gat_hidden * nheads, args.gat_hidden)

    def forward(self, features: torch.Tensor, adj: torch.FloatTensor) -> torch.Tensor:
        
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions], dim=1)
        features = self.dropout(features)

        return features


class TopoGAT(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False, nheads=8, sparse: bool = True):
        super(TopoGAT, self).__init__()
        self.input_dim = num_features
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse

        GC = Attntopo
        self.attentions_1 = [GC(in_features=num_features, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions_2 = [GC(in_features=args.gat_hidden * nheads, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions_3 = [GC(in_features=args.gat_hidden * nheads, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_3):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, features: torch.Tensor, adj: torch.FloatTensor) -> torch.Tensor:
        
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions_1], dim=1)
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions_2], dim=1)
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions_3], dim=1)
        features = self.dropout(features)

        return features
