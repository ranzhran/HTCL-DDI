import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Tuple, List
from .encoder import GAT, TopoGAT
from .decoder import InnerProductDecoder
from features import BatchMolGraph
from models.mpn import MPN
from data.mol_tree import Vocab
from argparse import Namespace
from data.mol_tree import Vocab
from nn_utils import get_activation_function
from .deepinfomax import GcnInfomax

class HTCL(nn.Module):

    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.3, bias: bool = False,
                 sparse: bool = True):
        super(HTCL, self).__init__()
        self.num_features = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.bias = bias
        self.sparse = sparse
        self.args = args
        self.create_encoder(args)       

        self.struc_enc = self.select_encoder1(args)
        self.seman_enc = self.select_encoder2(args)
        self.dec_local = InnerProductDecoder(args.hidden_size)
        self.dec_global = InnerProductDecoder(args.hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.DGI_setup()
        self.create_ffn(args)

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        self.encoder = MPN(args)
        return self.encoder


    def select_encoder1(self, args: Namespace):
        return GAT(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                          dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)


    def select_encoder2(self, args: Namespace):
        return TopoGAT(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                          dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)


    def create_ffn(self, args: Namespace):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        self.fusion_ffn_local = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fusion_ffn_global = nn.Linear(args.gat_hidden*8, args.ffn_hidden_size)

        ffn = []
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, args.drug_nums),
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout


    def DGI_setup(self):
        self.DGI_model1 = GcnInfomax(self.args)
        self.DGI_model2 = GcnInfomax(self.args)


    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                adj: torch.sparse.FloatTensor,
                adj_tensor,
                drug_nums,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        smiles_batch = batch
        features_batch = None

        # molecular view
        feat_orig = self.encoder(smiles_batch, features_batch)
        feat = self.dropout(feat_orig)
        fused_feat = self.fusion_ffn_local(feat)
        output = self.ffn(fused_feat)
        outputs = self.sigmoid(output)       
        outputs_l = outputs.view(-1)

        # structural view
        embeddings1 = self.struc_enc(feat_orig, adj)
        feat_g1 = self.dropout(embeddings1)
        fused_feat_g1 = self.fusion_ffn_global(feat_g1)
        output_g1 = self.ffn(fused_feat_g1)
        outputs_1 = self.sigmoid(output_g1)
        outputs_g1 = outputs_1.view(-1)

        # semantic view
        embeddings2 = self.seman_enc(feat_orig, adj)
        feat_g2 = self.dropout(embeddings2)
        fused_feat_g2 = self.fusion_ffn_global(feat_g2)
        output_g2 = self.ffn(fused_feat_g2)
        outputs_2 = self.sigmoid(output_g2)
        outputs_g2 = outputs_2.view(-1)
        
        # intergrate the two views
        embeddings = embeddings1 + embeddings2
        feat_g = self.dropout(embeddings)
        fused_feat_g = self.fusion_ffn_global(feat_g)
        output_g = self.ffn(fused_feat_g)
        outputs = self.sigmoid(output_g)

        local_embed = feat_orig
        DGI_loss1 = self.DGI_model1(embeddings1, local_embed, adj_tensor, drug_nums)
        DGI_loss2 = self.DGI_model2(embeddings2, local_embed, adj_tensor, drug_nums)
        
        if return_embeddings:
            return outputs, embeddings

        return outputs_l, outputs_g1, outputs_g2, DGI_loss1, DGI_loss2
