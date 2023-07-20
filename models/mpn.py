from argparse import Namespace
from typing import List, Union, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

from features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from nn_utils import index_select_ND, get_activation_function
from model_utils import convert_to_2D, convert_to_3D, compute_max_atoms, compute_max_bonds, convert_to_3D_bond

class MPN_Atom(nn.Module):
    """A message passing neural network for encoding atoms in a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPN_Atom, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size//2
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        w_h_input_size = self.hidden_size

        self.weight_tying = self.args.weight_tying
        n_message_layer = 1 if self.weight_tying else self.depth - 1

        self.W_h = nn.ModuleList([nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
                                  for _ in range(n_message_layer)])

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        self.i_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.j_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> Union[torch.FloatTensor, torch.Tensor]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()


        # Input

        input = self.W_i(f_bonds)  # num_bonds x hidden_size
    
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            step = 0 if self.weight_tying else depth
            message = self.W_h[step](message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        # readout
        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.args.attn_output:
            mol_vecs = self.attention(atom_hiddens, a_scope)
            return mol_vecs

    def attention(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        """
        :param atom_hiddens: (num_atoms, hidden_size)
        :param a_scope: list of tuple (int, int)
        :return: (num_atoms, hidden_size * attn_num_r)
        """
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        max_atoms = compute_max_atoms(a_scope)

        batch_hidden, batch_mask = convert_to_3D(atom_hiddens, a_scope, max_atoms, device=device, self_attn=True)

        e = torch.sum(torch.sigmoid(self.j_layer(batch_hidden)) * self.i_layer(batch_hidden), dim=1)
        return e




class MPN_Bond(nn.Module):
    """aggregating bonds in a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPN_Bond, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size//2
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.use_input_features = args.use_input_features
        self.args = args


        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        self.i_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.j_layer = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> Union[torch.FloatTensor, torch.Tensor]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()


        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        f_bonds = self.W_i(f_bonds)  # num_bonds x hidden_size

        #采用attention聚合化学键向量
        mol_vecs = self.attention(f_bonds, b_scope)
        return mol_vecs
        
    
    def attention(self, bond_hiddens: torch.Tensor, b_scope: List[Tuple[int, int]]) -> torch.Tensor:
        """
        :param atom_hiddens: (num_atoms, hidden_size)
        :param a_scope: list of tuple (int, int)
        :return: (num_atoms, hidden_size * attn_num_r)
        """
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        max_bonds = compute_max_bonds(b_scope)
    

        # batch_hidden: (batch_size, max_atoms, hidden_size)
        # batch_mask: (batch_size, max_atoms, max_atoms)
        batch_hidden, batch_mask = convert_to_3D_bond(bond_hiddens, b_scope, max_bonds, device=device, self_attn=True)

        e = torch.sum(torch.sigmoid(self.j_layer(batch_hidden)) * self.i_layer(batch_hidden), dim=1)
        return e



class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder_atom = MPN_Atom(self.args, self.atom_fdim, self.bond_fdim)
        self.encoder_bond = MPN_Bond(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output1 = self.encoder_atom.forward(batch, features_batch).cpu()
        output2 = self.encoder_bond.forward(batch, features_batch).cpu()
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        output = torch.cat([output1, output2], axis=1).to(device)

        return output


class Mixture(nn.Module):
    def __init__(self, feat_size, output_size):
        super(Mixture, self).__init__()
        self.feat_size = feat_size
        self.output_size = output_size
        ffn = [
            nn.Linear(feat_size * 2, output_size),
            nn.ReLU(),
        ]
        self.ffn = nn.Sequential(*ffn)

    def forward(self, feat_1, feat_2):
        if torch.cuda.is_available():
            feat_1, feat_2 = feat_1.cuda(), feat_2.cuda()
        return self.ffn(torch.cat((feat_1, feat_2), dim=-1))