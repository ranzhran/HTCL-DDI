from argparse import Namespace
import numpy as np
import torch.nn as nn

from typing import List
from .mpn import MPN
from data.mol_tree import Vocab
from nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network followed by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim
        if args.pooling == 'lstm':
            first_linear_dim *= (1 * 2)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


FEATURES_FUSIONER_REGISTRY = {}


def register_features_fusioner(features_fusioner_name: str):
    def decorator(features_fusioner):
        FEATURES_FUSIONER_REGISTRY[features_fusioner_name] = features_fusioner
        return features_fusioner

    return decorator


def get_features_fusioner(features_fusioner_name):
    if features_fusioner_name not in FEATURES_FUSIONER_REGISTRY:
        raise ValueError(f'Features fusioner "{features_fusioner_name}" could not be found. '
                         f'If this fusioner relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_FUSIONER_REGISTRY[features_fusioner_name]


def get_available_features_fusioners():
    """Returns the names of available features generators."""
    return list(FEATURES_FUSIONER_REGISTRY.keys())


def build_model(args: Namespace, ddi:bool = False) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes
    if args.dataset_type == 'multilabel':
        args.output_size = args.num_labels

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')

    if args.jt and args.jt_vocab_file is not None:
        vocab = [x.strip("\r\n ") for x in open(args.jt_vocab_file, 'r')]
        vocab = Vocab(vocab)
    else:
        vocab = None
    model.create_encoder(args, vocab=vocab)
    model.create_ffn(args)

    initialize_weights(model)
    return model