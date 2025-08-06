#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#

__all__ = ['RNN', 'RNNWithInit', 'Regressor', 'NeuralInitialization']


import os
import torch
import torch.utils.data
from torch import nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import *


class Regressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims, init_dim, rnn_type='LSTM', n_layers=2):
        super().__init__()
        self.n_outs = len(out_dims)

        self.rnn = getattr(nn, rnn_type.upper())(
            in_dim + init_dim, hid_dim, n_layers,
            bidirectional=False, batch_first=True, dropout=0.3)

        for i, out_dim in enumerate(out_dims):
            setattr(self, 'declayer%d'%i, nn.Linear(hid_dim, out_dim))
            nn.init.xavier_uniform_(getattr(self, 'declayer%d'%i).weight, gain=0.01)

    def forward(self, x, inits, h0):
        xc = torch.cat([x, *inits], dim=-1)
        xc, h0 = self.rnn(xc, h0)

        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d'%j)(xc)
            preds.append(out)

        return preds, xc, h0


class NeuralInitialization(nn.Module):
    def __init__(self, in_dim, hid_dim, rnn_type, n_layers):
        super().__init__()

        out_dim = hid_dim
        self.n_layers = n_layers
        self.num_inits = int(rnn_type.upper() == 'LSTM') + 1
        out_dim *= self.num_inits * n_layers

        self.init_net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim * self.n_layers),
            nn.ReLU(),
            nn.Linear(hid_dim * self.n_layers, out_dim),
        )

    def forward(self, x):
        b = x.shape[0]

        out = self.init_net(x)
        out = out.view(b, self.num_inits, self.n_layers, -1).permute(1, 2, 0, 3).contiguous()

        if self.num_inits == 2:
            return tuple([_ for _ in out])
        return out[0]


class Integrator(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel=1024):
        super().__init__()

        self.integrate_net = nn.Sequential(
            nn.Linear(in_channel, hid_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_channel, hid_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_channel, out_channel),
        )

    def forward(self, x, feat):
        res = x
        mask = (feat != 0).all(dim=-1).all(dim=-1)

        out = torch.cat((x, feat), dim=-1)
        out = self.integrate_net(out)
        out[mask] = out[mask] + res[mask]

        return out


class RNN(nn.Module):
    r"""
    An RNN net including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNN.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super().__init__()
        self.rnn = getattr(torch.nn, rnn_type.upper())(hidden_size, hidden_size, num_rnn_layer,
                                                       bidirectional=bidirectional, dropout=dropout)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, init=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains tensors in shape [num_frames, input_size].
        :param init: Initial hidden states.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        length = [_.shape[0] for _ in x]
        x = self.dropout(relu(self.linear1(pad_sequence(x))))
        x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), init)[0]
        x = self.linear2(pad_packed_sequence(x)[0])
        return [x[:l, i].clone() for i, l in enumerate(length)]


class RNNWithInit(RNN):
    r"""
    RNN with the initial hidden states regressed from the first output.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNNWithInit net.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        assert rnn_type == 'lstm' and bidirectional is False
        super().__init__(input_size, output_size, hidden_size, num_rnn_layer, rnn_type, bidirectional, dropout)

        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * num_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * num_rnn_layer, 2 * (2 if bidirectional else 1) * num_rnn_layer * hidden_size)
        )

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, _=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 2-tuple
                  (Tensor[num_frames, input_size], Tensor[output_size]).
        :param _: Not used.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        x, x_init = list(zip(*x))
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, (h, c))
