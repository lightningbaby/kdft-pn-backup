#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from toolkit import encoder
import numpy as np
from methods import Attention
import json


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


def init_1Dlayer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv1d):
        n = L.kernel_size[0] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm1d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):  # x[16,512]
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)  # x_norm[16,512]
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(
            self.L.weight.data)  # self.L Linear(512,200) L_norm [200,512]
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)  # [200,512]
        cos_dist = self.L(x_normalized)  # [16,200]
        scores = 10 * cos_dist  # [16,200]
        return scores


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class Conv2d_ft(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(1, 0), bias=True):
        super(Conv2d_ft, self).__init__(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, padding=self.padding)
            else:
                out = super(Conv2d_ft, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, padding=self.padding)
            else:
                out = super(Conv2d_ft, self).forward(x)
        return out


# --- Conv1d module ---
class Conv1d_ft(nn.Conv1d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Conv1d_ft, self).__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv1d(x, self.weight.fast, None, padding=self.padding)
            else:
                out = super(Conv1d_ft, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv1d(x, self.weight.fast, self.bias.fast, padding=self.padding)
            else:
                out = super(Conv1d_ft, self).forward(x)
        return out


# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


# --- feature transformation layer ---
class FeatureTransformation2d_ft(nn.BatchNorm2d):
    feature_augment = False

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureTransformation2d_ft, self).__init__(num_features, momentum=momentum,
                                                         track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.003)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.005)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature transformation
        if self.feature_augment and self.training:
            # gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
            # beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device) * (
                self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * (
                self.beta)).expand_as(out)
            out = gamma * out + beta
        return out


# --- feature transformation layer ---
class FeatureTransformation1d_ft(nn.BatchNorm1d):
    feature_augment = False

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureTransformation1d_ft, self).__init__(num_features, momentum=momentum,
                                                         track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1) * 0.003)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1) * 0.005)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            # gamma = (1 + torch.randn(1, self.num_features, 1,  dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
            # beta = (torch.randn(1, self.num_features, 1,  dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
            gamma = (1 + torch.randn(1, self.num_features, 1, dtype=self.gamma.dtype, device=self.gamma.device) * (
                self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, dtype=self.beta.dtype, device=self.beta.device) * (
                self.beta)).expand_as(out)
            out = gamma * out + beta
        return out



class BatchNorm2d_ft(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_ft, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- BatchNorm1d ---
class BatchNorm1d_ft(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_ft, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out




class OneDBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res, leaky=False):
        super(OneDBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv1d_ft(indim, outdim, kernel_size=3, padding=1)
            self.BN1 = BatchNorm1d_ft(outdim)
            self.C2 = Conv1d_ft(outdim, outdim, kernel_size=3, padding=1)
            self.BN2 = FeatureTransformation1d_ft(outdim)
        else:
            self.C1 = nn.Conv1d(indim, outdim, kernel_size=3, padding=1)
            self.BN1 = nn.BatchNorm1d(outdim)
            self.C2 = nn.Conv1d(outdim, outdim, kernel_size=3, padding=1)
            self.BN2 = nn.BatchNorm1d(outdim)
        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        # if indim!=outdim:
        #   if self.maml:
        #     self.shortcut = Conv1d_ft(indim, outdim, 1, 2 if half_res else 1)
        #     self.BNshortcut = FeatureTransformation1d_ft(outdim)
        #   else:
        #     self.shortcut = nn.Conv1d(indim, outdim, 1, 2 if half_res else 1)
        #     self.BNshortcut = nn.BatchNorm1d(outdim)
        #
        #   self.parametrized_layers.append(self.shortcut)
        #   self.parametrized_layers.append(self.BNshortcut)
        #   self.shortcut_type = '1x1'
        # else:
        #   self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_1Dlayer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        # short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        # out = out + short_out
        out = self.relu2(out)
        return out


# --- 2-d Block ---
class TwoDBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res, leaky=False):
        super(TwoDBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_ft(indim, outdim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
            self.BN1 = BatchNorm2d_ft(outdim)
            self.C2 = Conv2d_ft(outdim, outdim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
            self.BN2 = FeatureTransformation2d_ft(
                outdim)  # feature-wise transformation at the end of each residual block
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_ft(indim, outdim, 1, 1, padding=(0, 0), bias=False)
                self.BNshortcut = FeatureTransformation2d_ft(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 1, padding=0, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):  # [50,230,128,1]
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out




# --- Conv module ---
class Conv1d_ft(nn.Conv1d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=True):
        super(Conv1d_ft, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv1d(x, self.weight.fast, None, padding=self.padding)
            else:
                out = super(Conv1d_ft, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv1d(x, self.weight.fast, self.bias.fast, padding=self.padding)
            else:
                out = super(Conv1d_ft, self).forward(x)
        return out


class ContextPurificationEncoder(nn.Module):
    maml = False

    def __init__(self, block, word_vec_mat, list_of_num_layers, list_of_out_dims, max_length,
                 word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230, num_heads=4,
                 flatten=True, leakyrelu=False):
        nn.Module.__init__(self)
        self.grads = []
        self.fmaps = []
        self.final_feat_dim = 230
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = encoder.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)

        #        seq_len, feature_dim, head_num = max_length, word_embedding_dim + 2 * pos_embedding_dim, num_heads
        #        weights = np.random.standard_normal((feature_dim, feature_dim * 4))
        #        bias = np.random.standard_normal((feature_dim * 4,))

        seq_len, feature_dim, head_num = max_length, word_embedding_dim + 2 * pos_embedding_dim, num_heads
        weights = np.random.standard_normal((self.hidden_size * 2, self.hidden_size * 8))
        bias = np.random.standard_normal((self.hidden_size * 8,))

        self.attention = Attention.get_torch_layer_with_weights(self.hidden_size * 2, head_num, weights, bias)
        #        self.attention = Attention.get_torch_layer_with_weights(feature_dim, head_num, weights, bias)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        #        self.lstm = torch.nn.LSTM(feature_dim, feature_dim, bidirectional=True)
        self.gru = torch.nn.GRU(feature_dim, self.hidden_size, bidirectional=True)
        self.att_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        #        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        #        self.pool = nn.MaxPool1d(max_length)

        if self.maml:
            conv1 = Conv1d_ft(self.max_length, self.hidden_size, 3, padding=1)
            bn1 = BatchNorm1d_ft(self.hidden_size)
        else:
            conv1 = nn.Conv1d(self.max_length, self.hidden_size, 3, padding=1)

            bn1 = nn.BatchNorm1d(self.hidden_size)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        # pool1 = nn.MaxPool1d(3)

        init_1Dlayer(conv1)
        init_1Dlayer(bn1)

        trunk1 = [conv1, bn1, relu]
        trunk2 = []

        indim = 230

        for i in range(3):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
                trunk2.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            maxpool = nn.MaxPool1d(self.hidden_size)
            trunk2.append(maxpool)
            trunk2.append(Flatten())

        self.trunk1 = nn.Sequential(*trunk1)
        self.trunk2 = nn.Sequential(*trunk2)

    def forward(self, inputs):  # [50,512]
        # embedding
        x = self.embedding(inputs)  # x [50,128,60] [batch，128维，50+5+5（word 50维，每个pos5维）]

        # lstm & multi-head self-attention
        x = self.MHSFATT(x)  # [50,128,60]

        # cnn
        # x = self.cnn(x) # [50,230,128]
        # x = x.transpose(1,2)
        x = self.trunk1(x)  # [50,230,230]
        x = self.trunk2(x)  # x[batch,230,1] [50,230]
        #        x = x.squeeze(2) # x[batch,230]

        return x

    def MHSFATT(self, x):
        x = x.transpose(0, 1)
        x, hidden = self.gru(x)
        x = x.transpose(0, 1).double()
        # x = self.maxpool(x)
        x = self.attention(x, x, x).float()
        # x = x/100
        x = self.att_linear(x)
        return x




# -----------only conv2d--------------------
class TwoDCNNEncoder(nn.Module):  # only conv2d
    def __init__(self, word_vec_mat, max_length,
                 word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230, flatten=True, leakyrelu=False):
        super(TwoDCNNEncoder, self).__init__()
        self.grads = []
        self.fmaps = []
        self.final_feat_dim = 230
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = encoder.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)

        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv2d(1, self.hidden_size, kernel_size=(3, self.embedding_dim), padding=(1, 0), bias=False)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs):
        # embedding inputs [50,512]
        x = self.embedding(inputs)  # x [50,128,60] [batch，128维，50+5+5（word 50维，每个pos5维）]

        # cnn
        x = self.cnn(x)  # [50,230,128]
        return x

    def cnn(self, inputs):
        x = inputs.unsqueeze(1)  # [50,1,128,60]
        x = self.conv(x)  # [50,230,128,1]
        x = F.relu(x)
        x = x.squeeze(-1)
        x = self.pool(x)  # [50,230,1]
        return x.squeeze(2)  # n x hidden_size


# --------------att cnn--------------------------------
class ATTCNNEncoder(nn.Module):  # for gru
    def __init__(self, word_vec_mat, max_length,
                 word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230, num_heads=2, flatten=True, leakyrelu=False):
        super(ATTCNNEncoder, self).__init__()
        self.grads = []
        self.fmaps = []
        self.final_feat_dim = 230
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = encoder.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)

        seq_len, feature_dim, head_num = max_length, word_embedding_dim + 2 * pos_embedding_dim, num_heads
        weights = np.random.standard_normal((self.hidden_size * 2, self.hidden_size * 8))
        bias = np.random.standard_normal((self.hidden_size * 8,))

        self.attention = Attention.get_torch_layer_with_weights(self.hidden_size * 2, head_num, weights, bias)
        self.gru = torch.nn.GRU(feature_dim, self.hidden_size, bidirectional=True)
        self.att_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.conv = nn.Conv2d(1, self.hidden_size, kernel_size=(3, self.hidden_size), padding=(1, 0), bias=False)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs):
        # embedding inputs [50,512]
        x = self.embedding(inputs)  # x [50,128,60]

        # gru & multi-head self-attention
        x = self.MHSFATT(x)  # [50,128,230]

        # cnn
        x = x.unsqueeze(1)  # [50,1,128,230]
        x = self.cnn(x)  # [50,230]

        return x

    def MHSFATT(self, x):
        x = x.transpose(0, 1)  # [128,50,60]
        x, hidden = self.gru(x)  # [128,50,460]
        x = x.transpose(0, 1).double()  # [50,128,460]
        x = self.attention(x, x, x).float()  # [50,128,460]
        x = self.att_linear(x)  # [50,128,230]
        return x

    def cnn(self, inputs):
        x = self.conv(inputs)  # [50,230,128，1]
        x = F.relu(x)
        x = x.squeeze(-1)  # [50,230,128]
        x = self.pool(x)  # [50,230,1]
        return x.squeeze(2)  # n x hidden_size
        # return x  # n x hidden_size x 1


# -------------for gru---------------------------------
class ContextPurification2DEncoder(nn.Module):  # for gru
    maml = False

    def __init__(self, block, word_vec_mat, list_of_num_layers, list_of_out_dims, max_length,
                 word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230, num_heads=2,
                 flatten=True, leakyrelu=False):
        # nn.Module.__init__(self)
        super(ContextPurification2DEncoder, self).__init__()
        self.grads = []
        self.fmaps = []
        self.final_feat_dim = 230
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = encoder.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)

        seq_len, feature_dim, head_num = max_length, word_embedding_dim + 2 * pos_embedding_dim, num_heads
        weights = np.random.standard_normal((self.hidden_size * 2, self.hidden_size * 8))
        bias = np.random.standard_normal((self.hidden_size * 8,))

        self.attention = Attention.get_torch_layer_with_weights(self.hidden_size * 2, head_num, weights, bias)
        # self.maxpool = nn.MaxPool1d(2, stride=2)
        # self.lstm = torch.nn.LSTM(feature_dim, feature_dim, bidirectional=True)
        self.gru = torch.nn.GRU(feature_dim, self.hidden_size, bidirectional=True)
        self.att_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        # self.conv = nn.Conv2d(1, self.hidden_size, kernel_size=(3, self.embedding_dim), padding=(1, 0),bias=False)
        # self.pool = nn.MaxPool1d(max_length)

        if self.maml:
            conv1 = Conv2d_ft(1, self.hidden_size, kernel_size=(3, self.hidden_size), stride=1, padding=(1, 0),
                              bias=False)
            bn1 = BatchNorm2d_ft(self.hidden_size)
        else:
            conv1 = nn.Conv2d(1, self.hidden_size, kernel_size=(3, self.hidden_size), padding=(1, 0), bias=False)
            bn1 = nn.BatchNorm2d(self.hidden_size)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        # self.final_pool = nn.AvgPool1d(2)
        self.final_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        init_layer(conv1)
        init_layer(bn1)

        trunk1 = [conv1, bn1, relu]
        trunk2 = []

        indim = 230

        for i in range(3):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
                trunk2.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d((self.max_length, 1))
            trunk2.append(avgpool)
            trunk2.append(Flatten())
            # trunk2.append(nn.AvgPool1d())

        self.trunk1 = nn.Sequential(*trunk1)
        self.trunk2 = nn.Sequential(*trunk2)

    def forward(self, inputs):
        # embedding inputs [50,512]
        x = self.embedding(inputs)  # x [50,128,60] [batch，128维，50+5+5（word 50维，每个pos5维）]

        # lstm & multi-head self-attention
        x = self.MHSFATT(x)  # [50,128,230]

        # cnn
        # x = self.cnn(x) # [50,230,128]
        # x = x.transpose(1,2)
        x = x.unsqueeze(1)  # [50,1,128,230]
        x = self.trunk1(x)  # [50，230，128，1]
        x = self.trunk2(x)  # [50,460]
        x = self.final_linear(x)  # [50,230]

        return x

    def MHSFATT(self, x):
        x = x.transpose(0, 1)  # [128,50,60]
        x, hidden = self.gru(x)  # [128,50,460]
        x = x.transpose(0, 1).double()  # [50,128,460]
        x = self.attention(x, x, x).float()  # [50,128,460]
        x = self.att_linear(x)  # [50,128,230]
        return x

    # def cnn(self, inputs):
    #   x = self.conv(inputs.transpose(1, 2))  # [50,230,128]
    #   x = F.relu(x)
    #   # x = self.pool(x) #[50,230,1]
    #   # return x.squeeze(2) # n x hidden_size
    #   return x  # n x hidden_size x 1


class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230, num_heads=4):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = encoder.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)
        seq_len, feature_dim, head_num = max_length, word_embedding_dim + 2 * pos_embedding_dim, num_heads
        weights = np.random.standard_normal((feature_dim, feature_dim * 4))
        bias = np.random.standard_normal((feature_dim * 4,))

        self.attention = Attention.get_torch_layer_with_weights(feature_dim, head_num, weights, bias)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.lstm = torch.nn.LSTM(feature_dim, feature_dim, bidirectional=True)
        self.encoder = encoder.encoder.Encoder(max_length, word_embedding_dim,
                                               pos_embedding_dim, hidden_size)

        # self.word2id = word2id
        self.final_feat_dim = 230

    def forward(self, inputs):  # inputs[batch_size,512]，一个样本有 word,pos1,pos2,mask,每一个是128维，4个拼成一行 所以是512维
        x = self.embedding(inputs)  # x [4,128,60] [batch，128维，50+5+5（word 50维，每个pos5维）]
        # x = x.transpose(0,1)  # embedding output [50,128,60], transpose out[128,50,60]
        # x,hidden = self.lstm(x)  # x [128,50.120] hidden tuple 每一个元素是[2,50,60]
        # x = x.transpose(0,1).double()  # [50,128,120]
        # x = self.maxpool(x)  # [50,128,60]

        return x



def ContextCNN(flatten=True, leakyrelu=False):  # 如何使用flatten
    try:
        glove_mat = np.load('./glove/glove_mat.npy')
        # glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")

    return ContextPurificationEncoder(OneDBlock, glove_mat, [1, 1, 1, 1], [230, 230, 230, 230], max_length=128,
                                      word_embedding_dim=50,
                                      pos_embedding_dim=5, hidden_size=230, num_heads=2,
                                      flatten=True, leakyrelu=False)


def Context2DCNN(flatten=True, leakyrelu=False):  # 如何使用flatten
    try:
        glove_mat = np.load('./glove/glove_mat.npy')
        # glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")

    return ContextPurification2DEncoder(TwoDBlock, glove_mat, [1, 1, 1, 1], [460, 460, 460, 230], max_length=128,
                                        word_embedding_dim=50,
                                        pos_embedding_dim=5, hidden_size=230, num_heads=2,
                                        flatten=True, leakyrelu=False)


def OneCNN(flatten=True, leakyrelu=False):  # 如何使用flatten
    try:
        glove_mat = np.load('./glove/glove_mat.npy')
        # glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")

    return CNNSentenceEncoder(glove_mat, max_length=128, word_embedding_dim=50,
                              pos_embedding_dim=5, hidden_size=230)


def TwoDCNN(flatten=True, leakyrelu=False):
    try:
        glove_mat = np.load('./glove/glove_mat.npy')
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")

    return TwoDCNNEncoder(glove_mat, max_length=128, word_embedding_dim=50,
                          pos_embedding_dim=5, hidden_size=230, flatten=True, leakyrelu=False)


def ATTCNN(flatten=True, leakyrelu=False):
    try:
        glove_mat = np.load('./glove/glove_mat.npy')
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")

    return ATTCNNEncoder(glove_mat, max_length=128, word_embedding_dim=50,
                         pos_embedding_dim=5, hidden_size=230, num_heads=2, flatten=True, leakyrelu=False)


model_dict = dict(Context2DCnn=Context2DCNN,
                  OnlyCnn=TwoDCNN,
                  AttCnn=ATTCNN,
                  Context1DCnn=ContextCNN)
# if __name__=='__main__':
