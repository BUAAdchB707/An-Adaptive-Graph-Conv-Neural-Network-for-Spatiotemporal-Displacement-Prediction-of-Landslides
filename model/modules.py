import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class ChannelSharedEncoder(nn.Module):
    """
    = main variate
    """

    def __init__(self, n_channel, in_len, in_dim, embed_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange('b n d l -> b d n l'),
            nn.Conv2d(in_dim, embed_dim, kernel_size=(1, 3),
                      padding=(0, 1), padding_mode='replicate', bias=False),
            Rearrange('b d n l -> b n d l')
        )
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.encoder(x)
        return y


class ChannelWiseEncoder(nn.Module):
    """
    = co-variate
    x [b channel input_dim length] [b n d l]
    Return
    embedding [b channel latent_dim length] [b n d' l]
    """

    def __init__(self, n_channel, in_len, in_dim, embed_dim, dropout):
        super().__init__()
        self.n_channel = n_channel
        self.mlp = nn.ModuleList([
            nn.Sequential(
                Rearrange('b l d -> b d l'),
                nn.Conv1d(in_dim, embed_dim, kernel_size=3,
                          padding=1, padding_mode='replicate', bias=False),
                Rearrange('b d l -> b l d'),
            )
            for _ in range(n_channel)
        ])

    def forward(self, x):
        assert x.shape[1] == self.n_channel
        x = rearrange(x, 'b n d l -> n b l d')
        y = torch.stack([self.mlp[i](x[i]) for i in range(self.n_channel)])
        y = rearrange(y, 'n b l d -> b n d l')
        return y


class DGraphLearner(nn.Module):
    """
    return:
    A [n n]
    """
    def __init__(self, num_nodes, in_dim, node_dim, dropout, predefined=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.sta_vec = nn.Parameter(torch.randn(2, num_nodes, node_dim))
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.predefined = predefined
        self.norm = nn.LayerNorm([num_nodes], elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)

    def get_predefined_A(self):
        A = torch.tensor(self.predefined, dtype=self.alpha.dtype, device=self.alpha.device)
        A = torch.clip(A, min=0., max=1.)
        return A.detach()

    def get_sta_adm(self, activ=True):
        if self.predefined is not None:
            return self.get_predefined_A()

        vec_0 = self.sta_vec[0]
        vec_1 = self.sta_vec[1]
        sta_adm = torch.matmul(vec_0, vec_1.transpose(-1, -2))
        sta_adm = sta_adm + sta_adm.transpose(-1, -2)
        if activ:
            A = F.sigmoid(self.alpha * self.norm(sta_adm))
        else:
            A = F.sigmoid(self.alpha.detach() * self.norm(sta_adm))

        A = A * (1. - torch.eye(self.num_nodes, device=A.device))
        A = A + torch.eye(self.num_nodes, device=A.device)
        return A

    def forward(self, x, **kwargs):
        A = self.get_sta_adm()
        A = norm_adj(A)
        return A


class TimeDynGraph(DGraphLearner):
    def __init__(self, num_nodes, in_dim, node_dim, dropout, predefined=None):
        super().__init__(num_nodes, in_dim, node_dim, dropout, predefined)

    def get_sta_adm(self, activ=True):
        if self.predefined is not None:
            return self.get_predefined_A()

        sta_vec = self.sta_vec[0, :, 0]
        sta_vec = torch.sort(sta_vec).values

        if activ:
            sta_vec = F.sigmoid(self.alpha * self.norm(sta_vec))
        else:
            sta_vec = F.sigmoid(self.alpha.detach() * self.norm(sta_vec))

        sta_adm = []
        for i in range(self.num_nodes):
            sta_adm.append(torch.concatenate([sta_vec[-i - 1:], sta_vec[:-i - 1], ], dim=-1))
        A = torch.stack(sta_adm)
        A = torch.tril(A)

        A = A * (1. - torch.eye(self.num_nodes, device=A.device))
        A = A + torch.eye(self.num_nodes, device=A.device)
        return A


def norm_adj(adm, add_i=0, eps=1e-5):
    if add_i != 0:
        adm = adm + add_i * torch.eye(adm.shape[-1], device=adm.device)
    adm = torch.clip(adm, min=0., max=1.)

    batch_shape, A_shape = adm.shape, adm.shape[-2:]
    deg_out = torch.sum(torch.reshape(adm, [-1] + list(A_shape)), dim=-1)
    # deg_out[deg_out == 0.] = 1.

    deg_out_inv = torch.stack([torch.diag(torch.reciprocal(deg_o + eps)) for deg_o in deg_out])
    deg_out_inv = torch.reshape(deg_out_inv, batch_shape)
    A = torch.matmul(deg_out_inv, adm)
    return A


class NodeGConv(nn.Module):
    def __init__(self, ful_node, num_nodes, in_dim, in_len, latent_dim, out_dim, node_dim, dropout, valid, pre_adm):
        super().__init__()
        self.ful_node = ful_node
        self.valid = valid
        self.pre_adm = pre_adm
        self.num_nodes = num_nodes
        self.in_linear = nn.Sequential(
            Rearrange('b n d l -> b d n l'),
            nn.Conv2d(in_dim, latent_dim, kernel_size=(1, in_len), bias=False),
            Rearrange('b d n l -> b n (d l)'),
        )
        self.out_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, in_dim * in_len, bias=False),
            Rearrange('b n (d l) -> b n d l', d=in_dim)
        )
        self.weight = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim, bias=False)
            for _ in range(2)
        ])

    def forward(self, x, node_graph):
        """
        x         [b n d l]
        Return:
        y         [b n d l]
        """
        A = node_graph(x)

        if not self.valid:
            H = x
        else:
            x = self.in_linear(x)
            if self.ful_node != self.num_nodes and self.pre_adm is None:
                A_0 = A[..., :self.num_nodes]
                A_1 = A[..., self.num_nodes:]

                x_0 = x[:, :self.num_nodes]
                x_1 = x[:, self.num_nodes:]

                H0 = torch.einsum('nm, bm... -> bn...', A_0, x_0.detach()).contiguous()
                H1 = torch.einsum('nm, bm... -> bn...', A_1, x_1.detach()).contiguous()
                H = self.weight[0](H0) + self.weight[1](H1)
            else:
                H = torch.einsum('nm, bm... -> bn...', A, x.detach()).contiguous()
                H = self.weight[0](H)
            H = torch.concatenate([x, H], dim=-1)
            H = self.out_linear(H)
        return H


class TimeGConv(nn.Module):
    def __init__(self, ful_node, num_nodes, in_dim, in_len, latent_dim, node_dim, dropout, valid):
        super().__init__()
        self.valid = valid
        self.num_nodes = num_nodes
        self.in_linear = nn.Sequential(
            Rearrange('b n d l -> b d l n'),
            nn.Conv2d(in_dim, latent_dim, kernel_size=(1, ful_node), bias=False),
            Rearrange('b d l n -> b l (d n)'),
        )
        self.out_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, ful_node * in_dim, bias=False),
            Rearrange('b l (n d) -> b n d l', d=in_dim)
        )
        self.weight = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x, time_graph):
        """
        x         [b n d l]
        Return:
        y         [b n d l]
        """
        A = time_graph(x)

        if not self.valid:
            H = x
        else:
            x = self.in_linear(x)
            if A.ndim != 2:
                H = torch.einsum('bnhl, bndl -> bndh', A, x.detach()).contiguous()
            else:
                H = torch.einsum('hl, bld -> bhd', A, x.detach()).contiguous()
            H = self.weight(H)
            H = torch.concatenate([x, H], dim=-1)
            H = self.out_linear(H)
        return H

class Memory(nn.Module):
    def __init__(self, size, num_var=1, rate=0.5):
        super().__init__()
        self.size = size
        self.n = num_var
        self.queue = [[] for _ in range(num_var)]
        self.rate = rate

    def __len__(self):
        return len(self.queue[0])

    def add_item(self, x, i=0):
        self.queue[i].append(x)
        while len(self.queue[i]) > self.size:
            self.queue[i].pop(0)

    def get_mem(self, i=0):
        part = torch.stack(self.queue[i])
        return torch.mean(part, 0)

    def sample(self, i=0):
        import random
        k = max(1, int(self.rate * len(self.queue[i])))
        part = torch.stack(random.sample(self.queue[i], k=k))
        return torch.mean(part, 0)
