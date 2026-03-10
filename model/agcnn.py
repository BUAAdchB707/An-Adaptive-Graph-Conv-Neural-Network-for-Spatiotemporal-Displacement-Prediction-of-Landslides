import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .modules import (ChannelWiseEncoder, ChannelSharedEncoder, NodeGConv, TimeGConv,
                      DGraphLearner, TimeDynGraph, Memory)
from tensorboardX import SummaryWriter


class AGCNN(nn.Module):
    """
    Args:
        n_node: Number of monitoring points
        n_dim: Feature dimension of monitoring points
        in_len: Input lookback length
        out_len: Output forcasting length
        depth: Number of layer
        embed_dim: Embedding feature dimension
        latent_dim: Latent feature dimension of graph convolution module
        node_dim: Feature dimension of graph learning module
        node_gc: Whether spatial graph convolution module work
        time_gc: Whether temporal graph convolution module work
        dyn_c: Number of external influencing factors
        dyn_d: Feature dimension of external influencing factors
        dropout: dropout rate
        pre_adm: predefined spatial graph adjacency matrix
    """

    def __init__(self, n_node, n_dim, in_len, out_len,
                 depth, embed_dim, latent_dim, node_dim, node_gc=True, time_gc=True,
                 dyn_c=0, dyn_d=0, dropout=0.1, pre_adm=None, **kwargs):
        super().__init__()
        self.ts_embed = ChannelSharedEncoder(n_node, in_len, n_dim, embed_dim, dropout)
        self.dyn_embed = ChannelWiseEncoder(dyn_c, in_len, dyn_d, embed_dim, dropout)
        self.out_len = out_len
        self.n_node = n_node
        self.dyn_c = dyn_c
        self.ful_c = n_node + dyn_c

        self.main = Graph2(self.ful_c, n_node, embed_dim, in_len, out_len, latent_dim, depth,
                           node_dim, dropout, node_gc, time_gc, pre_adm)

        self.out_mlp = nn.Sequential(
            nn.Linear(in_len, out_len),
            Rearrange('... d l -> ... l d'),
            nn.Linear(embed_dim, n_dim),
            Rearrange('... l d -> ... d l'),
        )
        self.train_memory = Memory(100, 2, 0.5)

    def get_graph(self, activ=True):
        node_graph = self.main.node_graph.get_sta_adm(activ)
        time_graph = self.main.time_graph.get_sta_adm(activ)
        return node_graph, time_graph

    def forward(self, ts, dyn=None, logger: SummaryWriter = None, step=None, **kwargs):
        means = ts.mean(-1, keepdim=True).detach()
        ts = ts - means
        stdev = torch.sqrt(torch.var(ts, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        ts = ts / stdev
        # t_last = ts[..., -1:].detach()
        # ts = ts - t_last

        ts_embed = self.ts_embed(ts)
        ms_embed = [ts_embed]
        if self.dyn_c > 0:
            assert dyn is not None
            ms_embed.append(self.dyn_embed(dyn))
        ms_embed = torch.concatenate(ms_embed, dim=1)
        assert ms_embed.shape[1] == self.ful_c
        y_embed = self.main(ms_embed)

        y = self.out_mlp(y_embed[:, :self.n_node, ...])

        y = y * stdev + means
        # y = y + t_last

        if self.training and logger is not None and step % 500 == 0:
            logger.add_histogram('time_series_embed', ts_embed, step)
            logger.add_histogram('multi_source_embed', ms_embed, step)
            logger.add_histogram('predict_y_embed', y_embed, step)

        if self.training:
            node_g, time_g = self.get_graph(False)

            nf = rearrange(ms_embed, 'b n d l -> b n (d l)')
            nf = F.normalize(nf, p=2, dim=-1)
            cossim = torch.matmul(nf, nf.transpose(-1, -2))
            self.train_memory.add_item(cossim.mean(0), 0)
            mot_ng = self.train_memory.sample(0).detach()
            mot_ng, node_g = rm_diag(mot_ng)[:self.n_node], rm_diag(node_g)[:self.n_node]
            e_d = torch.abs(norm_dim(mot_ng[:, :self.n_node - 1],) - norm_dim(node_g[:, :self.n_node - 1],))
            e_a = torch.abs(norm_dim(mot_ng[:, self.n_node - 1:],) - norm_dim(node_g[:, self.n_node - 1:],))
            ssl_n = torch.mean(torch.cat([e_d, e_a], dim=-1))

            tf = rearrange(ms_embed, 'b n d l -> b l (n d)')
            tf = F.normalize(tf, p=2, dim=-1)
            cossim = torch.matmul(tf, tf.transpose(-1, -2))
            self.train_memory.add_item(cossim.mean(0)[-1], 1)
            mot_tg = self.train_memory.sample(1).detach()
            ssl_t = torch.mean(torch.abs(norm_dim(mot_tg) - norm_dim(time_g[-1])))

            ssl = (ssl_t + ssl_n) * len(self.train_memory) / self.train_memory.size
            return {'y': y, 'loss': ssl}

        return y

def rm_diag(x):
    n = x.shape[-1]
    mask = ~ torch.eye(n, device=x.device, dtype=torch.bool)
    x = x[mask].view(n, n - 1)
    return x

def norm_dim(x, dim=-1):
    # if x.shape[dim] == 1:
    #     return x
    # mean = torch.mean(x, dim=dim, keepdim=True)
    # std = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False) + 1e-5)
    # return (x - mean) / std
    return torch.layer_norm(x, normalized_shape=x.shape[dim:])


class G2Layer(nn.Module):
    def __init__(self, ful_node, num_nodes, in_dim, in_len, latent_dim, node_dim,
                 dropout, node_gc, time_gc):
        super().__init__()

        self.node_conv = NodeGConv(ful_node, num_nodes, in_dim, in_len, latent_dim, in_dim, node_dim,
                                   dropout, node_gc)
        self.time_conv = TimeGConv(ful_node, num_nodes, in_dim, in_len, latent_dim, node_dim,
                                   dropout, time_gc)

        self.post_norm = nn.LayerNorm([ful_node, in_dim, in_len])

    def forward(self, x, node_graph, time_graph):
        """
        x [b n d l]
        return:
        y [b n d l]
        """

        node_embed = self.node_conv(x, node_graph)
        x = self.post_norm(x + node_embed)
        time_embed = self.time_conv(x, time_graph)
        x = self.post_norm(x + time_embed)

        return x, node_embed, time_embed


class Graph2(nn.Module):
    def __init__(self, ful_node, num_node, in_dim, in_len, out_len, latent_dim, depth,
                 node_dim, dropout, node_gc, time_gc, pre_adm):
        super().__init__()
        self.depth = depth
        self.n_node = num_node

        self.node_graph = DGraphLearner(ful_node, in_dim, node_dim, dropout, pre_adm)
        self.time_graph = TimeDynGraph(in_len, in_dim, node_dim, dropout)

        self.layers = nn.ModuleList([
            G2Layer(ful_node, num_node, in_dim, in_len, latent_dim, node_dim, dropout,
                    node_gc, time_gc)
            for _ in range(depth)
        ])

        self.skip_out = nn.Sequential(
            Rearrange('b n d l -> b n l d'),
            nn.Dropout(dropout),
            nn.Linear(in_dim * depth * 2, in_dim, bias=False),
            nn.ReLU(),
            Rearrange('b n l d -> b n d l')
        )

    def forward(self, x):
        y = []
        for i in range(self.depth):
            x, x_skip_n, x_skip_t = self.layers[i](
                x, self.node_graph, self.time_graph)
            y.append(x_skip_n)
            y.append(x_skip_t)
        y = torch.concatenate(y, dim=-2)
        y = self.skip_out(y)
        return y


def _testnet():
    print()
    n_node = 12
    in_len = 12
    out_len = 6
    n_dim = 1
    n_dyn = 3
    dyn_dim = 1

    model = AGCNN(n_node=n_node, n_dim=n_dim, in_len=in_len, out_len=out_len,
                  depth=2, embed_dim=16, latent_dim=64, node_dim=32, dropout=0.1,
                  dyn_c=n_dyn, dyn_d=dyn_dim)

    x = torch.rand([1, n_node, n_dim, in_len])
    dyn = torch.rand([1, n_dyn, dyn_dim, in_len])
    y = model(x, dyn=dyn)
    print(y.shape)

if __name__ == '__main__':
    _testnet()
