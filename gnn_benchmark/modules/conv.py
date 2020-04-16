from torch import nn
from torch_geometric import nn as geom_nn
import torch
from torch_scatter import scatter_mean
from gnn_benchmark.common.definitions import Channels


class SimpleConvWrapper(nn.Module):
    conv_class = None
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self.conv_class(in_channels=self.in_channels.node, out_channels=self.out_channels.node)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        return data


class GCNConv(SimpleConvWrapper):
    conv_class = geom_nn.GCNConv


class GATConv(SimpleConvWrapper):
    conv_class = geom_nn.GATConv


class SAGEConv(SimpleConvWrapper):
    conv_class = geom_nn.SAGEConv


class MLPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Linear(in_features=self.in_channels.node, out_features=self.out_channels.node)

    def forward(self, data):
        data.x = self.conv(data.x)
        return data


class GINConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, n_layers, n_hiddens, train_eps):
        super().__init__()
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.train_eps = train_eps
        self.out_channels = out_channels
        nn_layers = []
        last_dim = in_channels.node
        for i in range(self.n_layers - 1):
            nn_layers.append(nn.Linear(last_dim, self.n_hiddens))
            last_dim = self.n_hiddens
            nn_layers.append(nn.ReLU())
        nn_layers.append(nn.Linear(last_dim, out_channels.node))
        net = nn.Sequential(*nn_layers)
        self.conv = geom_nn.GINConv(net, train_eps=self.train_eps)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        return data


class EdgeModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_hiddens, n_layers, residuals):
        super().__init__()
        self.residuals = residuals
        self._in_channels = in_channels
        self._out_channels = out_channels
        assert n_layers > 1
        mlp_layers = []
        mlp_layers.append(nn.Linear(2 * in_channels.node + in_channels.edge, n_hiddens))
        mlp_layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            mlp_layers.append(nn.Linear(n_hiddens, n_hiddens))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(n_hiddens, out_channels.edge))
        self.edge_mlp = nn.Sequential(*mlp_layers)


    @property
    def out_channels(self):
        return Channels(
            node=self._in_channels.node,
            edge=self._out_channels.edge,
            graph=self._in_channels.graph
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            if out.shape[1] < edge_attr.shape[1]:
                # handling downscaling
                edge_attr = edge_attr[:, :out.shape[1]]
            elif out.shape[1] > edge_attr.shape[1]:
                # and handling upscaling
                new_zeros = torch.zeros_like(out)
                new_zeros[:, :edge_attr.shape[1]] = edge_attr
                edge_attr = new_zeros
            out = out + edge_attr
        return out


class NodeModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_hiddens, n_layers, residuals):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        assert n_layers > 1
        mlp_1_layers = []
        mlp_1_layers.append(nn.Linear(in_channels.node + in_channels.edge, n_hiddens))
        mlp_1_layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            mlp_1_layers.append(nn.Linear(n_hiddens, n_hiddens))
            mlp_1_layers.append(nn.ReLU())
        mlp_1_layers.append(nn.Linear(n_hiddens, n_hiddens))
        self.node_mlp_1 = nn.Sequential(*mlp_1_layers)

        mlp_2_layers = []
        mlp_2_layers.append(nn.Linear(n_hiddens + in_channels.node, n_hiddens))
        mlp_2_layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            mlp_2_layers.append(nn.Linear(n_hiddens, n_hiddens))
            mlp_2_layers.append(nn.ReLU())
        mlp_2_layers.append(nn.Linear(n_hiddens, out_channels.node))
        self.node_mlp_2 = nn.Sequential(*mlp_2_layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)
        if self.residuals:
            if out.shape[1] < x.shape[1]:
                # handling downscaling
                x = x[:, :out.shape[1]]
            elif out.shape[1] > x.shape[1]:
                # and handling upscaling
                new_zeros = torch.zeros_like(out)
                new_zeros[:, :x.shape[1]] = x
                x = new_zeros
            out = out + x
        return out


class MetaConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_hiddens, n_layers, residuals):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        edge_model = EdgeModel(in_channels, out_channels, n_hiddens, n_layers=n_layers, residuals=residuals)
        node_model = NodeModel(edge_model.out_channels, out_channels, n_hiddens, n_layers=n_layers, residuals=residuals)
        self.conv = geom_nn.MetaLayer(
            edge_model=edge_model,
            node_model=node_model
        )

    def forward(self, data):
        data.x, data.edge_attr, _ = self.conv(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, u=None)
        return data

