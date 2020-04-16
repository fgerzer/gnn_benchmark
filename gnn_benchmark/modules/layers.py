from torch import nn
from torch.nn import functional as F
from gnn_benchmark.common.definitions import Channels


class GraphBatchNorm(nn.Module):
    def __init__(self, channels: Channels):
        super().__init__()
        def set_maybe(c):
            if c is not None:
                return nn.BatchNorm1d(c)
            else:
                return None
        self.bn_node = set_maybe(channels.node)
        self.bn_edge = set_maybe(channels.edge)
        self.bn_graph = set_maybe(channels.graph)

    def forward(self, data):
        if self.bn_node:
            data.x = self.bn_node(data.x)
        if self.bn_edge:
            data.edge_attr = self.bn_edge(data.edge_attr)
        if self.bn_graph:
            data.u = self.bn_graph(data.u)
        return data


class ReLU(nn.Module):
    def __init__(self, apply_nodes=True, apply_edges=True, apply_graph=True):
        super().__init__()
        self.apply_nodes = apply_nodes
        self.apply_edges = apply_edges
        self.apply_graph = apply_graph

    def forward(self, data):
        if self.apply_nodes and "x" in data:
            data.x = F.relu(data.x)
        if self.apply_edges and "edge_attr" in data:
            data.edge_attr = F.relu(data.edge_attr)
        if self.apply_graph and "u" in data:
            data.u = F.relu(data.u)
        return data


class Dropout(nn.Module):
    def __init__(self, p, apply_nodes=True, apply_edges=True, apply_graph=True):
        super().__init__()
        self.p = p
        self.apply_nodes = apply_nodes
        self.apply_edges = apply_edges
        self.apply_graph = apply_graph

    def forward(self, data):
        if self.apply_nodes and "x" in data:
            data.x = F.dropout(data.x, p=self.p, training=self.training)
        if self.apply_edges and "edge_attr" in data:
            data.edge_attr = F.dropout(data.edge_attr, p=self.p, training=self.training)
        if self.apply_graph and "u" in data:
            data.u = F.dropout(data.u, p=self.p, training=self.training)
        return data
