from torch_geometric import nn as geom_nn
from torch import nn
import abc


class Pooling(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.pool_layer = self.build_pool(in_channels, **kwargs)

    def build_pool(self, in_channels, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, data):
        # return x, edge_index, batch
        pass

    def reset_parameters(self):
        pass


class TopKPooling(Pooling):
    def build_pool(self, in_channels, ratio=0.5, min_score=None, multiplier=1):
        return geom_nn.TopKPooling(in_channels.node, ratio, min_score, multiplier)

    def forward(self, data):
        data.x, data.edge_index, data.edge_attr, data.batch, _, _ = self.pool_layer(data.x, data.edge_index,
                                                   batch=data.batch, edge_attr=data.edge_attr)
        return data

    def reset_parameters(self):
        self.pool_layer.reset_parameters()


class SortPooling(Pooling):
    def forward(self, x, edge_index, batch):
        # TODO: Requires to remove edges
        geom_nn.global_sort_pool()


class DiffPooling(Pooling):
    pass
    # TODO: Requires some way of dealing with the density matrix - probably another base class?


class DenseMinCutPooling(Pooling):
    # TODO: As DiffPool, requires dense matrices (also, auxilliary loss)
    pass


class GraclusPooling(Pooling):
    def forward(self, data):
        cluster = geom_nn.graclus(data.edge_index, num_nodes=data.x.size(0))
        data = geom_nn.max_pool(cluster, data)
        return data


class SAGPooling(Pooling):
    def build_pool(self, in_channels, ratio=0.5, min_score=None, multiplier=1):
        return geom_nn.SAGPooling(in_channels.node, ratio, min_score=min_score, multiplier=multiplier)

    def forward(self, data):
        data.x, data.edge_index, data.edge_attr, data.batch, _, _ = self.pool_layer(data.x, data.edge_index,
                                                   batch=data.batch, edge_attr=data.edge_attr)
        return data


class EdgePooling(Pooling):
    def build_pool(self, in_channels, edge_score_method=None, dropout=0, add_to_edge_score=0.5):
        return geom_nn.EdgePooling(in_channels.node, edge_score_method, dropout, add_to_edge_score)

    def forward(self, data):
        if "edge_attr" in data:
            # EdgePooling cannot deal with edge_attr
            del data.edge_attr
        data.x, data.edge_index, data.batch,_ = self.pool_layer(data.x, data.edge_index, batch=data.batch)
        return data
