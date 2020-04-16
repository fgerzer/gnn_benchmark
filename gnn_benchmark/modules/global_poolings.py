from torch_geometric import nn as geom_nn
from torch import nn
import abc
import torch
from gnn_benchmark.common.definitions import Channels


class GlobalPooling(nn.Module, metaclass=abc.ABCMeta):
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

    @property
    def out_channels(self):
        return Channels(graph=self.in_channels.node)


class GlobalMeanPooling(GlobalPooling):
    def forward(self, data):
        x = geom_nn.global_mean_pool(data.x, data.batch)
        return x


class GlobalMaxPooling(GlobalPooling):
    def forward(self, data):
        x = geom_nn.global_max_pool(data.x, data.batch)
        return x


class GlobalAddPooling(GlobalPooling):
    def forward(self, data):
        x = geom_nn.global_add_pool(data.x, data.batch)
        return x


class GlobalAllPooling(GlobalPooling):
    def forward(self, data):
        x_mean = geom_nn.global_mean_pool(data.x, data.batch)
        x_max = geom_nn.global_max_pool(data.x, data.batch)
        x_add = geom_nn.global_add_pool(data.x, data.batch)
        return torch.cat([x_mean, x_max, x_add], dim=1)

    @property
    def out_channels(self):
        return Channels(graph=3 * self.in_channels.node)


class GlobalSortPooling(GlobalPooling):
    def __init__(self, in_channels, k):
        super().__init__(in_channels)
        self.k = k

    def forward(self, data):
        x = geom_nn.global_sort_pool(data.x, data.batch, self.k)
        return x

    @property
    def out_channels(self):
        return Channels(graph=self.k * self.in_channels.node)


class GlobalSet2SetPooling(GlobalPooling):
    # TODO build with real in_channels
    def build_pool(self, in_channels, processing_steps=4, num_layers=1):
        return geom_nn.Set2Set(in_channels, processing_steps, num_layers)

    def forward(self, data):
        x = self.pool_layer(data.x, data.batch)
        return x

    @property
    def out_channels(self):
        # TODO
        return Channels(graph=self.pool_layer.out_channels)


class GlobalAttentionPooling(GlobalPooling):
    # TODO build with real in_channels
    def build_pool(self, in_channels):
        return geom_nn.GlobalAttention(nn.Linear(in_channels.node, 1))

    def forward(self, data):
        return self.pool_layer(data.x, data.batch)
