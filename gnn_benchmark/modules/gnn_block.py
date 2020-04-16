from torch import nn
from gnn_benchmark.modules.layers import GraphBatchNorm, ReLU, Dropout
from torch_geometric import nn as geom_nn

class GNNBlock(nn.Module):
    def __init__(self, conv, non_linearity=ReLU(), p_dropout=0., batch_norm=False):
        super().__init__()
        self.conv = conv
        self.non_linearity = non_linearity
        self.dropout = Dropout(p=p_dropout)
        if batch_norm:
            self.batch_norm = GraphBatchNorm(conv.out_channels)
        else:
            self.batch_norm = None

    def forward(self, data):
        data = self.conv(data)
        data = self.non_linearity(data)
        data = self.dropout(data)
        if self.batch_norm:
            data = self.batch_norm(data)
        return data

    @property
    def out_channels(self):
        return self.conv.out_channels


class JumpingBlocks(nn.Module):
    def __init__(self, blocks, jumping_mode):
        super().__init__()
        self.blocks = blocks
        self.jump = geom_nn.JumpingKnowledge(jumping_mode) if jumping_mode is not None else None

    def forward(self, data):
        xs = []
        for block in self.blocks:
            data = block(data)
            xs += [data.x]
        if self.jump is not None:
            data.x = self.jump(xs)
        return data

    @property
    def out_channels(self):
        last_channels = self.blocks[-1].out_channels
        if self.jump is not None and self.jump.mode == "cat":
            last_channels.node = last_channels.node * len(self.blocks)
        return last_channels
