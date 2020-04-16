from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from gnn_benchmark.common.definitions import Channels


class LinearEncoder(nn.Module):
    def __init__(self, in_channels: Channels, out_channels: Channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        if out_channels.edge is not None:
            self.edge_model = nn.Linear(in_features=in_channels.edge, out_features=out_channels.edge)
        else:
            self.edge_model = None

        if out_channels.node is not None:
            self.node_model = nn.Linear(in_features=in_channels.node, out_features=out_channels.node)
        else:
            self.node_model = None

    def forward(self, data):
        if self.node_model is not None:
            data.x = self.node_model(data.x)
        if self.edge_model is not None:
            data.edge_attr = self.edge_model(data.edge_attr)
        return data


class MolEncoder(nn.Module):
    def __init__(self, out_channels: Channels, encode_atom=True, encode_bond=True):
        super().__init__()
        self.out_channels = out_channels
        self.atom_encoder = AtomEncoder(out_channels.node) if encode_atom else None
        self.bond_encoder = BondEncoder(out_channels.edge) if encode_bond else None


    def forward(self, data):
        if self.atom_encoder is not None:
            data.x = self.atom_encoder(data.x)
        if self.bond_encoder is not None:
            data.edge_attr = self.bond_encoder(data.edge_attr)
        return data
