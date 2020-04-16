import torch


class GNNBlockNetGraph(torch.nn.Module):
    def __init__(self, encoder, blocks, global_pooling, output_layer):
        super().__init__()
        self.encoder = encoder
        self.blocks = blocks
        self.output_layer = output_layer
        self.global_pooling = global_pooling

    def forward(self, data):
        data = self.encoder(data)
        for block in self.blocks:
            data = block(data)
        x = self.global_pooling(data)
        x = self.output_layer(x)
        return x


class GNNBlockNetNode(torch.nn.Module):
    def __init__(self, encoder, blocks, output_layer):
        super().__init__()
        self.encoder = encoder
        self.blocks = blocks
        self.output_layer = output_layer

    def forward(self, data):
        data = self.encoder(data)
        for block in self.blocks:
            data = block(data)
        x = self.output_layer(data.x)
        return x
