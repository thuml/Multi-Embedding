import torch
import torch.nn.functional as F

from torchfm.layer import CompressedInteractionNetwork, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class CIN(torch.nn.Module):
    """
    CIN w/o final linear layer
    """

    def __init__(self, input_dim, cross_layer_sizes):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.conv_layers = torch.nn.ModuleList()
        self.output_dim = 0
        prev_dim = input_dim
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            # (B, N x C, K) -> (B, C', K)
            prev_dim = cross_layer_size
            self.output_dim += prev_dim

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        # x0: (B, N, 1, K)
        # h: (B, C, K)
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            h = x
            xs.append(x)
        return torch.sum(torch.cat(xs, dim=1), 2)


class XDeepFM(torch.nn.Module):
    """
    XDeepFM w/o Linear
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CIN(len(field_dims), cross_layer_sizes)
        self.cin_post = torch.nn.Linear(self.cin.output_dim, 1)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout) if mlp_dims else None

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.cin_post(self.cin(embed_x))
        if self.mlp:
            x += self.mlp(embed_x.flatten(1))
        return torch.sigmoid(x.squeeze(1))


class MultiXDeepFM(torch.nn.Module):

    def __init__(self, field_dims, embed_dims, mlp_dims, dropout, cross_layer_sizes):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.cins = torch.nn.ModuleList([CIN(len(field_dims), cross_layer_sizes) for _ in range(len(embed_dims))])
        self.cin_post = torch.nn.Linear(self.cins[0].output_dim, 1)
        self.mlps = torch.nn.ModuleList([
            MultiLayerPerceptron(embed_dim * len(field_dims), (mlp_dims[0],), dropout, output_layer=False)
            for embed_dim in embed_dims
        ]) if mlp_dims else None
        self.mlp_post = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout) if mlp_dims else None

    def forward(self, x):
        embs = [embedding(x) for embedding in self.embeddings]
        cin_feature = torch.stack([cin(emb) for cin, emb in zip(self.cins, embs)], dim=-1).mean(dim=-1)
        x = self.cin_post(cin_feature)
        if self.mlps:
            mlp_feature = torch.stack([mlp(emb.flatten(1)) for mlp, emb in zip(self.mlps, embs)], dim=-1).mean(dim=-1)
            x += self.mlp_post(mlp_feature)
        return torch.sigmoid(x.squeeze(1))
