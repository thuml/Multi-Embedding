import torch

from torchfm.layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding


class SingleHeadDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x


class MultiHeadWideAndDeepModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dims, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embeddings = torch.nn.ModuleList([SingleHeadDeepModel(field_dims, embed_dim, mlp_dims, dropout) for embed_dim in embed_dims])
        self.mlp = torch.nn.Linear(mlp_dims[-1], 1)
        self.weights = torch.nn.Parameter(torch.ones(len(embed_dims)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = torch.stack([embedding(x) for embedding in self.embeddings], dim=-1)
        embed_x = torch.sum(embed_x * self.weights.unsqueeze(0).unsqueeze(0), dim=-1)
        x = self.linear(x) + self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))


class MultiHeadWideAndDeepModelNoLinear(torch.nn.Module):
    def __init__(self, field_dims, embed_dims, mlp_dims, dropout):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([SingleHeadDeepModel(field_dims, embed_dim, mlp_dims, dropout) for embed_dim in embed_dims])
        self.mlp = torch.nn.Linear(mlp_dims[-1], 1)
        self.weights = torch.nn.Parameter(torch.ones(len(embed_dims)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = torch.stack([embedding(x) for embedding in self.embeddings], dim=-1)
        embed_x = torch.sum(embed_x * self.weights.unsqueeze(0).unsqueeze(0), dim=-1)
        x = self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))


class SharedEmbeddingMWDNoLinear(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, num_mlps, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlps = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False) for _ in range(num_mlps)])
        self.head = torch.nn.Linear(mlp_dims[-1], 1)
    

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        feat_x = torch.stack([mlp(embed_x) for mlp in self.mlps], dim=-1)
        feat_x = feat_x.mean(dim=-1)
        x = self.head(feat_x)
        return torch.sigmoid(x.squeeze(1))


"""
2023.08.03
"""
class DNNModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class MultiDNNModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dims, mlp_dims, dropout):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Sequential(
                FeaturesEmbedding(field_dims, embed_dim),
                torch.nn.Flatten(start_dim=1),
                MultiLayerPerceptron(embed_dim * len(field_dims), (mlp_dims[0],), dropout, output_layer=False)
            ) for embed_dim in embed_dims])
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_xs = torch.stack([embedding(x) for embedding in self.embeddings], dim=-1)
        embed_x = embed_xs.mean(dim=-1)
        x = self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))
