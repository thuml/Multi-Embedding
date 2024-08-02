import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear


class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class NeuralFactorizationMachineModelNoLinear(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.fm(self.embedding(x))
        x = self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class MultiNFMModelNoLinear(torch.nn.Module):

    def __init__(self, field_dims, embed_dims, mlp_dims, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.fms = torch.nn.ModuleList([
            torch.nn.Sequential(
                FactorizationMachine(reduce_sum=False),
                torch.nn.BatchNorm1d(embed_dim),
                torch.nn.Dropout(dropouts[0]),
                MultiLayerPerceptron(embed_dim, mlp_dims[:1], dropouts[1], output_layer=False)
            ) for embed_dim in embed_dims
        ])
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropouts[1])
    
    def forward(self, x):
        x_l1 = torch.stack([fm(embedding(x)) for embedding, fm in zip(self.embeddings, self.fms)], dim=-1)
        x_l1 = x_l1.mean(dim=-1)
        p = self.mlp(x_l1)
        return torch.sigmoid(p.squeeze(1))