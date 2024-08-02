import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class FactorizationMachineModelNoLinear(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class MultiFM(torch.nn.Module):
    """
    Multi-Kernel-FM: A Multi-Embedding & Kernelization Factorization Machine Framework for CTR Prediction
    """

    def __init__(self, field_dims, embed_dims):
        super().__init__()
        self.fms = torch.nn.ModuleList([FactorizationMachineModel(field_dims, embed_dim) for embed_dim in embed_dims])
        self.weights = torch.nn.Parameter(torch.ones(len(embed_dims)))

    def forward(self, x):
        x = torch.stack([fm.linear(x) + fm.fm(fm.embedding(x)) for fm in self.fms], dim=-1)
        x = torch.sum(x * self.weights.unsqueeze(0).unsqueeze(0), dim=-1)
        return torch.sigmoid(x.squeeze(1))
