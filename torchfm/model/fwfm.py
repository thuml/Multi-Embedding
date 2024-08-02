import torch
import torch.nn as nn

from torchfm.layer import FeaturesEmbedding, FeaturesEmbeddingMultiDevice, MultiLayerPerceptron, FeaturesLinear


class NFwFM(torch.nn.Module):

    def __init__(self, num_fields):
        super().__init__()
        self.num_fields = num_fields
        self.weight = nn.Parameter(torch.randn(
            1, self.num_fields, self.num_fields
        ), requires_grad=True)

    def forward(self, inputs):
        batch_size = inputs.shape[0] 
        weight = self.weight.expand(batch_size, -1, -1, -1)  # B x 1 x F x F
        inputs_a = inputs.transpose(1, 2).unsqueeze(dim=-1)  # B x E x F x 1
        inputs_b = inputs.transpose(1, 2).unsqueeze(dim=-2)  # B x E x 1 x F

        # fwfm_inter_list = []
        # for f1 in range(self.num_fields):
        #     fwfm_inter_list.append((inputs[:, f1, :].unsqueeze(1) * inputs[:, :, :] * self.weight[:, f1, :].unsqueeze(2)).sum(dim=1))
        # fwfm_inter = sum(fwfm_inter_list)

        fwfm_inter = torch.matmul(inputs_a, inputs_b) * weight  # B x E x F x F
        fwfm_inter = torch.sum(torch.sum(fwfm_inter, dim=-1), dim=-1)  # [batch_size, emb_dim]

        return fwfm_inter


class NFwFMModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = torch.nn.Sequential(
            NFwFM(len(field_dims)),
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


class MultiNFwFMModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dims, mlp_dims, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.fwfms = torch.nn.ModuleList([
            torch.nn.Sequential(
                NFwFM(len(field_dims)),
                torch.nn.BatchNorm1d(embed_dim),
                torch.nn.Dropout(dropouts[0]),
                MultiLayerPerceptron(embed_dim, mlp_dims[:1], dropouts[1], output_layer=False)
            ) for embed_dim in embed_dims
        ])
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropouts[1])
    
    def forward(self, x):
        x_l1 = torch.stack([fwfm(embedding(x)) for embedding, fwfm in zip(self.embeddings, self.fwfms)], dim=-1)
        x_l1 = x_l1.mean(dim=-1)
        p = self.mlp(x_l1)
        return torch.sigmoid(p.squeeze(1))
