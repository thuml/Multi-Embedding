import torch
from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron


class CrossNetworkV2(torch.nn.Module):  # layer

    def __init__(self, input_dim, num_layers, num_fields, embed_dim):
        """
        input_dim: num_fields*embed_dim
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.W = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(input_dim, input_dim)) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(input_dim, )) for _ in range(num_layers)
        ])
        for i in range(num_layers):
            torch.nn.init.xavier_uniform_(self.W[i])

    def forward(self, x):
        """
        x: Tensor of size ``(batch_size, num_fields*embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            x = x.unsqueeze(2)
            xw = torch.matmul(self.W[i], x)
            xw = xw.squeeze(2)
            x = x.squeeze(2)
            x = x0 * (xw + self.b[i]) + x
        return x


class CrossNetworkV2Model(torch.nn.Module):  # model
    """
    A pytorch implementation of Deep & Cross Network - V2.
    Only Cross Network, without deep network.
    Reference:
        R Wang, et al. DCN V2: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems, 2021
    """
    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetworkV2(self.embed_output_dim, num_layers, self.num_fields, embed_dim)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1], 1)

    def forward(self, x):
        """
        x: Tensor of size ``(batch_size, num_fields)``
        self.embedding(x): Tensor of size ``(batch_size, num_fields, embed_dim)``
        embed_x: Tensor of size ``(batch_size, num_fields*embed_dim)``
        x: Tensor of size ``(batch_size, num_fields*embed_dim)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(x_l1)
        p = self.linear(h_l2)
        return torch.sigmoid(p.squeeze(1))
