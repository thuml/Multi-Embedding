import torch

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, InnerProductNetwork, \
    OuterProductNetwork, MultiLayerPerceptron


class ProductNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.
    Reference:
        Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, method='inner'):
        super().__init__()
        num_fields = len(field_dims)
        if method == 'inner':
            self.pn = InnerProductNetwork()
        elif method == 'outer':
            self.pn = OuterProductNetwork(num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = num_fields * embed_dim
        self.mlp = MultiLayerPerceptron(num_fields * (num_fields - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        cross_term = self.pn(embed_x)
        x = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(x)
        return torch.sigmoid(x.squeeze(1))


class MultiPNNModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dims, mlp_dims, dropout, method='inner'):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.embed_output_dim = self.num_fields * (self.num_fields - 1) // 2
        if method == 'inner':
            self.pns = torch.nn.ModuleList([
                torch.nn.Sequential(
                    InnerProductNetwork(), 
                    MultiLayerPerceptron(self.embed_output_dim, mlp_dims[:1], dropout, output_layer=False)
                ) for _ in range(len(self.embeddings))
            ])
        elif method == 'outer':
            self.pns = torch.nn.ModuleList([
                torch.nn.Sequential(
                    OuterProductNetwork(self.num_fields, embed_dim),
                    MultiLayerPerceptron(self.embed_output_dim, mlp_dims[:1], dropout, output_layer=False)
                ) for embed_dim in embed_dims
            ])
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout)
    
    def forward(self, x):
        x_l1 = torch.stack([pn(embedding(x)) for embedding, pn in zip(self.embeddings, self.pns)], dim=-1)
        x_l1 = x_l1.mean(dim=-1)
        p = self.mlp(x_l1)
        return torch.sigmoid(p.squeeze(1))
