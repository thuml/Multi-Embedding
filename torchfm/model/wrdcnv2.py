import torch
from torchfm.layer import FeaturesEmbedding, MultiLayerPerceptron


class CriterionWithLoss(torch.nn.Module):

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, input, target):
        input_real, losses = input[0], input[1:]
        return self.criterion(input_real, target) + sum(losses)


class WeightedRestrictedCrossNetworkV2(torch.nn.Module):  # layer

    def __init__(self, input_dim, num_layers, num_fields, embed_dim):
        """
        input_dim: num_fields*embed_dim
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.N = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(num_fields, num_fields)) for _ in range(num_layers)
        ])
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
        reg_loss = 0
        for i in range(self.num_layers):
            x = x.unsqueeze(2)
            ni = self.N[i].unsqueeze(1).unsqueeze(3).repeat(1, self.embed_dim, 1, self.embed_dim).reshape(self.input_dim, self.input_dim)
            xw = torch.matmul(ni * self.W[i], x)
            xw = xw.squeeze(2)
            x = x.squeeze(2)
            x = x0 * (xw + self.b[i]) + x
            # Calculate regularization
            field_wise_w = self.W[i].reshape(self.num_fields, self.embed_dim, self.num_fields, self.embed_dim)
            field_wise_w = field_wise_w.transpose(1, 2).reshape(-1, self.embed_dim, self.embed_dim)
            identities = torch.eye(self.embed_dim, device=field_wise_w.device).unsqueeze(0)
            reg_loss += torch.square(identities - torch.bmm(field_wise_w, field_wise_w.transpose(1, 2))).sum()
        return x, reg_loss


class WeightedRestrictedCrossNetworkV2Model(torch.nn.Module):  # model
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
        self.cn = WeightedRestrictedCrossNetworkV2(self.embed_output_dim, num_layers, self.num_fields, embed_dim)
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
        x_l1, reg_loss = self.cn(embed_x)
        h_l2 = self.mlp(x_l1)
        p = self.linear(h_l2)
        if self.training:
            return torch.sigmoid(p.squeeze(1)), 1e-5 * reg_loss
        else:
            return torch.sigmoid(p.squeeze(1))


class WeightedRestrictedMultiDCN(torch.nn.Module):
    def __init__(self, field_dims, embed_dims, num_layers, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.embed_output_dims = [len(field_dims) * embed_dim for embed_dim in embed_dims]
        self.cns = torch.nn.ModuleList([
            WeightedRestrictedCrossNetworkV2(embed_output_dim, num_layers, self.num_fields, embed_dim)
            for embed_dim, embed_output_dim in zip(embed_dims, self.embed_output_dims)
        ])
        self.projs = torch.nn.ModuleList([
            MultiLayerPerceptron(embed_output_dim, mlp_dims[:1], dropout, output_layer=False)
            for embed_output_dim in self.embed_output_dims
        ])
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout)


    def forward(self, x):
        h_and_regs = [cn(embedding(x).view(-1, embed_output_dim))
                       for cn, embedding, embed_output_dim in zip(self.cns, self.embeddings, self.embed_output_dims)]
        x_l1 = torch.stack([proj(h_and_reg[0]) for proj, h_and_reg in zip(self.projs, h_and_regs)], dim=-1).mean(dim=-1)
        reg_loss = sum([h_and_reg[1] for h_and_reg in h_and_regs])
        p = self.mlp(x_l1)
        if self.training:
            return torch.sigmoid(p.squeeze(1)), 1e-5 * reg_loss
        else:
            return torch.sigmoid(p.squeeze(1))
