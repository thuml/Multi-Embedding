import torch

from torchfm.layer import FeaturesEmbedding, MultiLayerPerceptron


class FeatureSelection(torch.nn.Module):

    def __init__(self, embed_output_dim, embed_dim, fs_mlp_dims, dropout):
        super().__init__()
        self.ctx = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.gate = MultiLayerPerceptron(embed_dim, fs_mlp_dims + (embed_output_dim, ), dropout, output_layer=False)
    
    def forward(self, emb):
        return 2 * self.gate(self.ctx.repeat(emb.shape[0], 1)) * emb


class FinalMLPInter(torch.nn.Module):

    def __init__(self, embed_dim, embed_output_dim, mlp_dims, fs_mlp_dims, dropout):
        super().__init__()
        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                # FeatureSelection(embed_output_dim, embed_dim, fs_mlp_dims, dropout),
                torch.nn.Identity(),
                MultiLayerPerceptron(embed_output_dim, mlp_dims, dropout, output_layer=False)
            )
            for _ in range(2)
        ])
    
    def forward(self, emb):
        return torch.stack([mlp(emb.flatten(1)) for mlp in self.mlps], dim=-1)


class FinalMLPFusion(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.w_x = torch.nn.Linear(hidden_dim, 1)
        self.w_y = torch.nn.Linear(hidden_dim, 1)
        self.w_xy = torch.nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        # torch.nn.init.xavier_normal_(self.w_xy)
    
    def forward(self, hidden):
        x, y = hidden[:, :, 0], hidden[:, :, 1]
        xy = self.w_x(x) + self.w_y(y) + ((x @ self.w_xy) * y).sum(dim=1, keepdim=True)
        return xy


class FinalMLP(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, fs_mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.inter = FinalMLPInter(embed_dim, self.embed_output_dim, mlp_dims, fs_mlp_dims, dropout)
        self.fusion = FinalMLPFusion(mlp_dims[-1])
    
    def forward(self, x):
        cross_term = self.inter(self.embedding(x))
        x = self.fusion(cross_term)
        return torch.sigmoid(x.squeeze(1))


class MultiFinalMLP(torch.nn.Module):

    def __init__(self, field_dims, embed_dims, mlp_dims, fs_mlp_dims, dropout):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.embed_output_dims = [len(field_dims) * embed_dim for embed_dim in embed_dims]
        self.inters = torch.nn.ModuleList([
            FinalMLPInter(embed_dim, embed_output_dim, mlp_dims, fs_mlp_dims, dropout)
            for embed_dim, embed_output_dim in zip(embed_dims, self.embed_output_dims)
        ])
        self.fusion = FinalMLPFusion(mlp_dims[-1])
    
    def forward(self, x):
        cross_term = torch.stack([inter(embedding(x)) for inter, embedding in zip(self.inters, self.embeddings)], dim=-1).mean(dim=-1)
        x = self.fusion(cross_term)
        return torch.sigmoid(x.squeeze(1))
