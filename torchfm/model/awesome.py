import torch

from torchfm.layer import FeaturesEmbedding, MultiLayerPerceptron
from torchfm.model.dcnv2 import CrossNetworkV2, CrossNetworkV2Model


class MultiDCNnew2(torch.nn.Module):
    
    def __init__(self, field_dims, embed_dims, num_layers, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.embed_output_dims = [len(field_dims) * embed_dim for embed_dim in embed_dims]
        self.cns = torch.nn.ModuleList([
            torch.nn.Sequential(
                CrossNetworkV2(embed_output_dim, num_layers, self.num_fields, embed_dim), 
                MultiLayerPerceptron(embed_output_dim, mlp_dims[:1], dropout, output_layer=False)
            ) for embed_dim, embed_output_dim in zip(embed_dims, self.embed_output_dims)])
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout)
    
    def forward(self, x):
        x_l1 = torch.stack([cn(embedding(x).view(-1, embed_output_dim)) 
                            for embedding, embed_output_dim, cn in 
                            zip(self.embeddings, self.embed_output_dims, self.cns)], dim=-1)
        x_l1 = x_l1.mean(dim=-1)
        p = self.mlp(x_l1)
        return torch.sigmoid(p.squeeze(1))


class MultiESingleIDCNv2(torch.nn.Module):
    
    def __init__(self, field_dims, embed_dims, num_layers, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.embed_output_dims = [len(field_dims) * embed_dim for embed_dim in embed_dims]
        assert all([embed_output_dim == self.embed_output_dims[0] for embed_output_dim in self.embed_output_dims])
        self.cn = torch.nn.Sequential(
            CrossNetworkV2(self.embed_output_dims[0], num_layers, self.num_fields, embed_dims[0]), 
            MultiLayerPerceptron(self.embed_output_dims[0], mlp_dims[:1], dropout, output_layer=False)
        )
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout)
    
    def forward(self, x):
        x_l1 = torch.stack([self.cn(embedding(x).view(-1, embed_output_dim)) 
                            for embedding, embed_output_dim in 
                            zip(self.embeddings, self.embed_output_dims)], dim=-1)
        x_l1 = x_l1.mean(dim=-1)
        p = self.mlp(x_l1)
        return torch.sigmoid(p.squeeze(1))


class WeightNormAlignedMultiDCNnew2(MultiDCNnew2):
    
    def __init__(self, field_dims, embed_dims, num_layers, mlp_dims, dropout, reg_weight=0.0):
        assert all([embed_dim == embed_dims[0] for embed_dim in embed_dims])
        super().__init__(field_dims, embed_dims, num_layers, mlp_dims, dropout)
        self.embed_dim = embed_dims[0]
        self.reg_weight = reg_weight
    
    def forward(self, x):
        output = super().forward(x)
        if self.training:
            W_all = torch.stack([torch.stack(list(cn[0].W), dim=0) for cn in self.cns], dim=0)  # (num_embed, num_layer, ND, ND)
            W_all = W_all.reshape(W_all.shape[0], W_all.shape[1], self.num_fields, self.embed_dim, self.num_fields, self.embed_dim)
            W_norm_all = (W_all ** 2).sum(dim=(3, 5))  # (num_embed, num_layer, N, N)
            W_norm_mean =  W_norm_all.mean(dim=0, keepdim=True) + 1e-6  # (1, num_layer, N, N)
            W_norm_variance_normalized = (W_norm_all - W_norm_mean).var(dim=0, unbiased=False)  # (num_layer, N, N)
            reg_loss = W_norm_variance_normalized.mean()
            return output, self.reg_weight * reg_loss
        else:
            return output


class SpaceSimilarityRegularizedMultiDCNnew2(MultiDCNnew2):
    
    def __init__(self, field_dims, embed_dims, num_layers, mlp_dims, dropout, reg_weight=0.0):
        assert all([embed_dim == embed_dims[0] for embed_dim in embed_dims])
        super().__init__(field_dims, embed_dims, num_layers, mlp_dims, dropout)
        self.embed_dim = embed_dims[0]
        self.reg_weight = reg_weight
    
    def forward(self, x):
        es = [embedding(x) for embedding in self.embeddings]
        x_l1 = torch.stack([cn(e.view(-1, embed_output_dim))
                            for e, embed_output_dim, cn in
                            zip(es, self.embed_output_dims, self.cns)], dim=-1)
        x_l1 = x_l1.mean(dim=-1)
        p = self.mlp(x_l1)
        output = torch.sigmoid(p.squeeze(1))
        if self.training:
            sims = []
            for i, e_i in enumerate(es):
                for e_j in es[:i]:
                    simss = []
                    for k in range(e_i.shape[1]):
                        simss.append(torch.svd(e_i[:, k, :].t() @ e_j[:, k, :]).S)
                    sims.append(torch.stack(simss, dim=0).mean(dim=0))
            sim = torch.cat(sims, dim=0)
            reg_loss = (sim ** 2).mean()
            return output, self.reg_weight * reg_loss
        else:
            return output


class SingularValueRegularizedDCNv2(CrossNetworkV2Model):

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout, reg_weight=0.0):
        super().__init__(field_dims, embed_dim, num_layers, mlp_dims, dropout)
        self.reg_weight = reg_weight

    def forward(self, x):
        embed_x = self.embedding(x)
        x_l1 = self.cn(embed_x.view(-1, self.embed_output_dim))
        h_l2 = self.mlp(x_l1)
        p = self.linear(h_l2)
        output = torch.sigmoid(p.squeeze(1))
        if self.training:
            reg_losses = []
            for i in range(embed_x.shape[1]):  # field
                _, S, _ = torch.svd(embed_x[:, i, :])
                # regularize the diversity of S
                # var(S / mean(S))
                reg_losses.append((S / S.mean()).var())
            reg_loss = torch.stack(reg_losses).mean()
            return output, self.reg_weight * reg_loss
        else:
            return output
