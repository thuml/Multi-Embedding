import torch
import torch.nn.functional as F

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class AutomaticFeatureInteractionModel(torch.nn.Module):
    """
    A pytorch implementation of AutoInt.

    Reference:
        W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.
    """

    def __init__(self, field_dims, embed_dim, atten_embed_dim, num_heads, num_layers, mlp_dims, dropouts, has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        atten_x = self.atten_embedding(embed_x)
        cross_term = atten_x.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(embed_x)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class AttentionModule(torch.nn.Module):

    def __init__(self, atten_embed_dim, num_heads, dropout, residual=True, layer_norm=True):
        super().__init__()
        self.atten = torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropout)
        self.layer_norm = torch.nn.LayerNorm(atten_embed_dim) if layer_norm else None
        self.residual = residual
    
    def forward(self, x):
        if self.residual:
            x = x + self.atten(x, x, x)[0]
        else:
            x = self.atten(x, x, x)[0]
        if self.layer_norm:
            x = self.layer_norm(x)
        return x

class MultiHeadSelfAttentionInteraction(torch.nn.Module):
    """
    Multi-head self-attention only
    """
    def __init__(self, embed_dim, atten_embed_dim, num_heads, num_layers, dropout, residual=True, layer_norm=True):
        super().__init__()
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.attens = torch.nn.ModuleList([
            AttentionModule(atten_embed_dim, num_heads, dropout, residual=residual, layer_norm=layer_norm)
            for _ in range(num_layers)
        ])
    
    def forward(self, emb):
        atten_emb = self.atten_embedding(emb)
        atten_emb = atten_emb.transpose(0, 1)  # batch second: (N, B, H)
        for atten in self.attens:
            atten_emb = atten(atten_emb)
        atten_emb = atten_emb.transpose(0, 1)  # batch first: (B, N, H)
        atten_emb = F.relu(atten_emb)
        return atten_emb


class AutoInt(torch.nn.Module):
    """
    AutoInt w/o linear
    """
    def __init__(self, field_dims, embed_dim, atten_embed_dim, num_heads, num_layers, dropouts, mlp_dims, residual=True, layer_norm=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten = MultiHeadSelfAttentionInteraction(embed_dim, atten_embed_dim, num_heads, num_layers, dropouts[0], residual=residual, layer_norm=layer_norm)
        self.atten_post = torch.nn.Linear(self.atten_output_dim, 1)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1]) if mlp_dims else None
    
    def forward(self, x):
        emb = self.embedding(x)
        x = self.atten_post(self.atten(emb).flatten(1))
        if self.mlp:
            x += self.mlp(emb.flatten(1))
        return torch.sigmoid(x.squeeze(1))


class MultiAutoInt(torch.nn.Module):

    def __init__(self, field_dims, embed_dims, atten_embed_dim, num_heads, num_layers, dropouts, mlp_dims, residual=True, layer_norm=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_output_dims = [len(field_dims) * embed_dim for embed_dim in embed_dims]
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.embeddings = torch.nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for embed_dim in embed_dims])
        self.attens = torch.nn.ModuleList([
            MultiHeadSelfAttentionInteraction(embed_dim, atten_embed_dim, num_heads, num_layers, dropouts[0], residual=residual, layer_norm=layer_norm)
            for embed_dim in embed_dims
        ])
        self.atten_post = torch.nn.Linear(self.atten_output_dim, 1)
        self.mlps = torch.nn.ModuleList([
            MultiLayerPerceptron(embed_output_dim, mlp_dims[:1], dropouts[1], output_layer=False)
            for embed_output_dim in self.embed_output_dims
        ]) if mlp_dims else None
        self.mlp_post = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropouts[1]) if mlp_dims else None
    
    def forward(self, x):
        embs = [embedding(x) for embedding in self.embeddings]
        atten_hidden = torch.stack([atten(emb) for atten, emb in zip(self.attens, embs)], dim=-1).mean(dim=-1)
        x = self.atten_post(atten_hidden.flatten(1))
        if self.mlps:
            mlp_hidden = torch.stack([mlp(emb.flatten(1)) for mlp, emb in zip(self.mlps, embs)], dim=-1).mean(dim=-1)
            x += self.mlp_post(mlp_hidden)
        return torch.sigmoid(x.squeeze(1))
