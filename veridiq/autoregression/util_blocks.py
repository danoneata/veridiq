import torch
import torch.nn as nn
import torch.nn.functional as F


def build_windowed_causal_mask(T, W, key_padding_mask, device):

    idx = torch.arange(T, device=device)
    diff = idx[:, None] - idx[None, :]        # (T, T) = i - j
    allowed = (diff >= 0) & (diff <= W)       # causal and within window
    base = (~allowed).unsqueeze(0).unsqueeze(0)  # (1,1,T,T), True = mask

    assert key_padding_mask.shape == (key_padding_mask.size(0), T)
    B = key_padding_mask.size(0)
    kpad = key_padding_mask.view(B, 1, 1, T).to(torch.bool)  # (B,1,1,T)
    return base | kpad  # (B,1,T,T)


class WindowedCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)

    def forward(self, x, attn_mask):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class WindowedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = WindowedCausalSelfAttention(d_model, nhead, attn_dropout=dropout, proj_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
