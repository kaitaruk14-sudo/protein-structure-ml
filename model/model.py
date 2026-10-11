"""
BasicDistogramNet: a small version of the AlphaFold 1 / trRosetta idea.

  sequence  ->  [1] embedding        each amino acid -> 64 numbers
            ->  [2] 2 attention layers  each residue looks at all the others
            ->  [3] pair grid        combine residue i and j (+ their distance
                                     along the chain) -> 32 numbers per pair
            ->  [4] 8 2D residual blocks  convolutions over the pair grid,
                                     like image recognition (dilations 1,2,4,8)
            ->  [5] output           64 distance-bin probabilities per pair

Input is the single sequence only (no MSA / evolutionary information).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import N_BINS, PAD


class AttentionBlock(nn.Module):
    """Pre-norm Transformer layer: self-attention, then a small feed-forward net."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        B, L, D = x.shape
        q, k, v = (self.qkv(self.norm1(x))
                   .view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4))
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                             dropout_p=0.1 if self.training else 0.0)
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.norm2(x)))


class MaskedInstanceNorm2d(nn.Module):
    """Instance norm whose statistics ignore padding (so padding can't change results)."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x, mask):
        n = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=(2, 3), keepdim=True) / n
        var = (((x - mean) * mask) ** 2).sum(dim=(2, 3), keepdim=True) / n
        return ((x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias) * mask


class ResBlock2D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = MaskedInstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = MaskedInstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x, mask):
        h = self.conv1(F.elu(self.norm1(x, mask)))
        h = self.conv2(F.elu(self.norm2(self.drop(h), mask)))
        return (x + h) * mask


class BasicDistogramNet(nn.Module):
    def __init__(self, seq_dim=64, n_heads=4, n_seq_layers=2,
                 pair_channels=32, n_res_blocks=8, max_rel_pos=32):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.n_heads = n_heads
        n_rel = 2 * max_rel_pos + 1
        # [1] + [2] sequence track
        self.embed = nn.Embedding(21, seq_dim, padding_idx=PAD)
        self.attn_rel_pos = nn.Embedding(n_rel, n_heads)       # attention bias by |i-j|
        self.seq_layers = nn.ModuleList([AttentionBlock(seq_dim, n_heads)
                                         for _ in range(n_seq_layers)])
        self.seq_norm = nn.LayerNorm(seq_dim)
        # [3] pair grid
        self.left = nn.Linear(seq_dim, pair_channels)
        self.right = nn.Linear(seq_dim, pair_channels)
        self.pair_rel_pos = nn.Embedding(n_rel, pair_channels)
        # [4] 2D residual blocks
        self.blocks = nn.ModuleList([ResBlock2D(pair_channels, [1, 2, 4, 8][k % 4])
                                     for k in range(n_res_blocks)])
        # [5] output
        self.out_norm = MaskedInstanceNorm2d(pair_channels)
        self.head = nn.Conv2d(pair_channels, N_BINS, 1)

    def forward(self, tokens, seq_mask):
        B, L = tokens.shape
        idx = torch.arange(L, device=tokens.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-self.max_rel_pos, self.max_rel_pos) \
              + self.max_rel_pos                                     # [L, L]

        bias = self.attn_rel_pos(rel).permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]
        bias = bias.masked_fill((~seq_mask)[:, None, None, :], -1e4)  # ignore padding
        h = self.embed(tokens)
        for layer in self.seq_layers:
            h = layer(h, bias)
        h = self.seq_norm(h)                                         # [B, L, D]

        pair = self.left(h)[:, :, None] + self.right(h)[:, None, :] + self.pair_rel_pos(rel)
        pmask = (seq_mask[:, :, None] & seq_mask[:, None, :]).float().unsqueeze(1)
        x = pair.permute(0, 3, 1, 2) * pmask                         # [B, C, L, L]
        for block in self.blocks:
            x = block(x, pmask)

        logits = self.head(F.elu(self.out_norm(x, pmask)))           # [B, 64, L, L]
        return 0.5 * (logits + logits.transpose(2, 3))               # distances are symmetric


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
