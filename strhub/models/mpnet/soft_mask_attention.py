from typing import Optional
from einops import rearrange, repeat
import math
import torch
import torch.nn as nn
from torch import Tensor


class SoftMaskAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1, bias=True,
                 learnable_scaling_factors=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = nhead
        self.dropout = dropout
        self.head_dims = embed_dim // nhead
        assert self.head_dims * nhead == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if learnable_scaling_factors:
            self.scaling_factors = nn.Parameter(torch.as_tensor(self.get_default_scaling_factors(),
                                                                dtype=torch.float32))
        else:
            self.register_buffer("scaling_factors", torch.as_tensor(self.get_default_scaling_factors(),
                                                                    dtype=torch.float32))

    def get_default_scaling_factors(self):
        assert self.num_heads == 12
        return [-64, -64, -64, -32, -32, -32, -16, -16, -16, -8, -8, -8]
        # return [float('-inf')]*12

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                return_attn_weights: Optional[bool] = False, device: Optional[torch.device] = 'cuda:0'):
        """
        Forward method for soft key_padding_mask attention
        :param query: [bs, L, embed_dim]
        :param key: [bs, src_len, embed_dim]
        :param value: [bs, src_len, embed_dim]
        :param attn_mask: optional tensor of shape [L, src_len] of 0.0 or float('-inf')
        :param key_padding_mask: optional tensor of shape [bs, src_len] with values in range [0,1]
        :param return_attn_weights: if True, attention weights will also be returned as
        a tensor of shape [bs, nhead, L, src_len]
        :return: tensor of a shape [bs, L, embed_dim]
        """
        assert key.shape == value.shape, f"Shape mismatch: {key.shape}, {value.shape}"
        if attn_mask is not None:
            assert query.shape[1] == attn_mask.shape[0], f"Shape mismatch: {query.shape}, {attn_mask.shape}"
        # masking_num = [-32, -32, -32, -16, -16, -16, -8, -8, -8, -4, -4, -4]
        # scaling_factors = torch.as_tensor(masking_num, dtype=torch.float32, device=device)

        bs = query.size(0)
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = rearrange(query, "bs L (num_heads head_dims) -> (bs num_heads) L head_dims",
                          num_heads=self.num_heads, head_dims=self.head_dims)
        key = rearrange(key, "bs src_len (num_heads head_dims) -> (bs num_heads) head_dims src_len",
                        num_heads=self.num_heads, head_dims=self.head_dims)
        value = rearrange(value, "bs src_len (num_heads head_dims) -> (bs num_heads) src_len head_dims",
                          num_heads=self.num_heads, head_dims=self.head_dims)

        query = query / float(math.sqrt(self.head_dims))
        attn_wts = torch.bmm(query, key)

        if key_padding_mask is not None:
            # print(key_padding_mask)
            key_padding_mask = repeat(key_padding_mask, "bs src_len -> bs num_heads L src_len",
                                      num_heads=self.num_heads, L=query.size(1))
            key_padding_mask = (key_padding_mask * self.scaling_factors[None, :, None, None]).flatten(0, 1)
            attn_wts = attn_wts + key_padding_mask

        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "L src_len -> (bs num_heads) L src_len",
                               bs=bs, num_heads=self.num_heads)
            attn_wts += attn_mask

        attn_wts = attn_wts.softmax(-1)
        # attn_wts = torch.where(attn_wts < 1e-5, 0.0, attn_wts)
        attn_wts = self.dropout(attn_wts)

        attn_output = torch.bmm(attn_wts, value)
        attn_output = rearrange(attn_output, "(bs num_heads) L head_dims -> bs L (num_heads head_dims)",
                                num_heads=self.num_heads, head_dims=self.head_dims)

        if return_attn_weights:
            attn_wts = rearrange(attn_wts, "(bs num_heads) L src_len -> bs num_heads L src_len", bs=bs)
            # average the attention weights across the different heads
            return self.out_proj(attn_output), attn_wts
        else:
            return self.out_proj(attn_output)
