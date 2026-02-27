import os
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        qkv_bias: bool,
        norm: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.postnorm = norm(dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, _, H, W = q.shape

        q = rearrange(q, "b n d H W -> b n (H W) d")
        k = rearrange(k, "b n d h w -> b n (h w) d")
        v = rearrange(v, "b n d h w -> b (n h w) d")

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)

        b, n, q_len, d = q.shape
        _, _, k_len, _ = k.shape
        _ = (n, d, k_len)

        max_chunk_size = max(1, min(q_len, int(os.environ.get("SCVT_CROSS_ATTN_CHUNK", "128"))))
        outputs = []

        qk_compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        for q_start in range(0, q_len, max_chunk_size):
            q_end = min(q_start + max_chunk_size, q_len)
            q_chunk = q[:, :, q_start:q_end, :]

            chunk_dot = self.scale * torch.einsum(
                "b n Q d, b n K d -> b n Q K",
                q_chunk.to(dtype=qk_compute_dtype),
                k.to(dtype=qk_compute_dtype),
            )
            chunk_dot = rearrange(chunk_dot, "b n Q K -> b Q (n K)")
            chunk_dot = chunk_dot - chunk_dot.amax(dim=-1, keepdim=True)
            a = chunk_dot.softmax(dim=-1)
            out_chunk = torch.einsum("b Q K, b K d -> b Q d", a, v.to(dtype=qk_compute_dtype))
            outputs.append(out_chunk.to(dtype=q.dtype))

        z = torch.cat(outputs, dim=1)
        z = rearrange(z, "(b m) Q d -> b Q (m d)", m=self.heads)
        z = self.proj(z)

        if skip is not None:
            z = z + rearrange(skip, "b d H W -> b (H W) d")

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, "b (H W) d -> b d H W", H=H, W=W)
        return z

