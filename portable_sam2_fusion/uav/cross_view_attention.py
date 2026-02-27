import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple

from torchvision.models.resnet import Bottleneck

from .bev_embedding import BEVEmbedding
from .sparse_attention import CrossAttention


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing="xy"), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
    return indices


class SceneAwareAlignment(nn.Module):
    def __init__(
        self,
        max_scenes: int = 1000,
        embed_dim: int = 32,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.max_scenes = max_scenes
        self.embed_dim = embed_dim
        self.init_scale = init_scale
        
        self.scene_embeddings = nn.Embedding(max_scenes, embed_dim)
        nn.init.normal_(self.scene_embeddings.weight, 0, 0.01)
        
        self.alignment_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        
        self.global_alignment = nn.Parameter(self._init_alignment())
    
    def _init_alignment(self) -> torch.Tensor:
        eye = torch.eye(4)
        if self.init_scale > 0:
            noise = torch.randn(4, 4) * self.init_scale
            noise[3, :] = 0
            eye = eye + noise
        return eye
    
    def forward(
        self,
        scene_indices: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        scene_embed = self.scene_embeddings(scene_indices)
        alignment_delta = self.alignment_head(scene_embed).view(-1, 4, 4)
        
        eye = torch.eye(4, device=device).unsqueeze(0)
        alignment = eye + self.init_scale * torch.tanh(alignment_delta)
        
        return alignment
    
    def get_global_alignment(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.global_alignment.unsqueeze(0).expand(batch_size, -1, -1)


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer("image_plane", image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False),
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False),
            )

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = bool(skip)

    def forward(
        self,
        x: torch.Tensor,
        bev: BEVEmbedding,
        feature: torch.Tensor,
        I_inv: torch.Tensor,
        E_inv: torch.Tensor,
        alignment_transform: Optional[torch.Tensor] = None,#可学习的对齐矩阵
    ) -> torch.Tensor:
        b, n, _, h, w = feature.shape
        feature_flat = rearrange(feature, "b n c h w -> (b n) c h w")

        if alignment_transform is not None:
            # alignment_transform: (B, 4, 4)
            # E_inv: (B, N, 4, 4)
            # Apply alignment: T_aligned = T_alignment @ E_inv
            E_inv = torch.einsum("b i j, b n j k -> b n i k", alignment_transform, E_inv)

        cam = self.image_plane
        cam = cam.expand(b, n, -1, -1, -1)
        cam = rearrange(cam, "b n d h w -> b n d (h w)")
        cam = torch.einsum("b n d k, b n c d -> b n c k", cam, I_inv)
        cam = F.pad(cam, (0, 0, 0, 1), value=1.0)

        c = E_inv[..., -1:]
        c = rearrange(c, "b n d k -> (b n) d k")
        c = c[..., None]
        c_embed = self.cam_embed(c)

        d = E_inv @ cam
        d_flat = rearrange(d, "b n d (h w) -> (b n) d h w", h=h, w=w)
        d_embed = self.img_embed(d_flat)

        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)
        world = bev.grid[:2]
        w_embed = self.bev_embed(world[None])
        bev_embed = w_embed - c_embed
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        query_pos = rearrange(bev_embed, "(b n) ... -> b n ...", b=b, n=n)

        if img_embed.shape[2:] != feature_flat.shape[2:]:
            img_embed = F.interpolate(img_embed, size=feature_flat.shape[2:], mode="bilinear", align_corners=False)

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            key_flat = img_embed

        val_flat = self.feature_linear(feature_flat)

        query = query_pos + x[:, None]
        key = rearrange(key_flat, "(b n) ... -> b n ...", b=b, n=n)
        val = rearrange(val_flat, "(b n) ... -> b n ...", b=b, n=n)

        return self.cross_attend(query, key, val, skip=x if self.skip else None)


class CVTEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        cross_view: dict,
        bev_embedding: dict,
        dim: int = 128,
        middle: List[int] = [2, 2],
        scale: float = 1.0,
        use_checkpoint: bool = False,
        max_scenes: int = 1000,
        scene_embed_dim: int = 32,
        alignment_init_scale: float = 0.01,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.use_checkpoint = use_checkpoint

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = []
        layers = []

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewAttention(
                feat_height,
                feat_width,
                feat_dim,
                dim,
                **cross_view,
            )
            cross_views.append(cva)
            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

        self.scene_alignment = SceneAwareAlignment(
            max_scenes=max_scenes,
            embed_dim=scene_embed_dim,
            init_scale=alignment_init_scale,
        )

    def _forward_layer(self, x, cross_view, feature, bev_embedding, I_inv, E_inv, b, n):
        feature = rearrange(feature, "(b n) ... -> b n ...", b=b, n=n)
        x = cross_view(x, bev_embedding, feature, I_inv, E_inv, alignment_transform=None)
        return x

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        scene_indices: Optional[torch.Tensor] = None,
    ):
        b, n, _, _, _ = batch["image"].shape
        device = batch["image"].device

        image = batch["image"].flatten(0, 1).clone()
        intrinsics = batch["intrinsics"].to(dtype=torch.float32)
        extrinsics = batch["extrinsics"].to(dtype=torch.float32)

        if scene_indices is not None:
            alignment_transform = self.scene_alignment(
                scene_indices, b, device
            )
        else:
            alignment_transform = self.scene_alignment.get_global_alignment(b, device)

        I_inv = torch.linalg.pinv(intrinsics)
        E_inv = torch.linalg.pinv(extrinsics)

        eye3 = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
        eye4 = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
        bad_I = ~torch.isfinite(I_inv).all(dim=(-1, -2))
        bad_E = ~torch.isfinite(E_inv).all(dim=(-1, -2))
        if bad_I.any():
            I_inv = I_inv.clone()
            I_inv[bad_I] = eye3
        if bad_E.any():
            E_inv = E_inv.clone()
            E_inv[bad_E] = eye4

        I_inv = torch.where(torch.isfinite(I_inv), I_inv, torch.zeros_like(I_inv))
        E_inv = torch.where(torch.isfinite(E_inv), E_inv, torch.zeros_like(E_inv))

        E_inv = torch.einsum("b i j, b n j k -> b n i k", alignment_transform, E_inv)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()
        x = repeat(x, "... -> b ...", b=b)

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._forward_layer,
                    x, cross_view, feature, self.bev_embedding, I_inv, E_inv, b, n,
                    use_reentrant=False,
                )
            else:
                x = self._forward_layer(x, cross_view, feature, self.bev_embedding, I_inv, E_inv, b, n)
            x = layer(x)

        return x
