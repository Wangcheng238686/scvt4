import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing="xy"), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [0.0, -sw, w / 2.0],
        [-sh, 0.0, h * offset + h / 2.0],
        [0.0, 0.0, 1.0],
    ]


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int = None,
        w_meters: int = None,
        offset: int = 0,
        decoder_blocks: list = [2, 2],
        grid_refine_scale: float = 0.5,
        init_scale: float = 0.1,
    ):
        super().__init__()
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))
        self.h = h
        self.w = w

        grid_pixels = generate_grid(h, w).squeeze(0)
        grid_pixels[0] = bev_width * grid_pixels[0]
        grid_pixels[1] = bev_height * grid_pixels[1]
        self.register_buffer("grid_pixels", grid_pixels, persistent=False)

        if h_meters is None or w_meters is None:
            V_inv_init = torch.eye(3)
        else:
            V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
            V_inv_init = torch.FloatTensor(V).inverse()
        
        if init_scale > 0:
            noise = torch.randn(3, 3) * init_scale
            noise[2, :] = 0
            V_inv_init = V_inv_init + noise
        
        self.V_inv = nn.Parameter(V_inv_init)

        self.grid_offset = nn.Parameter(torch.zeros(3, h, w))
        self.grid_refine_scale = grid_refine_scale

        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))

    @property
    def grid(self):
        flat = rearrange(self.grid_pixels, "d h w -> d (h w)")
        grid = self.V_inv @ flat
        grid = rearrange(grid, "d (h w) -> d h w", h=self.h, w=self.w)
        return grid + self.grid_refine_scale * self.grid_offset

    def get_prior(self):
        return self.learned_features

