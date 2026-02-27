import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewContrastiveLoss(nn.Module):
    def __init__(self, bev_dim: int = 128, sat_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.bev_proj = nn.Linear(bev_dim, sat_dim)

    def forward(self, bev_feat: torch.Tensor, sat_feat: torch.Tensor) -> torch.Tensor:
        B = bev_feat.shape[0]

        bev_global = F.adaptive_avg_pool2d(bev_feat, 1).flatten(1)
        sat_global = F.adaptive_avg_pool2d(sat_feat, 1).flatten(1)

        bev_global = self.bev_proj(bev_global)
        bev_global = F.normalize(bev_global, dim=1)
        sat_global = F.normalize(sat_global, dim=1)

        logits = torch.matmul(bev_global, sat_global.T) / self.temperature

        labels = torch.arange(B, device=bev_feat.device)

        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, bev_dim: int = 128, sat_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.sat_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bev_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bev_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        sat_feat: torch.Tensor,
        bev_feat: torch.Tensor,
    ) -> torch.Tensor:
        sat_embed = self.sat_projector(sat_feat)
        bev_embed = self.bev_projector(bev_feat)

        sat_embed = F.normalize(sat_embed, dim=1)
        bev_embed = F.normalize(bev_embed, dim=1)

        similarity = torch.sum(sat_embed * bev_embed, dim=1)
        loss = 1 - similarity.mean()

        return loss


class GeometricConsistencyLoss(nn.Module):
    def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(self, alignment_matrix: torch.Tensor) -> torch.Tensor:
        R = alignment_matrix[:, :3, :3]
        t = alignment_matrix[:, :3, 3]
        
        RtR = torch.bmm(R.transpose(1, 2), R)
        eye = torch.eye(3, device=alignment_matrix.device).unsqueeze(0).expand_as(RtR)
        ortho_loss = torch.norm(RtR - eye, dim=(1, 2)).mean()
        
        det_R = torch.linalg.det(R)
        det_loss = torch.mean((det_R - 1.0) ** 2)
        
        trans_reg = torch.mean(t ** 2)
        
        total_loss = (
            self.rotation_weight * (ortho_loss + det_loss) +
            self.translation_weight * trans_reg
        )
        
        return total_loss


class SpatialSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        dx = bev_feat[:, :, :, 1:] - bev_feat[:, :, :, :-1]
        dy = bev_feat[:, :, 1:, :] - bev_feat[:, :, :-1, :]
        
        loss_x = torch.mean(torch.abs(dx))
        loss_y = torch.mean(torch.abs(dy))
        
        return loss_x + loss_y
