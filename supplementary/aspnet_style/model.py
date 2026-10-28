"""ASPnet-style adaptation, not an official reproduction (CVPR 2023).

Retains channel partition, MMD alignment and layer-wise shared bottlenecks.
Reuses this project's sensor encoders and compact ASFormer-style decoder.
"""
from __future__ import annotations

import itertools
import torch
from torch import nn
from torch.nn import functional as F

from atomseg.baselines import EarlyFusionContactTAS
from atomseg.formal500h_asformer import ASFormerStage, LocalTransformerBlock, SinusoidalPosition

MODALITIES = {
    "ASP_V": ("vision",),
    "ASP_VT": ("vision", "tactile"),
    "ASP_VPK": ("vision", "skeleton", "kinematics"),
    "ASP_VTPK": ("vision", "tactile", "skeleton", "kinematics"),
    "ASP_P": ("skeleton",),
    "ASP_PT": ("tactile", "skeleton"),
    "ASP_PK": ("skeleton", "kinematics"),
    "ASP_PKT": ("tactile", "skeleton", "kinematics"),
}


def config(variant, *, d_model=128, dropout=.15):
    return dict(
        variant="aspnet_style_adapted", formal_variant=variant,
        head="ASPNET_STYLE", enabled_modalities=list(MODALITIES[variant]),
        vision_dim=1536, tactile_dim=360, tactile_representation="raw_delta_abs",
        skeleton_joints=25, skeleton_joint_dim=7, kinematic_dim=48,
        d_model=d_model, dropout=dropout, modality_dropout=0., stream_layers=3,
        decoder_channels=d_model, decoder_layers=4, decoder_stages=2,
        asp_dimension=64, asp_layers=4, asp_heads=8, asp_radius=32,
        mmd_weight=1., mmd_bandwidths=[.2, .5, .9, 1.3], mmd_max_tokens=256,
        mmd_distance="mean_squared_channel_distance",
        num_actions=26, num_subtasks=0, use_boundary_head=False,
        output_rate_hz=30, event_rate_hz=30, hand_weight=0.,
        hand_loss_enabled=False, official_reproduction=False,
    )


def masked_average(values, masks):
    weights = torch.stack(masks).to(values[0].dtype)
    total = (torch.stack(values) * weights[..., None]).sum(0)
    return total / weights.sum(0).clamp_min(1)[..., None]


def mmd_squared(x, y, bandwidths=(.2, .5, .9, 1.3)):
    """Biased multi-kernel MMD², FP32, with RMS-scaled RBF distances.

    The explicit distance scale and bounded token sampling are adaptation
    choices, not claims about unavailable author implementation details.
    """
    x, y = x.float(), y.float()
    if x.shape[0] < 2 or y.shape[0] < 2:
        return (x.sum() + y.sum()) * 0
    def kernel(a, b):
        distance = ((a[:, None] - b[None]) ** 2).mean(-1)
        return torch.stack([torch.exp(-distance / (2 * sigma ** 2)) for sigma in bandwidths]).mean(0)
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()


def shared_alignment(shared, masks, max_tokens=256, bandwidths=(.2, .5, .9, 1.3)):
    loss = shared[0].float().sum() * 0
    for i, j in itertools.combinations(range(len(shared)), 2):
        # Matched, co-observed ticks only. Padding and absent sensors are excluded.
        indexes = torch.nonzero((masks[i] & masks[j]).flatten(), as_tuple=False).flatten()
        if indexes.numel() < 2:
            continue
        if indexes.numel() > max_tokens:
            take = torch.linspace(0, indexes.numel() - 1, max_tokens, device=indexes.device).long()
            indexes = indexes[take]
        a = shared[i].flatten(0, 1)[indexes]
        b = shared[j].flatten(0, 1)[indexes]
        loss = loss + mmd_squared(a, b, bandwidths)
    return loss


class ASPnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.modalities = tuple(cfg["enabled_modalities"])
        if self.modalities != MODALITIES[cfg["formal_variant"]]:
            raise ValueError("variant/modality contract mismatch")
        self.frontend = EarlyFusionContactTAS(cfg)
        # Keep only the existing sensor encoders; no dormant fusion or decoder.
        for name in ("early_projection", "early_norm", "early_temporal", "decoder"):
            delattr(self.frontend, name)
        dimensions = dict(vision=cfg["d_model"], tactile=cfg["d_model"],
                          skeleton=cfg["d_model"] // 2, kinematics=cfg["d_model"] // 2)
        dimension = cfg["asp_dimension"]
        if dimension % 2:
            raise ValueError("shared/private split requires an even dimension")
        self.project = nn.ModuleDict({m: nn.Sequential(nn.Linear(dimensions[m], dimension),
                                      nn.LayerNorm(dimension)) for m in self.modalities})
        self.position = SinusoidalPosition(dimension)
        self.layers = nn.ModuleList([
            nn.ModuleDict({m: LocalTransformerBlock(dimension, cfg["asp_heads"],
                cfg["asp_radius"], cfg["dropout"], 4) for m in self.modalities})
            for _ in range(cfg["asp_layers"])
        ])
        self.output_projection = nn.Linear((len(self.modalities) + 1) * dimension // 2, cfg["d_model"])
        self.action_head = nn.Linear(cfg["d_model"], cfg["num_actions"])
        self.refinement = ASFormerStage(
            cfg["d_model"] + cfg["num_actions"], cfg["d_model"], cfg["decoder_layers"],
            cfg["asp_heads"], cfg["asp_radius"], cfg["dropout"], 4, cfg["num_actions"])
        # Disabled encoders never learn and are excluded from parameter counts.
        if "vision" not in self.modalities:
            for module in (self.frontend.vision_adapter, self.frontend.vision_temporal, self.frontend.view_score):
                module.requires_grad_(False)
            self.frontend.view_null.requires_grad_(False)
        for m, attr in (("tactile", "tactile_encoder"), ("skeleton", "skeleton_encoder"), ("kinematics", "kinematic_encoder")):
            if m not in self.modalities:
                getattr(self.frontend, attr).requires_grad_(False)

    def forward(self, batch):
        enc = self.frontend._encode_modalities(batch, False)
        seq = enc["sequence_mask"].bool()
        names = dict(vision=("vision_token", "vision_quality"),
                     tactile=("tactile_token", "tactile_quality"),
                     skeleton=("skeleton_token", "skeleton_quality"),
                     kinematics=("kinematic_token", "kinematic_quality"))
        private, shared, masks = [], [], []
        for modality in self.modalities:
            feature_name, quality_name = names[modality]
            quality = enc[quality_name]
            valid = (quality > 0) & seq
            token = enc[feature_name] * quality[..., None]
            latent = self.project[modality](token) * valid[..., None]
            p, s = latent.chunk(2, dim=-1)
            private.append(p); shared.append(s); masks.append(valid)
        mmd = shared_alignment(shared, masks, self.cfg["mmd_max_tokens"], self.cfg["mmd_bandwidths"]) if self.training else shared[0].new_zeros((), dtype=torch.float32)
        bottleneck = masked_average(shared, masks)
        # For entirely padded batch rows, one inert attention key avoids NaNs.
        safe_seq = seq.clone()
        safe_seq[~seq.any(1), 0] = True
        for layer_index, layer in enumerate(self.layers):
            next_private, next_shared = [], []
            for i, modality in enumerate(self.modalities):
                value = torch.cat([private[i], bottleneck], -1)
                if layer_index == 0:
                    value = self.position(value)
                value = layer[modality](value * seq[..., None], safe_seq)
                p, s = value.chunk(2, -1)
                next_private.append(p * masks[i][..., None])
                next_shared.append(s * masks[i][..., None])
            private = next_private
            bottleneck = masked_average(next_shared, masks)
        features = F.gelu(self.output_projection(torch.cat([bottleneck, *private], -1))) * seq[..., None]
        first = self.action_head(features) * seq[..., None]
        last, _ = self.refinement(torch.cat([features, first.softmax(-1).detach()], -1), safe_seq)
        last = last * seq[..., None]
        boundary = last.new_full(last.shape[:2], -20.)
        return dict(atomic_logits=last, atomic_stages=[first, last],
                    boundary_logits=boundary, boundary_stages=[boundary, boundary],
                    hand_logits=last.new_zeros((*last.shape[:2], 3)),
                    hand_prediction_trained=False, asp_mmd=mmd,
                    view_weights=enc["view_weights"])
