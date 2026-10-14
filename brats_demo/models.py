"""Model definitions copied from HK7-CV-Class/brats-transunet.

The module paths do not matter to a PyTorch state dict, but module and parameter
names do. Keep the class layout aligned with the training project so its U-Net
and TransUNet checkpoints can be loaded strictly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(dims: int):
    return nn.Conv2d if dims == 2 else nn.Conv3d


def conv_t_nd(dims: int):
    return nn.ConvTranspose2d if dims == 2 else nn.ConvTranspose3d


def norm_nd(dims: int):
    return nn.InstanceNorm2d if dims == 2 else nn.InstanceNorm3d


class ConvBlock(nn.Module):
    def __init__(self, dims: int, cin: int, cout: int, stride: int = 1):
        super().__init__()
        conv, norm = conv_nd(dims), norm_nd(dims)
        self.net = nn.Sequential(
            conv(cin, cout, 3, stride, 1, bias=False),
            norm(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            conv(cout, cout, 3, 1, 1, bias=False),
            norm(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 4,
        out_channels: int = 3,
        features=(32, 64, 128, 256, 320),
        vit_bottleneck=None,
        patch_size=None,
    ):
        super().__init__()
        if vit_bottleneck:
            raise ValueError("The exported demo checkpoints do not use a ViT bottleneck.")
        self.dims, self.features = spatial_dims, list(features)
        self.encoders = nn.ModuleList()
        cin = in_channels
        for index, feature_count in enumerate(features):
            self.encoders.append(
                ConvBlock(spatial_dims, cin, feature_count, stride=1 if index == 0 else 2)
            )
            cin = feature_count
        self.vit = None
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(features) - 1, 0, -1):
            self.ups.append(
                conv_t_nd(spatial_dims)(features[index], features[index - 1], 2, 2)
            )
            self.decoders.append(
                ConvBlock(spatial_dims, features[index - 1] * 2, features[index - 1])
            )
        self.head = conv_nd(spatial_dims)(features[0], out_channels, 1)

    @property
    def decoder_channels(self):
        return [self.features[-1]] + self.features[-2::-1]

    def forward_features(self, x: torch.Tensor):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        features = [x]
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = decoder(torch.cat([up(x), skip], dim=1))
            features.append(x)
        return features

    def forward(self, x: torch.Tensor):
        return {"logits": self.head(self.forward_features(x)[-1])}


class SinePositionalEncoding(nn.Module):
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        self.dim, self.temperature = dim, temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        dimensions = len(spatial)
        per_dimension = self.dim // (2 * dimensions)
        coordinates = torch.meshgrid(
            *[
                torch.linspace(0, 2 * math.pi, size, device=x.device)
                for size in spatial
            ],
            indexing="ij",
        )
        frequency = self.temperature ** (
            torch.arange(per_dimension, device=x.device, dtype=torch.float32)
            / per_dimension
        )
        features = []
        for coordinate in coordinates:
            values = coordinate[None] / frequency.view(-1, *([1] * dimensions))
            features += [values.sin(), values.cos()]
        encoding = torch.cat(features, dim=0)
        if encoding.shape[0] < self.dim:
            padding = torch.zeros(
                self.dim - encoding.shape[0], *spatial, device=x.device
            )
            encoding = torch.cat([encoding, padding], dim=0)
        return encoding.unsqueeze(0).expand(x.shape[0], -1, *spatial).to(x.dtype)


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, memory, attention_mask, position, query_position):
        output = self.attn(
            query + query_position,
            memory + position,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        return self.norm(query + output)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, query_position):
        output = self.attn(
            query + query_position,
            query + query_position,
            query,
            need_weights=False,
        )[0]
        return self.norm(query + output)


class FFNLayer(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query):
        return self.norm(query + self.net(query))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int):
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(dimensions[:-1], dimensions[1:])
        )

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x) if index == len(self.layers) - 1 else F.relu(layer(x))
        return x


class MaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: list[int],
        num_classes: int = 3,
        hidden: int = 192,
        num_queries: int = 20,
        layers: int = 3,
        heads: int = 8,
        ffn_mult: int = 8,
        masked_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dims, self.heads, self.masked_attn = spatial_dims, heads, masked_attn
        self.num_levels = len(in_channels)
        conv = conv_nd(spatial_dims)
        self.input_proj = nn.ModuleList(
            nn.Sequential(conv(channels, hidden, 1), nn.GroupNorm(32, hidden))
            for channels in in_channels
        )
        self.level_embed = nn.Embedding(self.num_levels, hidden)
        self.pe = SinePositionalEncoding(hidden)
        self.query_feat = nn.Embedding(num_queries, hidden)
        self.query_embed = nn.Embedding(num_queries, hidden)
        self.cross = nn.ModuleList(
            CrossAttentionLayer(hidden, heads, dropout) for _ in range(layers)
        )
        self.selfa = nn.ModuleList(
            SelfAttentionLayer(hidden, heads, dropout) for _ in range(layers)
        )
        self.ffn = nn.ModuleList(
            FFNLayer(hidden, hidden * ffn_mult, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)
        self.class_embed = nn.Linear(hidden, num_classes + 1)
        self.mask_embed = MLP(hidden, hidden, hidden, 3)
        self._pe_cache = {}

    def _heads(self, query, mask_features, target_size):
        query = self.norm(query)
        logits = self.class_embed(query)
        masks = torch.einsum(
            "bqc,bc...->bq...", self.mask_embed(query), mask_features
        )
        attention = None
        if self.masked_attn:
            mode = "bilinear" if self.dims == 2 else "trilinear"
            resized = F.interpolate(
                masks.float(), size=target_size, mode=mode, align_corners=False
            )
            attention = resized.sigmoid().flatten(2) < 0.5
            attention = (
                attention.unsqueeze(1)
                .repeat(1, self.heads, 1, 1)
                .flatten(0, 1)
                .detach()
            )
        return logits, masks, attention

    def forward(self, features, mask_features):
        batch_size = mask_features.shape[0]
        sources, positions, sizes = [], [], []
        for index, feature in enumerate(features):
            sizes.append(feature.shape[2:])
            key = (tuple(feature.shape[2:]), feature.dtype, str(feature.device))
            if key not in self._pe_cache:
                self._pe_cache[key] = (
                    self.pe(feature[:1]).flatten(2).transpose(1, 2)
                )
            positions.append(self._pe_cache[key].expand(batch_size, -1, -1))
            source = self.input_proj[index](feature).flatten(2)
            source = source + self.level_embed.weight[index][None, :, None]
            sources.append(source.transpose(1, 2))

        query = self.query_feat.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_position = self.query_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        outputs = []
        logits, masks, attention = self._heads(
            query, mask_features, sizes[0]
        )
        outputs.append((logits, masks))
        for index in range(len(self.cross)):
            level = index % self.num_levels
            if attention is not None:
                attention = attention & ~attention.all(-1, keepdim=True)
            query = self.cross[index](
                query,
                sources[level],
                attention,
                positions[level],
                query_position,
            )
            query = self.selfa[index](query, query_position)
            query = self.ffn[index](query)
            logits, masks, attention = self._heads(
                query, mask_features, sizes[(index + 1) % self.num_levels]
            )
            outputs.append((logits, masks))
        return {
            "pred_logits": outputs[-1][0],
            "pred_masks": outputs[-1][1],
            "aux_outputs": [
                {"pred_logits": logits, "pred_masks": masks}
                for logits, masks in outputs[:-1]
            ],
        }


class TransUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 4,
        num_classes: int = 3,
        features=(32, 64, 128, 256, 320),
        decoder=None,
        vit_bottleneck=None,
        patch_size=None,
    ):
        super().__init__()
        decoder = dict(decoder or {})
        self.ms_idxs = decoder.pop("ms_idxs", [-4, -3, -2])
        self.unet = UNet(
            spatial_dims,
            in_channels,
            num_classes,
            features,
            vit_bottleneck,
            patch_size,
        )
        channels = self.unet.decoder_channels
        hidden = decoder.get("hidden", 192)
        self.mask_proj = conv_nd(spatial_dims)(channels[-1], hidden, 1)
        self.decoder = MaskedTransformerDecoder(
            spatial_dims,
            [channels[index] for index in self.ms_idxs],
            num_classes=num_classes,
            **decoder,
        )

    def forward(self, x):
        features = self.unet.forward_features(x)
        return self.decoder(
            [features[index] for index in self.ms_idxs],
            self.mask_proj(features[-1]),
        )


def build_model(config: dict) -> nn.Module:
    model_config = dict(config["model"])
    name = model_config.pop("name")
    dimensions = model_config.pop("spatial_dims")
    patch_size = config["train"]["patch_size"]
    if name == "unet":
        return UNet(
            dimensions,
            4,
            3,
            model_config.get("features"),
            model_config.get("vit_bottleneck"),
            patch_size,
        )
    if name == "transunet":
        return TransUNet(
            dimensions,
            4,
            3,
            model_config.get("features"),
            model_config.get("decoder"),
            model_config.get("vit_bottleneck"),
            patch_size,
        )
    raise ValueError(f"Unsupported model: {name}")


def region_probabilities(output: dict) -> torch.Tensor:
    if "logits" in output:
        return output["logits"].float().sigmoid()
    classes = F.softmax(output["pred_logits"].float(), dim=-1)[..., :-1]
    masks = output["pred_masks"].float().sigmoid()
    return torch.einsum("bqc,bq...->bc...", classes, masks).clamp(0, 1)
