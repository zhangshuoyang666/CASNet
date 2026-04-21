import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(
        self,
        in_channels,
        low_level_channels,
        num_classes,
        aspp_dilate=[12, 24, 36],
        attention_type='none',
        mid_level_channels=None,
        aspp_variant='standard',
        dense_aspp_rates=(1, 3, 6, 12, 18),
        enable_fg_fusion=False,
        enable_texture_enhance=False,
        enable_decoder_detail=False,
        enable_boundary_aux=False,
        use_saff=False,
        saff_f1_source='mid',
        saff_f2_source='aspp',
    ):
        super(DeepLabHeadV3Plus, self).__init__()
        self.enable_fg_fusion = enable_fg_fusion
        self.enable_texture_enhance = enable_texture_enhance
        self.enable_decoder_detail = enable_decoder_detail
        self.use_saff = use_saff
        self.saff_f1_source = (saff_f1_source or 'mid').lower()
        self.saff_f2_source = (saff_f2_source or 'aspp').lower()
        self.use_legacy_decoder = (
            (aspp_variant or 'standard').lower() == 'standard'
            and not enable_fg_fusion
            and not enable_texture_enhance
            and not enable_decoder_detail
        )
        self.enable_boundary_aux = enable_boundary_aux

        if mid_level_channels is None:
            mid_level_channels = low_level_channels

        if self.enable_fg_fusion:
            self.fg_fusion = FineGrainedFusionModule(
                low_channels=low_level_channels,
                mid_channels=mid_level_channels,
                high_channels=in_channels,
            )
        else:
            self.fg_fusion = None

        if self.enable_texture_enhance:
            self.texture_enhance = TextureEnhanceModule(in_channels)
        else:
            self.texture_enhance = None

        self.aspp = build_aspp(
            in_channels=in_channels,
            atrous_rates=aspp_dilate,
            attention_type=attention_type,
            aspp_variant=aspp_variant,
            dense_aspp_rates=dense_aspp_rates,
        )
        f1_channel_map = {
            'low': low_level_channels,
            'mid': mid_level_channels,
            'high': in_channels,
        }
        f2_channel_map = {
            'aspp': 256,
            'high': in_channels,
        }
        if self.saff_f1_source not in f1_channel_map:
            raise ValueError("Unsupported saff_f1_source: {}".format(saff_f1_source))
        if self.saff_f2_source not in f2_channel_map:
            raise ValueError("Unsupported saff_f2_source: {}".format(saff_f2_source))
        self.saff = SAFFModule(
            f1_channels=f1_channel_map[self.saff_f1_source],
            f2_channels=f2_channel_map[self.saff_f2_source],
            out_channels=256,
        ) if self.use_saff else None

        if not self.use_legacy_decoder:
            self.decoder = ImprovedDecoder(
                low_level_channels=low_level_channels,
                num_classes=num_classes,
                enable_detail_path=enable_decoder_detail,
                detail_channels=48,
                mid_level_channels=mid_level_channels,
            )
        else:
            self.decoder = None

        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        boundary_in_channels = 304 if self.use_legacy_decoder else 256
        self.boundary_head = BoundaryAuxHead(in_channels=boundary_in_channels) if self.enable_boundary_aux else None
        self._init_weight()

    def forward(self, feature):
        high_level_feature = feature['out']
        low_level_feature = feature['low_level']
        mid_level_feature = feature.get('mid_level', low_level_feature)
        detail_feature = None

        if self.fg_fusion is not None:
            high_level_feature, detail_feature = self.fg_fusion(
                low_level_feature, mid_level_feature, high_level_feature
            )
        if self.texture_enhance is not None:
            high_level_feature = self.texture_enhance(high_level_feature)

        output_feature = self.aspp(high_level_feature)
        if self.saff is not None:
            if self.saff_f1_source == 'low':
                saff_f1 = low_level_feature
            elif self.saff_f1_source == 'mid':
                saff_f1 = mid_level_feature
            else:
                saff_f1 = high_level_feature

            if self.saff_f2_source == 'high':
                saff_f2 = high_level_feature
            else:
                saff_f2 = output_feature
            output_feature = self.saff(saff_f1, saff_f2)

        boundary_feature = None
        if self.decoder is not None:
            if self.boundary_head is not None:
                seg_logits, boundary_feature = self.decoder(
                    low_level=low_level_feature,
                    aspp_feature=output_feature,
                    mid_level=mid_level_feature,
                    detail_feature=detail_feature,
                    return_feature=True,
                )
            else:
                seg_logits = self.decoder(
                    low_level=low_level_feature,
                    aspp_feature=output_feature,
                    mid_level=mid_level_feature,
                    detail_feature=detail_feature,
                )
        else:
            low_level_feature = self.project(low_level_feature)
            output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
            boundary_feature = torch.cat([low_level_feature, output_feature], dim=1)
            seg_logits = self.classifier(boundary_feature)

        if self.boundary_head is not None:
            edge_logits = self.boundary_head(boundary_feature)
            return {"seg_logits": seg_logits, "edge_logits": edge_logits}
        return seg_logits
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        aspp_dilate=[12, 24, 36],
        attention_type='none',
        aspp_variant='standard',
        dense_aspp_rates=(1, 3, 6, 12, 18),
    ):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            build_aspp(
                in_channels=in_channels,
                atrous_rates=aspp_dilate,
                attention_type=attention_type,
                aspp_variant=aspp_variant,
                dense_aspp_rates=dense_aspp_rates,
            ),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BoundaryAuxHead(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )
        self._init_weight()

    def forward(self, x):
        return self.block(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class IdentityAttention(nn.Module):
    def forward(self, x):
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(attn)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attn)

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))

class CBAMAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMAttention, self).__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel(x)
        return self.spatial(x)

class SpatialCbamAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SpatialCbamAttention, self).__init__()
        self.cbam = CBAMAttention(channels, reduction=reduction)
        self.spatial_refine = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.cbam(x)
        return self.spatial_refine(x)

class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

def build_attention(attention_type, channels):
    attn = (attention_type or 'none').lower()
    if attn == 'none':
        return IdentityAttention()
    if attn == 'channel':
        return ChannelAttention(channels)
    if attn == 'spatial':
        return SpatialAttention()
    if attn == 'cbam':
        return CBAMAttention(channels)
    if attn == 'spatial_cbam':
        return SpatialCbamAttention(channels)
    if attn == 'cbam_light':
        return CBAMAttention(channels)
    if attn == 'se':
        return SEAttention(channels)
    if attn in ('ca', 'coordinate'):
        return CoordinateAttention(channels)
    raise ValueError("Unsupported attention_type: {}".format(attention_type))

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, attention_type='none'):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.attention = build_attention(attention_type, 5 * out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.attention(res)
        return self.project(res)


class DeformLikeASPPConv(nn.Module):
    """
    Lightweight deform-like branch:
    predict offsets -> warp feature map -> atrous conv.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.offset = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        offset = self.offset(x)
        offset = torch.tanh(offset) * 2.0

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing='ij',
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        norm_offset = torch.zeros_like(base_grid)
        norm_offset[..., 0] = offset[:, 0] / max(float(w - 1), 1.0) * 2.0
        norm_offset[..., 1] = offset[:, 1] / max(float(h - 1), 1.0) * 2.0
        sampling_grid = base_grid + norm_offset
        warped = F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return self.conv(warped)


class DenseASPP(nn.Module):
    def __init__(self, in_channels, dense_rates=(1, 3, 6, 12, 18), attention_type='none'):
        super().__init__()
        out_channels = 256
        branch_channels = 128

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList()
        running_in = out_channels
        for rate in dense_rates:
            self.branches.append(
                ASPPConv(running_in, branch_channels, dilation=rate)
            )
            running_in += branch_channels
        self.global_pool = ASPPPooling(in_channels, branch_channels)
        total_channels = out_channels + len(dense_rates) * branch_channels + branch_channels
        self.attention = build_attention(attention_type, total_channels)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self._init_weight()

    def forward(self, x):
        base = self.stem(x)
        feats = [base]
        for branch in self.branches:
            branch_out = branch(torch.cat(feats, dim=1))
            feats.append(branch_out)
        pooled = self.global_pool(x)
        concat = torch.cat(feats + [pooled], dim=1)
        concat = self.attention(concat)
        return self.project(concat)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeformASPP(nn.Module):
    """
    Hybrid ASPP: standard atrous branches + deform-like branch.
    """
    def __init__(self, in_channels, atrous_rates, attention_type='none'):
        super().__init__()
        out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
            ASPPConv(in_channels, out_channels, rate1),
            ASPPConv(in_channels, out_channels, rate2),
            DeformLikeASPPConv(in_channels, out_channels, rate3),
            ASPPPooling(in_channels, out_channels),
        ])
        self.attention = build_attention(attention_type, 5 * out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self._init_weight()

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        res = self.attention(res)
        return self.project(res)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FineGrainedFusionModule(nn.Module):
    """
    Multi-level fusion where high-level semantics suppress low-level noise while
    keeping discriminative texture/edge cues.
    """
    def __init__(self, low_channels, mid_channels, high_channels, inner_channels=256):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.Sigmoid(),
        )
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(inner_channels * 3, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, 3, 1, bias=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(inner_channels * 3, high_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
        )
        self.detail_proj = nn.Sequential(
            nn.Conv2d(inner_channels * 2, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self._init_weight()

    def forward(self, low_level, mid_level, high_level):
        target_size = high_level.shape[-2:]
        low = self.low_proj(low_level)
        mid = self.mid_proj(mid_level)
        high = self.high_proj(high_level)
        low = F.interpolate(low, size=target_size, mode='bilinear', align_corners=False)
        mid = F.interpolate(mid, size=target_size, mode='bilinear', align_corners=False)

        semantic_mask = self.semantic_gate(high)
        guided_low = low * semantic_mask
        guided_mid = mid * semantic_mask

        fusion_input = torch.cat([guided_low, guided_mid, high], dim=1)
        fusion_weight = torch.softmax(self.fusion_weight(fusion_input), dim=1)
        fused = (
            guided_low * fusion_weight[:, 0:1]
            + guided_mid * fusion_weight[:, 1:2]
            + high * fusion_weight[:, 2:3]
        )
        enhanced = self.out_proj(torch.cat([fused, high, guided_mid], dim=1))
        detail = self.detail_proj(torch.cat([guided_low, guided_mid], dim=1))
        return enhanced, detail

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class TextureEnhanceModule(nn.Module):
    """
    Lightweight high/low-frequency decomposition and fusion to retain texture cues.
    """
    def __init__(self, channels):
        super().__init__()
        self.low_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.high_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=max(1, channels // 16), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self._init_weight()

    def forward(self, x):
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq
        low_feat = self.low_encoder(low_freq)
        high_feat = self.high_encoder(high_freq)
        gate = self.fusion_gate(torch.cat([low_feat, high_feat], dim=1))
        fused_high = high_feat * gate
        return self.out_proj(torch.cat([low_feat, fused_high], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ImprovedDecoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, enable_detail_path=False, detail_channels=48, mid_level_channels=None):
        super().__init__()
        self.enable_detail_path = enable_detail_path
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.mid_proj = None
        if enable_detail_path:
            if mid_level_channels is None:
                mid_level_channels = low_level_channels
            self.mid_proj = nn.Sequential(
                nn.Conv2d(mid_level_channels, detail_channels, 1, bias=False),
                nn.BatchNorm2d(detail_channels),
                nn.ReLU(inplace=True),
            )
            in_ch = 256 + 48 + detail_channels
        else:
            in_ch = 256 + 48
        self.classifier = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, low_level, aspp_feature, mid_level=None, detail_feature=None, return_feature=False):
        low = self.low_proj(low_level)
        aspp_feature = F.interpolate(aspp_feature, size=low.shape[2:], mode='bilinear', align_corners=False)
        feats = [low, aspp_feature]
        if self.enable_detail_path:
            if detail_feature is None:
                if mid_level is None:
                    detail_feature = low
                else:
                    detail_feature = self.mid_proj(mid_level)
            detail_feature = F.interpolate(detail_feature, size=low.shape[2:], mode='bilinear', align_corners=False)
            feats.append(detail_feature)
        merged = torch.cat(feats, dim=1)
        decoder_feature = merged
        for layer in self.classifier[:-1]:
            decoder_feature = layer(decoder_feature)
        logits = self.classifier[-1](decoder_feature)
        if return_feature:
            return logits, decoder_feature
        return logits

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SAFFModule(nn.Module):
    """
    Shallow Advanced Feature Fusion:
      F1(mid-level) -> 1x1 Conv+BN+ReLU
      F2(ASPP out)  -> 2x bilinear upsample -> 3x3 dilated Conv(d=3,p=3)+BN+ReLU
      sum -> 1x1 Conv+BN+ReLU
    """
    def __init__(self, f1_channels, f2_channels=256, out_channels=256):
        super().__init__()
        self.f1_proj = nn.Sequential(
            nn.Conv2d(f1_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.f2_refine = nn.Sequential(
            nn.Conv2d(f2_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weight()

    def forward(self, f1, f2):
        f1 = self.f1_proj(f1)
        f2 = F.interpolate(f2, scale_factor=2.0, mode='bilinear', align_corners=False)
        f2 = self.f2_refine(f2)
        if f2.shape[-2:] != f1.shape[-2:]:
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        fused = f1 + f2
        return self.out_proj(fused)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_aspp(in_channels, atrous_rates, attention_type='none', aspp_variant='standard', dense_aspp_rates=(1, 3, 6, 12, 18)):
    variant = (aspp_variant or 'standard').lower()
    if variant == 'standard':
        return ASPP(in_channels, atrous_rates, attention_type=attention_type)
    if variant == 'dense':
        return DenseASPP(in_channels, dense_rates=dense_aspp_rates, attention_type=attention_type)
    if variant == 'deform':
        return DeformASPP(in_channels, atrous_rates=atrous_rates, attention_type=attention_type)
    raise ValueError("Unsupported aspp_variant: {}".format(aspp_variant))



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module