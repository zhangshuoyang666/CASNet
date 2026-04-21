from collections import OrderedDict

import torch
import torch.nn as nn
import timm


class TimmBackbone(nn.Module):
    """Wrap timm feature extractor and return DeepLab-compatible feature dict."""

    def __init__(self, model_name, pretrained=True, output_stride=None):
        super().__init__()
        create_kwargs = {
            "features_only": True,
            "pretrained": pretrained,
        }
        if output_stride is not None:
            create_kwargs["output_stride"] = output_stride
        try:
            # Helps transformer-style models expose NCHW tensors.
            create_kwargs["output_fmt"] = "NCHW"
            self.model = timm.create_model(model_name, **create_kwargs)
        except TypeError:
            create_kwargs.pop("output_fmt", None)
            self.model = timm.create_model(model_name, **create_kwargs)

        channels = self.model.feature_info.channels()
        reductions = self.model.feature_info.reduction()
        if len(channels) < 2 or len(reductions) < 2:
            raise ValueError("timm backbone {} returned insufficient feature maps".format(model_name))

        self.low_level_idx = 0
        for i, r in enumerate(reductions):
            if r >= 4:
                self.low_level_idx = i
                break
        self.out_idx = len(channels) - 1
        if self.low_level_idx >= self.out_idx:
            self.low_level_idx = 0

        self.mid_level_idx = min(self.out_idx - 1, max(self.low_level_idx + 1, len(channels) // 2))
        for i, r in enumerate(reductions):
            if reductions[self.low_level_idx] < r < reductions[self.out_idx]:
                self.mid_level_idx = i
                break
        if self.mid_level_idx <= self.low_level_idx or self.mid_level_idx >= self.out_idx:
            self.mid_level_idx = max(self.low_level_idx, self.out_idx - 1)

        self.low_level_channels = channels[self.low_level_idx]
        self.mid_level_channels = channels[self.mid_level_idx]
        self.out_channels = channels[self.out_idx]

    def _to_nchw(self, feat, expected_channels):
        if feat.ndim == 4:
            if feat.shape[1] == expected_channels:
                return feat
            if feat.shape[-1] == expected_channels:
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        feats = self.model(x)
        low_level = self._to_nchw(feats[self.low_level_idx], self.low_level_channels)
        mid_level = self._to_nchw(feats[self.mid_level_idx], self.mid_level_channels)
        out = self._to_nchw(feats[self.out_idx], self.out_channels)
        return OrderedDict(low_level=low_level, mid_level=mid_level, out=out)

