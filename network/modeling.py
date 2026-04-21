from collections import OrderedDict
from torch import nn

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception,
    timm_backbone,
)

class MobileNetV2SegBackbone(nn.Module):
    def __init__(self, backbone, return_mid_level=True):
        super().__init__()
        self.backbone = backbone
        self.return_mid_level = return_mid_level

    def forward(self, x):
        low_level_feat, mid_level_feat, high_level_feat = self.backbone.forward_features(x, return_multi=True)
        out = OrderedDict()
        out["out"] = high_level_feat
        out["low_level"] = low_level_feat
        if self.return_mid_level:
            out["mid_level"] = mid_level_feat
        return out

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone, attention_type='none', **kwargs):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate, attention_type=attention_type, **kwargs
        )
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, attention_type=attention_type, **kwargs)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, attention_type='none', **kwargs):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256
    mid_level_planes = 512

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level', 'layer2': 'mid_level'}
        classifier = DeepLabHeadV3Plus(
            inplanes,
            low_level_planes,
            num_classes,
            aspp_dilate,
            attention_type=attention_type,
            mid_level_channels=mid_level_planes,
            **kwargs
        )
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, attention_type=attention_type, **kwargs)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone, attention_type='none', **kwargs):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    mid_level_planes = 256
    
    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level', 'block2': 'mid_level'}
        classifier = DeepLabHeadV3Plus(
            inplanes,
            low_level_planes,
            num_classes,
            aspp_dilate,
            attention_type=attention_type,
            mid_level_channels=mid_level_planes,
            **kwargs
        )
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, attention_type=attention_type, **kwargs)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone, attention_type='none', **kwargs):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    mobilenet_backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    inplanes = 320
    low_level_planes = 24
    mid_level_planes = 32
    
    if name=='deeplabv3plus':
        classifier = DeepLabHeadV3Plus(
            inplanes,
            low_level_planes,
            num_classes,
            aspp_dilate,
            attention_type=attention_type,
            mid_level_channels=mid_level_planes,
            **kwargs
        )
        backbone = MobileNetV2SegBackbone(mobilenet_backbone, return_mid_level=True)
    elif name=='deeplabv3':
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, attention_type=attention_type, **kwargs)
        backbone = MobileNetV2SegBackbone(mobilenet_backbone, return_mid_level=False)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_timm(name, timm_name, num_classes, output_stride, pretrained_backbone, attention_type='none', **kwargs):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    # Not all timm models support output_stride argument.
    timm_output_stride = output_stride
    if "swin" in timm_name:
        timm_output_stride = None

    backbone = timm_backbone.TimmBackbone(
        model_name=timm_name,
        pretrained=pretrained_backbone,
        output_stride=timm_output_stride,
    )
    inplanes = backbone.out_channels
    low_level_planes = backbone.low_level_channels
    mid_level_planes = backbone.mid_level_channels

    if name == 'deeplabv3plus':
        classifier = DeepLabHeadV3Plus(
            inplanes,
            low_level_planes,
            num_classes,
            aspp_dilate,
            attention_type=attention_type,
            mid_level_channels=mid_level_planes,
            **kwargs
        )
    elif name == 'deeplabv3':
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, attention_type=attention_type, **kwargs)
    else:
        raise NotImplementedError

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, attention_type='none', **kwargs):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)
    elif backbone in ('convnext_tiny', 'convnext_small', 'swin_tiny_patch4_window7_224', 'efficientnet_b3'):
        model = _segm_timm(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, attention_type='none', **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, attention_type='none', **kwargs):
    return _load_model('deeplabv3', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, attention_type='none', **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, attention_type='none', **kwargs):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


def deeplabv3plus_convnext_tiny(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    return _load_model('deeplabv3plus', 'convnext_tiny', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


def deeplabv3plus_convnext_small(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    return _load_model('deeplabv3plus', 'convnext_small', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


def deeplabv3plus_swin_tiny(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    return _load_model('deeplabv3plus', 'swin_tiny_patch4_window7_224', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)


def deeplabv3plus_efficientnet_b3(num_classes=21, output_stride=8, pretrained_backbone=True, attention_type='none', **kwargs):
    return _load_model('deeplabv3plus', 'efficientnet_b3', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, attention_type=attention_type, **kwargs)