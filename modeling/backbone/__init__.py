from modeling.backbone import resnet, xception, drn, mobilenet
import torch.nn as nn

def Norm(planes):
	return nn.GroupNorm(32, planes)

def build_backbone(backbone, output_stride, BatchNorm):
    if BatchNorm == None:
        BatchNorm = Norm

    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_38(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
