#!/usr/bin/env python3
import os
import torch
from torchvision.models.detection.faster_rcnn import resnet_fpn_backbone, fasterrcnn_resnet50_fpn


os.environ["TORCH_HOME"] = '.'

fasterrcnn_resnet50_fpn()
model = resnet_fpn_backbone(
    True,
)

print(model)
