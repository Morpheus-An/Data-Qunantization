import torch
from torch import Tensor
import inspect
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModel, merge_dict
import torch.nn.functional as F
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList

@MODELS.register_module()
class SamePadConv3d(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)
    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))



