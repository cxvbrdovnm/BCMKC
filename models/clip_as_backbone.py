import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from typing import Dict, List

from util.misc import NestedTensor
from models.position_encoding import build_position_encoding

from .san import get_clip_features


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        # self.strides = backbone.strides
        self.num_channels = 256 #backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        _, t = tensor_list.tensors.shape[:2]
        tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w')
        tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')

        #xs = self[0](tensor_list, num_frames=t)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(tensor_list.tensors.items()):
            out.append(x)
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

class CLIP_Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        # available models = ['RN50', 'RN50-quickgelu', 'RN101', 'RN101-quickgelu', 'RN50x4',
        # 'RN50x16', 'RN50x64', 'ViT-B-32', 'ViT-B-32-quickgelu', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336']
        self.clip_features = get_clip_features("ViT-L-14-336")

        # vit-b: 768, vit-l:1024
        clip_img_c = 1024
        # backbone_dim = [192, 384, 768, 768]
        backbone_dim = [128, 256, 512, 1024]
        clip_conv = []
        for l in range(4):
            clip_conv.append(nn.Conv2d(clip_img_c, backbone_dim[l], kernel_size=(2, 2)))
        self.clip_conv = nn.ModuleList(clip_conv)

        self.position_embedding = build_position_encoding(args)

    def forward(self, samples: NestedTensor):
        # num_frames is needed, because we put time in batch dimension.
        # load a whole video and split into clips
        # samples: [B*T, 3, H, W]
        clip_features = self.clip_features(samples)

        # print(samples.tensors.shape)
        h_tar = samples.tensors.shape[-2]
        w_tar = samples.tensors.shape[-1]
        out: List[NestedTensor] = []
        size_hw = [[h_tar//4, w_tar//4], [h_tar//8, w_tar//8], [h_tar//16, w_tar//16], [h_tar//32, w_tar//32]]
        for i in range(4):
            tensors = self.clip_conv[i](clip_features[i * 8]) #bt c h w
            tensors = F.interpolate(
                    tensors,
                    size=size_hw[i],
                    mode='bilinear',
                    align_corners=False
                )
            m = samples.mask
            # print(m.shape) #1 5 128 224
            b, t, h, w = m.shape
            # print(tensors.shape)
            # mask = torch.zeros([bt, h, w], dtype=torch.bool, device=tensors.device)
            mask = F.interpolate(m.view(b*t, h, w).unsqueeze(1).float(), size=tensors.shape[-2:]).squeeze(1).to(torch.bool)
            nested_tensor = NestedTensor(tensors, mask)
            out.append(nested_tensor)

        # for idx, o in out.items():
        #     out[idx] = o

        # for key, value in out.items():
        #     print(f"Key: {key}, Value: {value}")

        # out_p: List[NestedTensor] = []
        #
        # for name, x in sorted(out.items()):
        #     out_p.append(x)


        # position encoding
        pos = []
        for x in out:
            pos.append(self.position_embedding(x).to(x.tensors.dtype))

        return out, pos



def build_clip_backbone(args):
    # position_embedding = build_position_encoding(args)
    backbone = CLIP_Backbone(args)
    # model = Joiner(backbone, position_embedding)
    return backbone