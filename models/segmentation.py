"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

import fvcore.nn.weight_init as weight_init

from .position_encoding import PositionEmbeddingSine1D

BN_MOMENTUM = 0.1

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, text,
                text_key_padding_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None):
        visual = rearrange(visual, 't h w b c -> (t h w) b c')
        visual2 = self.multihead_attn(query=self.with_pos_embed(visual, visual_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        visual = visual * visual2
        return visual

class LanguageVisionFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, text,
                text_key_padding_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None):
        #visual = rearrange(visual, 't h w b c -> (t h w) b c')
        # print(visual.shape)
        # print(text.shape)
        visual2 = self.multihead_attn(query=self.with_pos_embed(visual, visual_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        visual = visual * visual2
        return visual

class LanguageVisionFusionModule_new(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, text, visual,
                visual_key_padding_mask: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None):

        # #get visual prototype
        # bt, c, h, w = visual.shape
        # probs = visual.view(bt, c, -1)
        # ss_map = F.softmax(probs, dim=2) #生成了一个软注意力图，表示每个空间位置在特征通道上的相对重要性
        # ss_map = ss_map.view(bt, c, h, w)
        # # print(ss_map.shape, 'ss map') #5 256 20 28
        # # print(ss_map)
        # # pb = get_prototype(x, ss_map.clone().detach())
        # visual = visual.view(bt, -1, h * w)
        # ss_map = ss_map.view(bt, -1, h * w)
        # visual_proto = torch.bmm(ss_map, visual.transpose(1, 2)) #bt c' c
        # # print(visual_proto.shape, 'visual proto') #5 256 256
        # # print(visual_proto)
        # visual_proto = rearrange(visual_proto, '(b t) c1 c -> (t c1) b c', b=1)
        # print(visual_proto)
        #text = rearrange(text, 't h w b c -> (t h w) b c')
        text2 = self.multihead_attn(query=self.with_pos_embed(text, text_pos),
                                   key=self.with_pos_embed(visual, visual_pos),
                                   value=visual, attn_mask=None,
                                   key_padding_mask=visual_key_padding_mask)[0]
        text = text * text2

        return text


class cross_layer_fusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.selfattn = nn.MultiheadAttention(d_model)
        self.multihead_attn12 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn13 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, langs, text_word_masks):
        # normalization text feature
        text_features_new = []
        tempt = []
        for idx, lang in enumerate(langs): #list(B L C)
            lang_feature = [t_mem[~pad_mask] for t_mem, pad_mask in zip(lang, text_word_masks)]  # [List B S C]
            for obj in lang_feature:
                obj = torch.mean(obj, dim=0)  # [C]
                tempt.append(obj)
            text_features_new.append(torch.stack(tempt, dim=0))  # [B C]
            # text_features_new = torch.stack(text_features_new, dim=0)[0]  # [b, c] #without layer

        # #cross layer attention
        # attention_maps = [self.selfattn(fm) for fm in text_features_new]
        #
        # query = attention_maps[0]
        # key2 = attention_maps[1]
        # value2 = attention_maps[1]
        # key3 = attention_maps[3]
        # value3 = attention_maps[3]
        # b, c = query.shape
        #
        # # 计算注意力权重
        # attention1_2 = self.softmax(torch.bmm(query.unsqueeze(1), key2.unsqueeze(0).transpose(1, 2)) / (c ** 0.5), dim=-1) #b 1 b
        # attention1_3 = self.softmax(torch.bmm(query.unsqueeze(1), key3.unsqueeze(0).transpose(1, 2)) / (c ** 0.5), dim=-1)
        #
        # # 应用注意力权重并融合特征图
        # layer1 = attention_maps[0] + torch.bmm(attention1_2, value2).squeeze(1) #b 1 c -> b c
        # layer1 = layer1 + torch.bmm(attention1_3, value3).squeeze(1)
        text12 = text_features_new[0] * self.multihead_attn12(text_features_new[0], text_features_new[1])
        text123 = text12 + self.multihead_attn13(text12, text_features_new[2])

        return text123

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        return out