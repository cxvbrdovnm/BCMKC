
from typing import List

import clip
from clip.factory import create_model
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
import logging

from .clip_utils import (
    FeatureExtractor,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    RecWithAttnbiasHead,
    get_predefined_templates,
    build_fusion_layer
)

from einops import rearrange, repeat

class get_clip_features(nn.Module):
    def __init__(self, backbone_name):
        super(get_clip_features, self).__init__()

        #"resnet50" "ViT-B/16"
        clip_model, preprocess = create_model(
            backbone_name, pretrained="openai")

        self.ov_classifier = LearnableBgOvClassifier(
            clip_model, templates=get_predefined_templates("vild")
        )

        self.clip_visual_extractor = FeatureExtractor(
            clip_model.visual,
            last_layer_idx=9,
            frozen_exclude=["positional_embedding"],
        )

        self.clip_rec_head = RecWithAttnbiasHead(
            clip_model.visual,
            first_layer_idx=9,
            frozen_exclude=[],
            cross_attn=False,
            sos_token_format="cls_token",
            sos_token_num=100,
            downsample_method="max",
        )

        #todo
        '''pixel_mean, pixel_std = (
            preprocess.transforms[-1].mean,
            preprocess.transforms[-1].std,
        )'''
        pixel_mean, pixel_std = (preprocess['mean'], preprocess['std'])

        pixel_mean = [255.0 * x for x in pixel_mean]
        pixel_std = [255.0 * x for x in pixel_std]
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.size_divisibility = 32
        self.asymetric_input = True
        self.clip_resolution = 0.5


    def forward(self,samples):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        #todo
        #保证是同一个数据集
        '''if "vocabulary" in targets[0]:
            ov_classifier_weight = (
                    self.ov_classifier.logit_scale.exp()
                    * self.ov_classifier.get_classifier_by_vocabulary(
                targets[0]["vocabulary"]
            )
            )
        else:
            dataset_names = [x["meta"]["dataset_name"] for x in targets]
            assert (
                    len(list(set(dataset_names))) == 1
            ), "All images in a batch must be from the same dataset."
            ov_classifier_weight = (
                    self.ov_classifier.logit_scale.exp()
                    * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
            )  # C+1,ndim'''


        #print(samples.tensors.shape) #5 3 160 224 t 3 h w

        # prepare image for clip
        # images = [x["image"].to(self.device) for x in samples.tensors]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)
        # print(self.pixel_mean.shape, self.pixel_std.shape) #3 1 1, 3 1 1
        images = [(x - self.pixel_mean) / self.pixel_std for x in samples.tensors]

        clip_input = torch.stack(images, dim=0)  # 5 3 160 224
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear", align_corners=False
            )

        clip_image_features = self.clip_visual_extractor(clip_input)
        #print(len(clip_image_features))
        #20 一共10层残差网络，【0-9】为bt c h w， 【0-9_cls_token】为1 bt c



        return clip_image_features




class prepare_srcs(nn.Module):
    def __init__(self, token_num_queries=100, hidden_dim=256):
        super(prepare_srcs, self).__init__()

        # add query token
        self.num_features = hidden_dim
        self.query_embed = nn.Parameter(torch.zeros(1, token_num_queries, self.num_features))
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, token_num_queries, self.num_features)
        )
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)

        self.src_norm = nn.LayerNorm(self.num_features)

    def forward(self, srcs, poses):
        '''
        srcs: bt hw c
        poses: bt hw c
        clip_image_features:
        '''

        bt, hw, c = srcs.size()

        # 将查询位置嵌入（query_pos_embed）与调整后的位置嵌入合并
        pos_embed = torch.cat(
            [poses, self.query_pos_embed.expand(bt, -1, -1)], dim=1
        )  # bt 100+hw c

        # 将查询嵌入（query_embed）与图像分块合并
        srcs = torch.cat(
            [srcs, self.query_embed.expand(bt, -1, -1)],
            dim=1,
        )  # bt, 100+hw, c

        # 将位置嵌入添加到图像分块和查询嵌入的合并结果中
        #srcs = srcs + pos_embed

        #x = self.vit_model.norm_pre(x)
        srcs = self.src_norm(srcs)

        return srcs, pos_embed


class fuse(nn.Module):
    def __init__(self, hidden_dim):
        super(fuse, self).__init__()

        self.fusion_map = ["0->0", "3->1", "6->2", "9->3"]

        # build fusion layers
        fusion_type: str = "add"
        x2side_map = {int(j): int(i) for i, j in [x.split("->") for x in self.fusion_map]} #1:3 2:6
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer(
                    fusion_type, hidden_dim, hidden_dim
                )
                for tgt_idx, src_idx in x2side_map.items()
            }
        )

        self.fusion_layers = fusion_layers
        self.conv = nn.Conv2d(768, 256, kernel_size=(2,2))

    def forward(self, block_idx, srcs, clip_features):
        #srcs: bt hw c
        #print(self.fusion_map) #['0->0', '3->1', '6->2', '9->3']
        #print(block_idx) #0
        #clip_idx = self.fusion_map[block_idx]
        #print('clip_idx', clip_idx) #  clip id 0->0
        #print('block_idx', block_idx) #decoder id 0

        clip_idx = block_idx * 3
        #L = (spatial_shape[block_idx][0] * spatial_shape[block_idx][1]) #hw 16*28

        # 将ViT模型的输出特征与经过融合层处理的CLIP特征拼接在一起。
        # 这里使用 torch.cat 沿着第一个维度（通常是批量大小维度）拼接张量
        bt, l, c = srcs.shape #l=L+100
        #print(srcs.shape) #5 696 256
        #print(clip_features[clip_idx].shape) # 5 768 4 7
        clip_feature = self.conv(clip_features[clip_idx]).flatten(2).unsqueeze(2)
        #print(clip_feature.shape) #5 256 1 18

        '''x = torch.cat(
    [
                srcs[:, :-L, ...],
                self.fusion_layers[f"layer_{block_idx}"](
                    srcs[:, -L:, ...], clip_feature, (1, L)
                ),
            ],
            dim=1,
        )

        log_first_n(
            logging.INFO,
            f"fuse clip {clip_idx} to {block_idx}",
            len(self.fusion_map),
        )

        #print(x.shape) 5 696 256

        return x[:, -L:, ...]'''
        x = self.fusion_layers[f"layer_{block_idx}"](
                    srcs, clip_feature, (1, l)
                )

        return x
