
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
from .segmentation import VisionLanguageFusionModule

from .clip_utils import (
    FeatureExtractor,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    RecWithAttnbiasHead,
    get_predefined_templates,
    build_fusion_layer
)
from models.video_clip import video_header, sentence_text_logit
from models.text_prompt import text_prompt
from einops import rearrange, repeat

from transformers import CLIPModel
import open_clip
from torchvision import transforms

class get_clip_features(nn.Module):
    def __init__(self, backbone_name):
        super(get_clip_features, self).__init__()

        #"resnet50" "ViT-B/16"
        clip_model, preprocess = create_model(
            backbone_name, pretrained="openai")

        # arch = 'TinyCLIP-ViT-61M-32-Text-29M'
        # clip_model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

        # self.ov_classifier = LearnableBgOvClassifier(
        #     clip_model, templates=get_predefined_templates("vild")
        # )

        self.clip_visual_extractor = FeatureExtractor(
            clip_model.visual,
            last_layer_idx=-1,
            frozen_exclude=["positional_embedding"],
        )

        # self.clip_rec_head = RecWithAttnbiasHead(
        #     clip_model.visual,
        #     first_layer_idx=24,
        #     frozen_exclude=[],
        #     cross_attn=False,
        #     sos_token_format="cls_token",
        #     sos_token_num=100,
        #     downsample_method="max",
        # )

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

        #tinyclip
        # # 遍历 preprocess 的转换，找到 Normalize 转换
        # pixel_mean = None
        # pixel_std = None
        # for transform in preprocess.transforms:
        #     if isinstance(transform, transforms.Normalize):
        #         pixel_mean = transform.mean
        #         pixel_std = transform.std
        #         break
        #
        # if pixel_mean is None or pixel_std is None:
        #     raise ValueError("Normalize transformation not found in preprocess")
        #
        # # 将 mean 和 std 转换为 255.0 的倍数
        # pixel_mean = [255.0 * x for x in pixel_mean]
        # pixel_std = [255.0 * x for x in pixel_std]
        #
        # # 注册为缓冲区
        # self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        # self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

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
        # print(len(images), images[0].shape) #1 [5 3 192 224]

        clip_input = torch.stack(images, dim=0)  # 5 3 160 224
        # clip_input = images[0]
        # print(clip_input.shape, 'clip_input')
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear", align_corners=False
            )

        clip_image_features = self.clip_visual_extractor(clip_input)
        #print(len(clip_image_features))
        #20 一共10层残差网络，【0-9】为bt c h w， 【0-9_cls_token】为1 bt c



        return clip_image_features


class san_bike(nn.Module):
    def __init__(self, backbone_name, args):
        super(san_bike, self).__init__()

        #"resnet50" "ViT-B/16"
        clip_model, preprocess = create_model(
            backbone_name, pretrained="openai")

        self.clip_visual_extractor = FeatureExtractor(
            clip_model.visual,
            last_layer_idx=-1,
            frozen_exclude=["positional_embedding"],
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

        # 获取模型的状态字典
        clip_state_dict = clip_model.state_dict()
        self.video_head = video_header(
            args.sim_header,
            args.bike_clip_interaction,
            clip_state_dict)

        self.classes = text_prompt(args.classnames)

        for param in clip_model.parameters():
            param.requires_grad = False

        self.clip_model = clip_model


    def forward(self,samples, captions):
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
        # print(len(images), images[0].shape) #1 [5 3 192 224]

        clip_input = torch.stack(images, dim=0)  # 5 3 160 224
        # clip_input = images[0]
        # print(clip_input.shape, 'clip_input')
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear", align_corners=False
            )

        clip_image_features = self.clip_visual_extractor(clip_input)
        #print(len(clip_image_features))
        #20 一共10层残差网络，【0-9】为bt c h w， 【0-9_cls_token】为1 bt c

        # return clip_image_features
        all_class = self.classes.to("cuda")
        text_tokens = clip.tokenize(all_class)
        cls_emb, text_emb = self.clip_model.encode_text(text_tokens) #89 512  89 77 512
        image_emb = self.clip_model.encode_image(clip_input) # 5 512

        images_fusion, S_V = self.video_head(image_emb, text_emb, cls_emb)
        clip_visual = self.clip_visual_linear(image_emb.float())  # bt c 512->256
        # 视频与哪个类别最相关的索引
        max_index = torch.argmax(S_V)
        clip_visual_text = self.clip_visual_text_linear(images_fusion[max_index].float())
        clip_cls_new = self.clip_cls_linear((cls_emb[max_index]).unsqueeze(0)).unsqueeze(0)

        return clip_image_features, clip_visual_text, clip_cls_new






class prepare_query(nn.Module):
    def __init__(self, token_num_queries=100, hidden_dim=256):
        super(prepare_query, self).__init__()

        # add query token
        self.num_features = hidden_dim
        self.query_embed = nn.Parameter(
            torch.zeros(1, 1, token_num_queries, self.num_features))
        self.query_pos_embed = nn.Parameter(
            torch.zeros(token_num_queries, self.num_features)
        )
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)

        self.query_norm = nn.LayerNorm(self.num_features)
        self.pos_norm = nn.LayerNorm(self.num_features)

    def forward(self, query, query_emb):

        b, t, q, c = query.size()
        # 将查询位置嵌入（query_pos_embed）与调整后的位置嵌入合并
        pos_embed = torch.cat(
            [query_emb, self.query_pos_embed], dim=0)  # q+100 c

        # 将查询嵌入（query_embed）与查询合并
        add_query = torch.cat(
            [query, self.query_embed.expand(b, t, -1, -1)],
            dim=2,
        )  # b, t, q+100, c
        add_query = self.query_norm(add_query)
        pos_embed = self.pos_norm(pos_embed)



        return add_query, pos_embed


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

        #vit-b: 768, vit-l:1024 #tinyclip: 640
        clip_img_c = 768
        #tinyclip todo kernel 2->1
        clip_conv = []
        for l in range(3):
            clip_conv.append(nn.Conv2d(clip_img_c, 256*(2**l), kernel_size=(2,2)))
        clip_conv.append(nn.Conv2d(clip_img_c, 1024, kernel_size=(2,2)))
        self.clip_conv = nn.ModuleList(clip_conv)

        self.linear = nn.Linear(clip_img_c, 256)
        self.feature_fuse = VisionLanguageFusionModule

    def forward(self, block_idx, srcs, clip_features, clip_cls_new, image=True):
        #srcs: bt hw c
        #print(self.fusion_map) #['0->0', '3->1', '6->2', '9->3']
        #print(block_idx) #0
        #clip_idx = self.fusion_map[block_idx]
        #print('clip_idx', clip_idx) #  clip id 0->0
        #print('block_idx', block_idx) #decoder id 0

        # todo
        # tinyclip: 26 layers
        #clip_idx = block_idx * 3
        clip_idx = block_idx * 8
        # clip_idx = block_idx * 4
        #L = (spatial_shape[block_idx][0] * spatial_shape[block_idx][1]) #hw 16*28

        # 将ViT模型的输出特征与经过融合层处理的CLIP特征拼接在一起。
        # 这里使用 torch.cat 沿着第一个维度（通常是批量大小维度）拼接张量
        bt, L, c = srcs.shape
        #print(srcs.shape) #5 696 256
        # print(clip_features[clip_idx].shape, 'clip img') # 5 768 4 7 bt c h w

        if image:
            # clip_feature = self.clip_conv[block_idx](clip_features[clip_idx]).flatten(2).unsqueeze(2)
            #print(clip_feature.shape) #5 256 1 18

            clip_feature = self.clip_conv[block_idx](clip_features[clip_idx].unsqueeze(2))
            # print(clip_feature.shape) #5 256 1 50

            #todo bike
            #将视觉特征重新排列以适应计算，变为(h*w*b, c)
            # visual_features = rearrange(clip_feature, 'bt c h w -> (h w bt) c')
            text_features = clip_cls_new.expand(clip_feature.shape[0], -1, -1, -1)  # bt c h w
            # text_features = rearrange(text_features, 't b l c -> (b t l) c')

            # 将 visual_features 和 cls_features_expanded 调整为二维张量
            # 为了计算余弦相似度，需要将它们调整为 (bt * h * w, c) 的形状
            visual_features_reshaped = clip_feature.view(clip_feature.shape[0], clip_feature.shape[1],
                                                            -1).permute(0, 2, 1)  # bt (h*w) c
            cls_features_reshaped = text_features.view(text_features.shape[0],
                                                               text_features.shape[1], -1).permute(0, 2, 1)  # bt (h*w) c

            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(visual_features_reshaped, cls_features_reshaped, dim=-1)  # bt (h*w)

            # 如果需要，可以将结果调整为 (bt, h, w) 的形状
            cosine_similarity = cosine_similarity.view(clip_feature.shape[0], clip_feature.shape[2],
                                                       clip_feature.shape[3])  # bt h w

            # 归一化相似度到 [0, 1] 范围内
            weights = F.softmax(cosine_similarity, dim=(1, 2))  # bt h w

            # 将权重扩展到与 visual_features 的形状一致
            weights_expanded = weights.unsqueeze(1)  # bt 1 h w

            # 使用权重对 visual_features 进行加权
            weighted_visual_features = clip_feature * weights_expanded  # bt c h w

            clip_feature = weighted_visual_features




            # # 计算余弦相似度
            # cosine_similarity = F.cosine_similarity(visual_features.unsqueeze(1), text_features, dim=2)
            # print(cosine_similarity)
            # # 应用softmax获取帧显著性（Frame Saliency）
            # frame_Saliency = F.softmax(cosine_similarity, dim=1)
            #
            # print(frame_Saliency.shape)
            # # 文本池化（Textual Pooling），这里简单地将帧显著性进行求和
            # textual_pooling = frame_Saliency.sum(dim=1, keepdim=True)
            # print(textual_pooling.shape)
            #
            # # 计算视频嵌入（Video Embedding），通过帧嵌入和帧显著性的加权和
            # video_embedding = (visual_features * textual_pooling).sum(dim=0, keepdim=True)
            #
            # print("Frame Saliency:", frame_Saliency)
            # print("Video Embedding:", video_embedding)
            # print(video_embedding.shape)
            #
            #
            # prin


        else:
            index = str(clip_idx) + "_cls_token"
            #print(clip_features[index].shape, 'clip text') #1 5 768 bt c
            clip_feature = self.linear(clip_features[index])
            #print(clip_feature.shape, 'text clip') #5 256 1 1

        x = self.fusion_layers[f"layer_{block_idx}"](
            block_idx, srcs, clip_feature, (1, L), image
        )

        return x

