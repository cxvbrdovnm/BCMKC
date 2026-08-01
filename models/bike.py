from models.clip import clip as clip
from models.video_clip import video_header, sentence_text_logit
from models.text_prompt import text_prompt
from einops import rearrange, repeat

from typing import List

# import clip
# from clip.factory import create_model
import torch
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

def bike(args):
    # todo get fp16 model and weight
    model, clip_state_dict = clip.load(
        args.bike_clip_name,
        device='cpu', jit=False,
        internal_modeling=args.bike_clip_tm,
        T=args.data_num_segments,
        dropout=args.bike_clip_dropout,
        emb_dropout=args.bike_clip_dropout,
        pretrain=args.bike_clip_init,
        joint_st=args.bike_clip_joint_st)  # Must set jit=False for training  ViT-B/32

    #数据增强
    #transform_train = get_augmentation(True, args)

    video_head = video_header(
        args.sim_header,
        args.bike_clip_interaction,
        clip_state_dict)

    #get all class prompt from clip: 'This is a video about {}'
    classes = text_prompt(args.classnames)
    # n_class = classes.size(0) #89

    # #冻结非video相关的参数
    # if args.bike_clip_fix_text:
    #     for name, param in model.named_parameters():
    #         if "visual" not in name and "logit_scale" not in name:
    #             param.requires_grad_(False)
    #
    # #冻结video相关参数
    # if args.bike_clip_fix_video:
    #     for name, param in model.named_parameters():
    #         if "visual" in name:
    #             param.requires_grad_(False)

    for param in model.parameters():
        param.requires_grad = False

    # todo init S_A
    # sentence_head = sentence_text_logit(clip_state_dict)

    return model, video_head, classes, #sentence_head

class san_bike(nn.Module):
    def __init__(self, args):
        super(san_bike, self).__init__()
        # "resnet50" "ViT-B/16"
        # clip_model, preprocess = create_model(
        #     args.bike_clip_name, pretrained="openai")

        clip_model, clip_state_dict = clip.load(
            args.bike_clip_name,
            device='cpu', jit=False,
            internal_modeling=args.bike_clip_tm,
            T=args.data_num_segments,
            dropout=args.bike_clip_dropout,
            emb_dropout=args.bike_clip_dropout,
            pretrain=args.bike_clip_init,
            joint_st=args.bike_clip_joint_st
            )

        # print(clip_state_dict)
        # self.clip_visual_extractor = FeatureExtractor(
        #     clip_model.visual,
        #     last_layer_idx=-1,
        #     frozen_exclude=["positional_embedding"],
        # )

        self.video_head = video_header(
            args.sim_header,
            args.bike_clip_interaction,
            clip_state_dict)

        self.classes = text_prompt(args.classnames)

        for param in clip_model.parameters():
            param.requires_grad = False

        # pixel_mean, pixel_std = (clip_state_dict['mean'], clip_state_dict['std'])
        #
        # pixel_mean = [255.0 * x for x in pixel_mean]
        # pixel_std = [255.0 * x for x in pixel_std]
        # self.register_buffer(
        #     "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        # )
        # self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        #
        # self.size_divisibility = 32
        # self.asymetric_input = True
        # self.clip_resolution = 0.5
        self.clip_model = clip_model
        self.clip_visual_text_linear = nn.Linear(512, 768)

    def forward(self, samples, t):
        # images = [(x - self.pixel_mean) / self.pixel_std for x in samples.tensors]
        # # print(len(images), images[0].shape) #1 [5 3 192 224]
        #
        # clip_input = torch.stack(images, dim=0)  # 5 3 160 224
        # # clip_input = images[0]
        # # print(clip_input.shape, 'clip_input')
        # if self.asymetric_input:
        #     clip_input = F.interpolate(
        #         clip_input, scale_factor=self.clip_resolution, mode="bilinear", align_corners=False
        #     )

        # clip_image_features = self.clip_visual_extractor(clip_input)
        # print(len(clip_image_features)) #26
        # print(clip_image_features[0].size()) #5 768 2 3
        # 20 一共10层残差网络，【0-9】为bt c h w， 【0-9_cls_token】为1 bt c

        # print(clip_input.shape)#5 3 64 112
        # print(self.classes.shape) #93 77
        num_segments = t  # 视频帧数
        images_clip = samples.tensors.view((-1, num_segments, 3) + samples.tensors.size()[-2:])  # bt 3 h w
        b_o, t_o, c_o, h_o, w_o = images_clip.size()
        images_clip = images_clip.view(-1, c_o, h_o, w_o)  # 5 3 160 224
        img_size = 224
        clip_input = F.interpolate(images_clip, size=(img_size, img_size), mode='bilinear', align_corners=False)
        all_class = self.classes.to(clip_input.device)
        image_embedding, cls_embedding, text_embedding, logit_scale, all_img_feat = \
            self.clip_model(clip_input, all_class, return_token=True)

        # print(len(all_img_feat)) #12  l bt c

        # print(image_embedding.shape, 'img emb')  #5 512
        # print(image_embedding)
        # print(cls_embedding.shape, 'cls emb')  #89 512
        # print(cls_embedding)
        # print(text_embedding.shape, 'txt emb')  #89 77 512
        # print(text_embedding)
        # print(logit_scale) #100.
        img_embedding_sv = rearrange(image_embedding, '(b t) c -> b t c', t=t)
        images_fusion, S_V = self.video_head(img_embedding_sv, text_embedding, cls_embedding)
        # logits = logit_scale * S_V
        # print(S_V.shape, 'S_V') #1 89
        # print(S_V)
        # print(images_fusion.shape, "image fusion") #89 1 512
        # print(images_fusion)  #0.03, 应该乘logit_scale

        # clip_visual = self.clip_visual_linear(image_embedding.float())  # bt c 512->256
        # 视频与哪个类别最相关的索引
        max_index = torch.argmax(S_V)
        # print(images_fusion[max_index].shape)  # 1 512
        # 获取与最相关类的视频特征
        clip_visual_text = self.clip_visual_text_linear(images_fusion[max_index].float()) #1 768
        # clip_visual_add = clip_visual + clip_visual_text.repeat(t, 1)
        # print(clip_visual_add.shape, "clip add")  #5 256 bt c
        # print(clip_visual_add)

        clip_feat = []
        for i in range(4):
            layer_feat = (all_img_feat[i-4] + clip_visual_text.unsqueeze(0).repeat(all_img_feat[i-4].shape[0], t, 1))
            # print(all_img_feat[i-4].shape) #50 5 768
            # print(all_img_feat[i-4])
            # print(layer_feat.size()) #50 5 768

            clip_feat.append(layer_feat.permute(1,2,0)) #5 768 50

        return clip_feat




