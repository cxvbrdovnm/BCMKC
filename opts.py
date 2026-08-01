import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('SgMg training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)  # MTTR 1e-4 1e-5 1e-5 wd 1e-4
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=[6, 8], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # New Parameters
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--exp_name', default='main', type=str)
    parser.add_argument('--current_epoch', default=0, type=int)

    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.") 

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true') # NOTE: must be false

    # * Backbone
    # ["resnet50", "resnet101", "swin_t_p4w7", "swin_s_p4w7", "swin_b_p4w7", "swin_l_p4w7"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--text_backbone', default='Roberta', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str, 
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--use_checkpoint', action='store_true', help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true', # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--output_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, 
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries") 
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--freeze_text_encoder', action='store_true') # default: False

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int, 
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=2, type=int,
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=16, type=int,
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=2, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_boundary', default=2, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=5, type=float,
                        help="mask coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--boundary_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--conditioned_iou_loss_coef', default=5, type=float)



    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name') 
    parser.add_argument('--coco_path', type=str, default='../datasets/coco')
    parser.add_argument('--ytvos_path', type=str, default="../ReferFormer/data/ref-youtube-vos")
    parser.add_argument('--davis_path', type=str, default='../ReferFormer/data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='../ReferFormer/data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='../ReferFormer/data/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float) # binary threshold for mask
    parser.add_argument('--ngpu', default=8, type=int, help='gpu number when inference for ref-ytvos and refer_davis')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    #rpe
    parser.add_argument('--rpe_enc_layers', default=1, type=int)
    parser.add_argument('--rpe_dec_layers', default=1, type=int)
    parser.add_argument('--enc_rpe2d', default='rpe-2.0-product-ctx-1-kv')  # k qkv kv qk
    parser.add_argument('--time_gap', default=1, type=int)

    #maple
    parser.add_argument('--maple_prec', default="amp")
    parser.add_argument('--clip_backbone', default="ViT-B/16")
    parser.add_argument('--maple_train_n_ctx', default=2, type=int, help='number of maple_length')
    parser.add_argument('--maple_train_ctx_init', default='a photo of a', type=str, help="初始化上下文向量的初始文本")
    parser.add_argument('--maple_train_prompt_depth', default=3, type=int, help='Default is 1, 1~12, which is compound shallow prompting')
    class_names = [
        "person", "cat", "train", "hedgehog", "squirrel", "table",
        "ape", "snake", "owl", "eagle", "rope", "camera",
        "parrot", "zebra", "plant", "snail", "chameleon", "watch",
        "giant_panda", "giraffe", "airplane", "toilet", "box", "stuffed_toy",
        "sedan", "bear", "bus", "camel", "tissue", "guitar",
        "lizard", "fox", "shark", "frisbee", "kangaroo", "microphone",
        "duck", "leopard", "tiger", "whale", "cloth", "cup",
        "dog", "elephant", "surfboard", "knife", "bottle", "shovel",
        "skateboard", "horse", "earless_seal", "tennis_racket", "small_panda", "flag",
        "monkey", "others", "frog", "crocodile", "spider", "mirror",
        "sheep", "deer", "mouse", "umbrella", "ball", "ring",
        "fish", "motorbike", "boat", "paddle", "jellyfish", "necklace",
        "rabbit", "turtle", "snowboard", "raccoon", "eyeglasses", "ant",
        "hat", "bird", "penguin", "parachute", "backpack",
        "cow", "truck", "lion", "bucket", "butterfly",
        "hand", "dolphin", "sign", "bike", "handbag"
    ]
    parser.add_argument('--classnames', default=class_names)


    #soc
    parser.add_argument('--set_cost_con', default=0, help='soft tokens coefficient in the matching cost')
    parser.add_argument('--con_loss_coef', default=1)
    parser.add_argument('--window_size', default=0, help='the windows attention of enco layer')
    parser.add_argument('--num_frame_queries', default=5, help='the initial query of frame')
    parser.add_argument('--voc_dec_layers', default=3)

    #bike
    #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14-336px']
    parser.add_argument('--bike_clip_name', default='ViT-B/32')
    parser.add_argument('--bike_clip_tm', default=False)
    parser.add_argument('--data_num_segments', default=8)
    parser.add_argument('--bike_clip_dropout', default=0.0)
    parser.add_argument('--bike_clip_init', default=True)
    parser.add_argument('--bike_clip_joint_st', default=False)
    parser.add_argument('--sim_header', default="Transf")
    parser.add_argument('--bike_clip_interaction', default="VCS")
    #train video: True False, train text: False, True
    parser.add_argument('--bike_clip_fix_text', default=True)
    parser.add_argument('--bike_clip_fix_video', default=False)

    parser.add_argument('--model_name', default='')

    return parser


