#!/usr/bin/env bash

GPUS='0,1,2,3'
GPUS_PER_NODE=4
CPUS_PER_TASK=6
PORT=29500
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="video_swin_t_p4w7"
BACKBONE_PRETRAINED="./checkpoints/backbones/swin_tiny_patch244_window877_kinetics400_1k.pth"
OUTPUT_DIR="./checkpoints/results/SgMg_${BACKBONE}_finetune/eval"
CHECKPOINT="./checkpoints/results/SgMg_video_swin_t_p4w7_finetune/checkpoint.pth"
python inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
  --eval \
  --ngpu=${GPUS_PER_NODE} \
  --output_dir=${OUTPUT_DIR} \
  --resume=${CHECKPOINT} \
  --backbone=${BACKBONE} \
  --backbone_pretrained=${BACKBONE_PRETRAINED} \
  --amp \
