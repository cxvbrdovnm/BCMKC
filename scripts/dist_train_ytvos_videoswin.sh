#!/usr/bin/env bash
#set -x
#cd ..
#
#GPUS='0,1'
#PORT=25500
#GPUS_PER_NODE=2
#CPUS_PER_TASK=6
#export CUDA_VISIBLE_DEVICES=${GPUS}
#echo "using gpus ${GPUS}, master port ${PORT}."
#now=$(date +"%T")
#echo "Current time : $now"
#echo "Current path : $PWD"
#
#BACKBONE="video_swin_t_p4w7"
#BACKBONE_PRETRAINED="./checkpoints/backbones/swin_tiny_patch244_window877_kinetics400_1k.pth"
#OUTPUT_DIR1="./checkpoints/results/SgMg_${BACKBONE}_pretrain"
#EXP_NAME1="SgMg_${BACKBONE}_pretrain"
#CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main_pretrain.py \
#  --dataset_file all \
#  --with_box_refine --binary \
#  --output_dir=${OUTPUT_DIR1} \
#  --exp_name=${EXP_NAME1} \
#  --backbone=${BACKBONE} \
#  --backbone_pretrained=${BACKBONE_PRETRAINED} \
#  --batch_size 2 \
#  --num_frames 1 \
#  --epochs 11 --lr_drop 8 10 \
#
#
#OUTPUT_DIR2="./checkpoints/results/SgMg_${BACKBONE}_finetune"
#EXP_NAME2="SgMg_${BACKBONE}_finetune"
#CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
#  --with_box_refine --binary --freeze_text_encoder \
#  --output_dir=${OUTPUT_DIR2} \
#  --exp_name=${EXP_NAME2} \
#  --backbone=${BACKBONE} \
#  --backbone_pretrained=${BACKBONE_PRETRAINED} \
#  --epochs 6 --lr_drop 3 5 \
#  --dataset_file ytvos \
#  --pretrained_weights ${OUTPUT_DIR1}"/checkpoint0010.pth" \


set -x

GPUS='0,1,2,3'
PORT=25501
GPUS_PER_NODE=4
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
export OMP_NUM_THREADS=${CPUS_PER_TASK}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="video_swin_t_p4w7"
BACKBONE_PRETRAINED="./checkpoints/backbones/swin_tiny_patch244_window877_kinetics400_1k.pth"
#BACKBONE_PRETRAINED="../ReferFormer/pretrained_weights/video_swin_base_pretrained.pth"
PRETRAINED_WEIGHT="./checkpoints/results/sgmg_videoswint_ytvos.pth"

OUTPUT_DIR2="./checkpoints/results/SgMg_${BACKBONE}_finetune"
EXP_NAME2="SgMg_${BACKBONE}_finetune"
python3 -m torch.distributed.launch --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} --use_env main.py \
  --with_box_refine --binary --freeze_text_encoder \
  --output_dir=${OUTPUT_DIR2} \
  --exp_name=${EXP_NAME2} \
  --backbone=${BACKBONE} \
  --backbone_pretrained=${BACKBONE_PRETRAINED} \
  --epochs 6 --lr_drop 3 5 \
  --dataset_file ytvos \
  --pretrained_weights ${PRETRAINED_WEIGHT} \
  --epochs 9 --lr_drop 6 8 \
