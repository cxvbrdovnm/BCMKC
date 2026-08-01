#!/usr/bin/env bash
set -x
#cd ..

GPUS='0,1'
PORT=25501
GPUS_PER_NODE=2
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
export OMP_NUM_THREADS=${CPUS_PER_TASK}
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="video_swin_t_p4w7"
BACKBONE_PRETRAINED="./checkpoints/backbones/swin_tiny_patch244_window877_kinetics400_1k.pth"
MODEL_NAME="SgMg"
OUTPUT_DIR="./checkpoints/results/${MODEL_NAME}_${BACKBONE}_finetune_a2d"

EXP_NAME="${MODEL_NAME}_${BACKBONE}_finetune_a2d"
PRETRAINED_WEIGHTS="./checkpoints/results/sgmg_videoswint_a2d.pth"

##CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
#python3 -m torch.distributed.launch --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} --use_env main.py \
#  --with_box_refine --binary --freeze_text_encoder \
#  --exp_name=${EXP_NAME} \
#  --output_dir=${OUTPUT_DIR} \
#  --backbone=${BACKBONE} \
#  --backbone_pretrained=${BACKBONE_PRETRAINED} \
#  --dataset_file a2d \
#  --batch_size 1 \
#  --epochs 6 --lr_drop 3 5 \
#  --pretrained_weights=${PRETRAINED_WEIGHTS} \
#  --max_size 448 \
##  --epochs 12 --lr_drop 6 8 \
##  --max_size 200

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env main.py \
  --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 2 \
  --resume ./checkpoints/results/sgmg_videoswint_a2d.pth \
  --backbone video_swin_t_p4w7  --eval

#python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env main.py \
#  --dataset_file jhmdb --with_box_refine --freeze_text_encoder --batch_size 2 \
#  --resume ./checkpoints/results/SgMg_video_swin_t_p4w7_finetune_a2d/checkpoint.pth \
#  --backbone video_swin_t_p4w7  --eval
