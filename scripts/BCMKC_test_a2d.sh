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

BACKBONE="video_swin_b_p4w7"
BACKBONE_PRETRAINED="./checkpoints/backbones/swin_base_patch244_window877_kinetics600_22k.pth"
MODEL_NAME="BCMKC"
OUTPUT_DIR="./checkpoints/final_results/${MODEL_NAME}_${BACKBONE}_finetune_a2d"

EXP_NAME="${MODEL_NAME}_${BACKBONE}_finetune_a2d"
PRETRAINED_WEIGHTS="./checkpoints/results/sgmg_videoswinb_a2d.pth"

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env main.py \
  --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 4 \
  --resume ./checkpoints/results/SgMg_Sv_Sa_205_6_video_swin_b_p4w7_finetune_a2d/checkpoint.pth \
  --backbone video_swin_b_p4w7  --eval --model_name BCMKC