#!/usr/bin/env bash

################### Please Edit Here ############################
export DATA_ROOT=/home/yij/remote_home/datasets/MiPlo

SOURCE_DATASET=B    # B | H | J | M
TARGET_DATASET=H    # B | H | J | M
SUFFIX=''

## checkpoint dir (Note: adapt SOURCE_DATASET and TARGET_DATASET accordingly!)
LOG_DIR=MiPlo_B2H_hard_69.6
# LOG_DIR=MiPlo_B2H_soft_67.4
# LOG_DIR=MiPlo_J2M_hard_66.9
# LOG_DIR=MiPlo_J2M_soft_62.5
# LOG_DIR=MiPlo_B2J_soft_42.2
#################################################################

python mv-match.py data/miplo --log logs/mv-match/${LOG_DIR} \
 -d MiPlo -s ${SOURCE_DATASET} -t ${TARGET_DATASET} \
 -a resnet50 --epochs 20 --seed 3407 \
 --per-class-eval \
 --bottleneck-dim 256 \
 --train-resizing res. --val-resizing res. --resize-size 1344 \
 --batch-size 4 --workers 2 \
 --lr 0.003 --lr-gamma 0.0004 --lr-decay 0.75 --weight-decay 0.001 \
 --trade-off 1 --unlabeled-batch-size 4 --threshold 0.8 \
 --sample mutual \
 --iters-per-epoch 1200 \
 --phase test
#  --use_soft_label
#  --print-freq 1