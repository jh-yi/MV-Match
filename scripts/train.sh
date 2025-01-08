#!/usr/bin/env bash

################### Please Edit Here ############################
export DATA_ROOT=/home/yij/remote_home/datasets/MiPlo

SOURCE_DATASET=B    # B | H | J | M
TARGET_DATASET=H    # B | H | J | M
SUFFIX='hard'
LOG_DIR=MiPlo_${SOURCE_DATASET}_${TARGET_DATASET}_${SUFFIX}
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
#  --use_soft_label
 # --print-freq 1
