#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=2 python mv-match.py data/dnd_cka_miplov2 \
 --log logs/fixmatch/MiPlo_J2M_lr3e3_wd1e3_sms+ts+tms_mutualp5_s3407 \
 -d MiPlo -s J -t M -a resnet50 --epochs 20 --seed 3407 \
 --per-class-eval \
 --bottleneck-dim 256 \
 --train-resizing res. --val-resizing res. --resize-size 1344 \
 --batch-size 4 --workers 2 \
 --lr 0.003 --lr-gamma 0.0004 --lr-decay 0.75 --weight-decay 0.001 \
 --trade-off 1 --unlabeled-batch-size 4 --threshold 0.8 \
 --sample mutual \
 --iters-per-epoch 1200 \
#  --phase test
#  --print-freq 1
#  --use_soft_label