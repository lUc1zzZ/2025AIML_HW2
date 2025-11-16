#!/bin/bash

OUTDIR="outputs"

echo "===== 1. Baseline 实验 ====="
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --k-folds 1 --augment --output-dir $OUTDIR --run-name baseline_bn_dropout0.3

echo "===== 2. Dropout 对比 ====="
for drop in 0.0 0.1 0.3 0.5
do
    python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout $drop \
        --norm batch --augment --output-dir $OUTDIR --run-name drop_${drop}
done

echo "===== 3. Normalization 对比 ====="
for norm in batch group instance none
do
    python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 \
        --norm $norm --augment --output-dir $OUTDIR --run-name norm_${norm}
done

echo "===== 4. Model Width 扩展 ====="
for width in 48 64
do
    python train.py --epochs 40 --batch-size 128 --lr 0.01 --dropout 0.3 \
        --norm batch --augment --base-channels $width --output-dir $OUTDIR \
        --run-name wide${width}
done

echo "===== 5. Weight Decay 实验 ====="
for wd in 0.0005 0.001
do
    python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 \
        --norm batch --weight-decay $wd --augment --output-dir $OUTDIR \
        --run-name wd_${wd}
done

echo "===== 6. Label Smoothing ====="
python train.py --epochs 30 --batch-size 128 --lr 0.01 --dropout 0.3 \
    --norm batch --label-smoothing 0.1 --output-dir $OUTDIR --run-name label_smooth_0.1

echo "======== 所有实验完成 ========"
