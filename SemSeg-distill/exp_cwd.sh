#!/bin/bash

CITYSCAPES=/mnt/lustre/share/open-mmlab/datasets/cityscapes
NORM=$1
DIVERGENCE=$2
WEIGHT=$3
TEMPERATURE=$4
KD=$5
ADV=$6
FEAT=$7
VERSION=$8

NAME=${KD}_${ADV}_${FEAT}_${NORM}_${DIVERGENCE}_${WEIGHT}_${TEMPERATURE}_${VERSION}
python   train.py --kd $KD --adv ${ADV} --cwd True --cwd-feat ${FEAT} --temperature $TEMPERATURE --norm-type $NORM  --divergence  $DIVERGENCE  --lambda-cwd $WEIGHT --data-dir $CITYSCAPES  --save-name ${NAME} --gpu 0
#python val.py --data-dir $CITYSCAPES --restore-from ckpt/${NAME}/city_39999_G.pth --gpu 0 > ckpt/${NAME}/test.out

