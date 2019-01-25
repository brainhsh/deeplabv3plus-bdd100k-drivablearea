backbone='resnet'
bs=2
cs=720
lr=0.00125
epoch=30
clear
echo "DEEPLABV3+ $backbone training start"
echo "batchsize: $bs"
echo "cropsize: ORIGINAL INPUT"
echo "lr: $lr"

CUDA_VISIBLE_DEVICES=0 python train.py --backbone $backbone --gpu-ids 0 --eval-interval 1 --dataset bdd --batch-size $bs --lr $lr --crop-size $cs --epochs $epoch
