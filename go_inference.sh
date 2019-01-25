backbone='resnet'
cs=720

rm -rf './prd'
mkdir prd
clear
echo "DEEPLABV3+ $backbone training start"
echo "Cropsize: $cs"

CUDA_VISIBLE_DEVICES=0 python train.py --backbone $backbone --gpu-ids 0 --eval-interval 1 --dataset bdd --batch-size 1 --resume ./bdd/model_best.pth.tar --inference --crop-size $cs
