backbone='resnet'

rm -rf './prd'
mkdir prd
clear

# mode='train' for usual eval
# mode='window' for sliding window eval

cs=720
echo "DEEPLABV3+ $backbone submission inferencing start"
echo "Cropsize: $cs"

CUDA_VISIBLE_DEVICES=0 python train.py --backbone $backbone --gpu-ids 0 --eval-interval 1 --dataset bdd --batch-size 1 --resume ./bdd/model_best.pth.tar --crop-size $cs --submit --inference
