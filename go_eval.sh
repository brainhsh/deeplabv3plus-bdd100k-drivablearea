backbone='resnet'

#rm -rf './prd'
#mkdir prd
clear

# mode='train' for usual eval
# mode='window' for sliding window eval

mode='window'
cs=720
echo "DEEPLABV3+ $backbone evaluation start"
echo "$mode .py"
echo "Cropsize: $cs"

CUDA_VISIBLE_DEVICES=0 python $mode.py --backbone $backbone --gpu-ids 0 --eval-interval 1 --dataset bdd --batch-size 1 --resume ./bdd/model_best.pth.tar --eval --crop-size $cs
