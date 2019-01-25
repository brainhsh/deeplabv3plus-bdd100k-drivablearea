clear

logdir='./run/bdd/deeplab-resnet/'

echo "tensorboard summary @$logdir"

tensorboard --logdir $logdir
