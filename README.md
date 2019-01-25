# Deeplabv3+ BDD100k/drivable_area implementation

### Introduction

This repository is from https://github.com/jfzhang95/pytorch-deeplab-xception.
For BDD100k/drivable_area semantic segmentation, I added 
1. bdd100k drivable area dataloader.
2. prediction visualization for both color(visual result) and id(greyscale png file for submission).

For more detail, please visit the repository above.

### How to train
```Shell
	bash go_train.sh
```


### How to evaluate
```Shell
	bash go_eval.sh
```

### How to inference
For visual result,
```Shell
	bash go_inference.sh
```
The code results out only 100 example, but you can easily change the code. 

For submission, 
```Shell
	bash go_submit.sh
```

### Experiment & result

0. Architecture
	I used Resnet101 as backbone. Used ImageNet pretrianed model, and finetuned. I replaced all Batch Normalization layers with Group Normalization layer which groups 16 channels. 

1. Hyperparameter & environment
	I used 720 or noncrop for input. Using Group Normalization, I could avoid hurts from reducing batchsize due to increasing cropsize. If you want to specify cropsize, please check the code 'dataloaders/datasets/bdd100k.py'. Default is non-cropping input.
	Learning rate: 0.01 / 16 * batchsize
	weight_decay: 0.0005
	momentum: 0.9
	max epoch: 30
	Single 12GB GPU

| Backbone  | train/eval os  |mIoU in val | mIoU in test |
| :-------- | :------------: |:---------: | :-----------:|
| ResNet101 | 16/16          | 82.55%     | 82.61%       |

![Results](prd/result.png)
