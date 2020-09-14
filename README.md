Attentive Few-Shot Object Detection Network (AttFDNet)
=====================================

This code implements the Attentive Few-Shot Object Detection Network (AttFDNet)

Disclaimer
------------------
We adopt the official implementation of the [`RFBNet`](https://github.com/ruinmessi/RFBNet) as a baseline model for few-shot object detection. We also use the [`Boolean Map Saliency algorithm`](https://github.com/fzliu/saliency-bms) to extract human saliency result of given an image. Please refer to these links for further README information.

Requirements
------------------
1. Requirements for Pytorch. We use Pytorch 0.4.1 in our experiments.
2. Python 3.6+

3. I also provide the conda environment [RFBNet.yaml](https://github.com/chenxy99/AttFDNet/blob/master/RFBNet.yaml), you can directly run

```bash
$ conda env create -f RFBNet.yaml
```
to create the same environment where I successfully run my code.

Dataset
------------------

We provide the spitted VOC dataset in the [Link](https://drive.google.com/file/d/1fmI7CjDFqFTOM4UhBARvLpGCbfqlF_zm/view?usp=sharing).

1. You need to manually change the home directory in [code](https://github.com/chenxy99/AttFDNet/blob/master/data/config.py#L6).
2. You need to manually change the classes of each splits (e.g., split 1, split 2, split 3) in [code](https://github.com/chenxy99/AttFDNet/blob/master/data/voc0712.py#L86) corresponding to the given task as well as the training stages (e.g., base stage or novel stage).

Start training
------------------

- Base stage training

As an example, for base stage of training split 1, you can directly use the follow command to train the model

```bash
$ python train_RFB.py --split split1 --save_folder ./weights/task1/source_300_0712_320embedding_20200227/
```

It needs `vgg16_reducedfc.pth` and we include it in the pretrained models [Link](https://drive.google.com/file/d/1teUfobkg3SpHRmL4GFeC5HuiGRqfKqMm/view?usp=sharing).

- Novel stage training

As an example, for novel stage of training split 1, you can directly use the follow command to train the model

```bash
$ python train_RFB_target.py --split split1 --shots 2 --save_folder ./weights/task1/novel_2shot_05kd_seed0_2dist_div8_new/ --resume_net ./weights/task1/source_300_0712_320embedding_20200227/Final_RFB_vgg_VOC.pth
```

Evaluation
------------------

- Base stage evaluation

As an example, for evaluation of the base classes of split 1, you can directly use the follow command to evaluate the model

```bash
$ python test_RFB.py -split split1 --trained_model ./weights/task1/source_300_0712_320embedding_20200227/Final_RFB_vgg_VOC.pth
```

- Novel stage evaluation

As an example, for evaluation of the novel classes of split 1, you can directly use the follow command to evaluate the model

```bash
$ python test_RFB_target.py --trained_model ./weights/task1/novel_2shot_05kd_seed0_2dist_div8_new/Final_RFB_vgg_VOC.pth
```

`To evaluate the different dataset, you need to remove the file ''annots.pkl'' in the document './data/VOCdevkit/annotations_cache/'' where you put your dataset.`

Pretrained models
------------------

We also provide some of the pretrained model from the [Link](https://drive.google.com/file/d/1teUfobkg3SpHRmL4GFeC5HuiGRqfKqMm/view?usp=sharing).

It includes three models

- Base model for split 1 named `./weights/task1/source_300_0712_320embedding_20200227`

- Novel model for split 1 for 2 shots scenario named `./weights/task1/novel_2shot_05kd_seed0_2dist_div8_new`
- Novel model for split 1 for 3 shots scenario named `./weights/task1/novel_3shot_05kd_seed0_2dist_div8_new`

You can run the follow command to evaluate the model for `./weights/task1/novel_2shot_05kd_seed0_2dist_div8_new`

```bash
$ python test_RFB_target.py --trained_model ./weights/task1/novel_2shot_05kd_seed0_2dist_div8_new/Final_RFB_vgg_VOC.pth
```