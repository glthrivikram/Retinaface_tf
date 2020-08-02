# Retinaface_tf
Single-stage Dense Face Localisation, Implemented ResNet50, MobileNetV2 trained on single GPU using Tensorflow 2.0+.

Based on [this](https://arxiv.org/abs/1905.00641) research paper.

Original implimentation of the research paper [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace) .

# WiderFace Validation Set Performance 
| Model Name          | Easy   | Medium | Hard   |
|---------------------|--------|--------|--------|
| Tensorflow2 MobileNetV2 | 93.23% | 91.74% | 80.01% |
| Tensorflow2 ResNet50 | 94.21% | 93.25% | 83.55% |

## Environment

### Pip

```bash
pip install -r requirements.txt
```

## Data
Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/index.html) dataset images from the download links bellow.

| Dataset Name | Link |
|:------------:|:----------:|
| WIDER Face Training Images | [Google Drive](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing) |
| WIDER Face Validation Images | [Google Drive](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing) |

### Annotations
Download the [Retinaface official annotations](https://github.com/deepinsight/insightface/tree/master/RetinaFace#Data) (face bounding boxes & five facial landmarks) from the download links bellow.

| Dataset Name | Link |
|:------------:|:----------:|
| Retinaface Annotations | [Google Drive](https://drive.google.com/file/d/1vgCABX1JI3NGBzsHxwBXlmRjaLV3NIsG/view?usp=sharing) / [Dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0) |


Extract downloaded files into `./data/widerface/`. The directory structure should be like bellow.
```
./data/widerface/
    train/
        images/
        label.txt
    val/
        images/
        label.txt
```

Convert the training images and annotations to tfrecord file with the the script bellow.
```bash

python data/convert_train_tfrecord.py --output_path="./data/widerface_train_bin.tfrecord" --is_binary=True
```
 You can run `python ./dataset_checker.py` to check if the dataloader work.

### Config
You can modify your own dataset path or other settings of model in [./configs/*.yaml] for training and testing, which like below.


```python
# general setting
batch_size: 8
input_size: 640
backbone_type: 'ResNet50'  # 'ResNet50', 'MobileNetV2'
sub_name: 'retinaface_res50'

# training dataset
dataset_path: './data/widerface_train_bin.tfrecord'
dataset_len: 12880  # number of training samples
using_bin: True
using_flip: True
using_distort: True

# testing dataset
testing_dataset_path: './data/widerface/val'

# network
out_channel: 256

# anchor setting
min_sizes: [[16, 32], [64, 128], [256, 512]]
steps: [8, 16, 32]
match_thresh: 0.45
ignore_thresh: 0.3
variances: [0.1, 0.2]
clip: False

# training setting
epoch: 100
init_lr: !!float 1e-2
lr_decay_epoch: [50, 68]
lr_rate: 0.1
warmup_epoch: 5
min_lr: !!float 1e-3

weights_decay: !!float 5e-4
momentum: 0.9

pretrain: True

save_steps: 2000
```

### Training


Train the Retinaface model by yourself, or dowload it from BenchmarkModels.
```bash
# train ResNet50 backbone model
python train.py --cfg_path="./configs/retinaface_res50.yaml" --gpu=0
# or train MobileNetV2 backbone model
python train.py --cfg_path="./configs/retinaface_mbv2.yaml" --gpu=0
```
- You can visualize the learning rate scheduling by running "`python ./modules/lr_scheduler.py`".

### Testing on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/index.html) Validation Set

Produce txt results and visualizations from model.
```bash
# Test ResNet50 backbone model
python test_widerface.py --cfg_path="./configs/retinaface_res50.yaml" --gpu=0
# or
# Test ResNet50 backbone model
python test_widerface.py --cfg_path="./configs/retinaface_mbv2.yaml" --gpu=0
```

Evaluate txt results. (Code from [Here](https://github.com/wondervictor/WiderFace-Evaluation))
```bash
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

### Detect on Input Image

```bash
python test.py --cfg_path="./configs/retinaface_res50.yaml" --img_path="./PATHtoIMG.jpg" --down_scale_factor=1.0
# or
python test.py --cfg_path="./configs/retinaface_mbv2.yaml" --img_path="./PATHtoIMG.jpg" --down_scale_factor=1.0
```

### Demo on Webcam

Demo face detection on your webcam.
```bash
python test.py --cfg_path="./configs/retinaface_res50.yaml" --webcam=True --down_scale_factor=1.0
# or
python test.py --cfg_path="./configs/retinaface_mbv2.yaml" --webcam=True --down_scale_factor=1.0
```


## References


- https://github.com/deepinsight/insightface/tree/master/RetinaFace (Official)
    - Face Analysis Project on MXNet http://insightface.ai
- https://github.com/biubug6/Pytorch_Retinaface
    - Retinaface get 80.99% in widerface hard val using mobilenet0.25.
- https://github.com/wondervictor/WiderFace-Evaluation
    - Python Evaluation Code for Wider Face Dataset
- https://github.com/zzh8829/yolov3-tf2
    - YoloV3 Implemented in TensorFlow 2.0