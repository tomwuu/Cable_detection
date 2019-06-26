# Cable_detection
Using a fake self-made dataset to detect cable in real world

# Content

* [Installation](#Installation)
* [Datasets](#Datasets)
  * [TuSimple](#TuSimple)
  * [CULane](#CULane)
  * [BDD100K](#BDD100K)
* [SCNN-Tensorflow](#SCNN-Tensorflow)
  * [Train](#Train)

# Installation

1. Install necessary packages:
```
    conda create -n tensorflow_gpu pip python=3.6
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.13.0
    pip3 install -r SCNN-Tensorflow/lane-detection-model/requirements.txt
```

2. Download VGG-16:

Download the vgg.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in SCNN-Tensorflow/lane-detection-model/data.

3. Pre-trained model for testing:

Download the pre-trained model [here](https://drive.google.com/open?id=1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5).

# Datasets

## TuSimple

The ground-truth labels of TuSimple testing set is now available at [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). The annotated training (#frame = 3268) and validation labels (#frame = 358) can be found [here](https://github.com/cardwing/Codes-for-Lane-Detection/issues/11), please use them (list-name.txt) to replace the train_gt.txt and val_gt.txt in [train_lanenet.py](SCNN-Tensorflow/cable-detection-model/tools/train_lanenet.py). Moreover, you need to resize the image to 256 x 512 instead of 288 x 800 in TuSimple. Remember to change the maximum index of rows and columns, and detailed explanations can be seen [here](https://github.com/cardwing/Codes-for-Lane-Detection/issues/18). Please evaluate your pred.json using the labels and [this script](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py). Besides, to generate pred.json, you can refer to [this issue](https://github.com/cardwing/Codes-for-Lane-Detection/issues/4).

## CULane

The whole dataset is available at [CULane](https://xingangpan.github.io/projects/CULane.html).

## BDD100K

The whole dataset is available at [BDD100K](http://bdd-data.berkeley.edu/).

## My fake data for cable detection
The whole dataset is available at [cable_fake]().
You can make your own data using [POV-Ray](http://www.povray.org/). It can simulate lots of textures, use these textures 
and combine them with your background pictures. 


# SCNN-Tensorflow

## Train
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/

Note that path/to/CULane-dataset/ should contain files like [train_gt.txt](SCNN-Tensorflow/cable-detection-model/demo_file/train_gt.txt) and [val_gt.txt](SCNN-Tensorflow/cable-detection-model/demo_file/train_gt.txt).

## you should change some dir parameter
