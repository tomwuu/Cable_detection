# -*- coding: utf-8 -*-
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
设置全局变量
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 50000  # 200010  # 90100
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 1000
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.01  # 0.0005
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.95
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 8
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 8
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 210000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 2
# Set the image height
__C.TRAIN.IMG_HEIGHT = 288
# Set the image width
__C.TRAIN.IMG_WIDTH = 800
# Set GPU number
__C.TRAIN.GPU_NUM = 1    # 8
# Set CPU thread number
__C.TRAIN.CPU_NUM = 8    # 4

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.90
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
# Set the test batch size
__C.TEST.BATCH_SIZE = 8
# Set the test CPU thread number
__C.TEST.CPU_NUM = 8  # 4
