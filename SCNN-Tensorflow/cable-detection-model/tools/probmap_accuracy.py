"""
If you get output probmap and have a GT probmap, you can get the accuracy (confidence) of your output 
"""

import cv2
import argparse
import numpy as np

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--probmap_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--filename', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    global acc
    thresh = 50
    # init args
    args = init_args()
    GT_path = '/Users/wutong/Pictures/grass/segGT/' + args.filename[-8:-6].replace('_', '') + '.jpg'
    img1 = cv2.imread(GT_path, 0)#GT
    img2 = cv2.imread(args.probmap_path, 0)
    img2_ = cv2.resize(img2, (640, 360))#prob map
    img2_[np.where(img2_<thresh)] = 0
    count = 0
    count_a = 0
    for i in range(360):
        for j in range(640):
            if img1[i, j] > 0:
                count += 1
                if img2_[i, j] > 0:
                    count_a += 1

    acc = count_a/float(count)
    with open('/Users/wutong/Desktop/uploads/acc.txt', 'w') as f:
        f.writelines(str(acc))


