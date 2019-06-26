"""
This is to draw output cable(from prob maps) lines using linear regression, the number of lines is up to 2
"""
import cv2
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--probmap_path', type=str, help='The probmap path ')
    parser.add_argument('--filename', type=str)

    return parser.parse_args()

def get_middle_max(list_):
    l = []
    mid_value = np.mean(list_)
    for i in range(len(list_)):
        if list_[i] >= mid_value:
            l.append(i)
    return l[int(len(l)/2)]

def judge(dataset_info_file):
    with open(dataset_info_file, 'r') as file:
        for _info in file:
            info_tmp = _info.strip(' ').split()
    count = int(info_tmp[0]) + int(info_tmp[1]) + int(info_tmp[2]) + int(info_tmp[3])
    return count

if __name__ == '__main__':
    # init args
    args = init_args()
    lr = LinearRegression()
    x1_ = []
    y1_ = []
    points1 = []

    x2_ = []
    y2_ = []
    points2 = []

    dataset_info_file = args.probmap_path   #/Users/wutong/Pictures/grass/pictures/cable_pic/uploads/1_22_1.exist.txt
    dataset_info_file = dataset_info_file[:-10] + '.exist.txt'####################### 可以改进
    print('dataset_info_file:###########', dataset_info_file)
    print('judge(dataset_info_file): #############', judge(dataset_info_file))
    if judge(dataset_info_file) == 1:
        img1 = cv2.imread(args.image_path, cv2.IMREAD_COLOR)  # 原图
        img2 = cv2.imread(args.probmap_path, 0)  # GT
        img2_ = cv2.resize(img2, (640, 360))
        for i in range(18):
            y1_.append(20*i)
            x1_.append(get_middle_max(img2_[20*i, :]))
            points1.append([x1_[i], y1_[i]])

        print(points1)
        rows, cols = img1.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(np.array(points1, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        print([vx, vy, x, y])
        righty = int((y * vx / vy) + x)
        lefty= int((x - (rows - y) * vx / vy))
        res = cv2.line(img1, (lefty, 0), (righty, rows - 1), (255, 0, 140), 2)
        print("res.shape:   ", res.shape)
        cv2.imwrite("/Users/wutong/Desktop/uploads/saved/" + args.filename, res)

    elif judge(dataset_info_file) == 2:
        img1 = cv2.imread(args.image_path, cv2.IMREAD_COLOR)  # 原图
        img2 = cv2.imread(args.probmap_path, 0)  # GT /Users/wutong/Pictures/grass/pictures/cable_pic/uploads/1_22_1_1_avg.png
        img2_ = cv2.resize(img2, (640, 360))

        path_ = args.probmap_path
        path_ = path_[:-9]+"2"+path_[-8:]
        img3 = cv2.imread(args.probmap_path, 0)  # GT  /Users/wutong/Pictures/grass/pictures/cable_pic/uploads/1_22_1_2_avg.png
        img3_ = cv2.resize(img2, (640, 360))

        for i in range(18):
            y1_.append(20*i)
            x1_.append(get_middle_max(img2_[20*i, :320]))
            points1.append([x1_[i], y1_[i]])

            y2_.append(20*i)
            x2_.append(get_middle_max(img3_[20*i, 320:]))
            points2.append([320 + x2_[i], y2_[i]])


        rows, cols = img1.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(np.array(points1, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        righty = int((y * vx / vy) + x)
        lefty= int((x - (rows - y) * vx / vy))
        res = cv2.line(img1, (lefty, 0), (righty, rows - 1), (255, 0, 140), 2)
        #cv2.imwrite("/Users/wutong/Desktop/uploads/saved/" + '66666' + args.filename, res)

        rows, cols = res.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(np.array(points2, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        righty = int((y * vx / vy) + x)
        lefty= int((x - (rows - y) * vx / vy))
        res_ = cv2.line(res, (lefty, 0), (righty, rows - 1), (255, 0, 140), 2)

        cv2.imwrite("/Users/wutong/Desktop/uploads/saved/" + args.filename, res_)







#cv2.imshow('lane', res)
#cv2.waitKey()
