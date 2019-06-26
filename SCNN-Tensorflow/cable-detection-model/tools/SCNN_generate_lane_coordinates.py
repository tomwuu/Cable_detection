import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os

def getLane(probmap):
    thre = 0.3
    coordinate = np.zeros((1,18))
    for i in range(18):
        lineId = int(287-i*20/590*288)
        line = probmap[lineId, :]
        try:
            value = max(line[1])
        except Exception as e:
            value = 0
        if float(value)/255 > thre:
            id = line[1].index(value)
            coordinate[i] = id
    try:
        if sum(coordinate[i] > 0 for i in range(18)):
            return coordinate
        else:
            return np.zeros((1,18))
    except Exception as _:
        return np.zeros((1,18))


if __name__ == '__main__':
    test = 'test'
    h = 288
    w = 800
    #exp = 'r101_SCNN_w8_6_ft_all'
    if test == 'val':
        exp = 'r50_6_ft'
        List = open('list/val.txt', 'r')
        pred = open('pred_val_r50_6_03_03.json', 'w')
    else:
        exp = os.path.join('/Users/wutong/Desktop/prob_map/pictures_before_finetune', 'vgg_SCNN_DULR_w9')
        List = open('/Users/wutong/Desktop/prob_map/list_test_copy1.txt', 'r')  #图片地址
        pred = open('/Users/wutong/Desktop/prob_map/pred_test.txt', 'w') #要生成的pred json

    thr = 0.2  #threshold
    Lines = List.readlines()  #test

    for n in range(0, 819):
        line = Lines[n]
        img_path = line.split()[0]
        exist_path = exp + '/' + img_path[-26:-3] + 'exist.txt'
        exist = open(exist_path, 'r').readline().split()
        time = 165
        #time = int(float(exist[6])*1000)
        #if time > 200:
        #    time = 165
        #    print('time larger than 0.2s!')
        exist = [int(e) for e in exist[:4]]
        lanes = []
        for i in range(4):
            if exist[i] == 1:
                prob_path = exp + '/' + img_path[-26:-4] + '_' +str(i+1) + '_avg.png'   #得到第一步生成的prob map img_path[-26:-4]
                probs = cv2.imread(prob_path, 0)
                print(probs)
                Coordinate = getLane(probs)  # 得到prob map车道点的坐标
                for coordinate in Coordinate:
                    lanes.append(coordinate)
            for j in lanes:
                print(str(j))
                pred.write(str(j))
                pred.write('\n')

        #lanes = rmShort(lanes, 20)
        #lanes = connect(lanes)
        #lanes = rmShort(lanes, 70)
        #lanes = cutMax(lanes)
        #show(img_path[1:], lanes)
