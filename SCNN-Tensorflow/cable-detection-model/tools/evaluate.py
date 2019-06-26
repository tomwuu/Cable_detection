"""
Test FINE-TUNE result of tusimple data
step three
test tusimple data pred & gt
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20  # 设定20的宽度
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)   #做了x,y的线性回归
            k = LaneEval.lr.coef_[0]
            print('k: ', k)
            theta = np.arctan(k)
        else:
            theta = 0
        return theta   # 返回夹角

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        print(pred)
        gt = np.array([g if g >= 0 else -100 for g in gt])
        print(gt)
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

# a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
    @staticmethod  # pred coordinates, GT coordinates, y_coordinates, runtime
    def bench(pred, gt, y_samples, running_time):
        for p in pred:
            print("len(p): ", len(p))
        #print("len(y_samples): ",len(y_samples))
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        #if running_time > 200 or len(gt) + 2 < len(pred):
            #return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        print("threshs: ", threshs)
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]  #注意这里的两个for循环
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:         # s最多只能为4
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)
                # accuracy,                    fp,                                      fn

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]   #每行读取放入list
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]     #每行读取放入list
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')   # GT & pred same len
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']  #pred picture address
            #kh = raw_file.split('/')
            #raw_file = str(kh[-4])+'/'+str(kh[-3])+'/'+str(kh[-2])+'/'+str(kh[-1])
            #print("raw_file: ", raw_file)

            pred_lanes = pred['lanes']   # pred coordinates
            run_time = pred['run_time']

            if raw_file not in gts:
                print(raw_file)
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]         #GT picture address
            gt_lanes = gt['lanes']     #GT coordinates
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])


if __name__ == '__main__':
    pred = open('/Users/wutong/Desktop/prob_map/out.json', 'w')

    js = LaneEval.bench_one_submit('/Users/wutong/Desktop/prob_map/pred_test.json', '/Users/wutong/Downloads/test_set/ground_truth_test_label_copy 2.json')
    pred.write(js)
    pred.write('\n')
