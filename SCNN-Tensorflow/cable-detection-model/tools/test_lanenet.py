
"""
测试LaneNet模型 第一步
Step 1
"""
import os,sys
import os.path as ops
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import argparse
import math
import tensorflow as tf
import glog as log
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test
import numpy as np

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='False')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=4) #############
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()



def test_lanenet(image_path, weights_path, use_gpu, image_list, batch_size, save_dir):

    """
    :param image_path:
    :param weights_path:  ***
    :param use_gpu:
    :return:
    """
    print("6666666666666")
    global total_img
    test_dataset = lanenet_data_processor_test.DataSet(image_path, batch_size)
    input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)
    phase_tensor = tf.constant('test', tf.string)  # str常量

    net = lanenet_merge_model.LaneNet()

    binary_seg_ret, instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')
    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    print(len(final_var))   # 85
    # Pass the variables as a list:
    saver = tf.train.Saver(final_var)
    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=weights_path)       #.ckpt //save_path: Path where parameters were previously saved.
        # There is a mismatch between the graph and the checkpoint being loaded.
        print("Model restored.")
        #print(len(image_list) / batch_size)   #1.0
        for i in range(int(math.ceil(len(image_list) / batch_size))):
            print("i: ", i)
            paths = test_dataset.next_batch()
            #print(paths)    # 1.jpg ~ 8.jpg

            instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: paths})
            #print('instance_seg_image shape: ',instance_seg_image.shape)

            #image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in paths]
            #image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            #instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],
                                                            #feed_dict={input_tensor: image_list_epoch})


            for cnt, image_name in enumerate(paths):
                total_img = np.zeros([288, 800])
                #print(image_name)
                parent_path = os.path.dirname(image_name)
                #print(parent_path)  /Users/wutong/Downloads/test_set/clips/0601/1494453197736907986
                #                   //Users/wutong/Downloads/train_set/clips/0313-1/8580/20.jpg
                ph = parent_path.split('/')
                directory = os.path.join(save_dir, 'cable_pic', ph[-1])

                if not os.path.exists(directory):
                    os.makedirs(directory)

                file_exist = open(os.path.join(directory, os.path.basename(image_name)[:-3] + 'exist.txt'), 'w')
                for cnt_img in range(4):  # 4 lines
                    cv2.imwrite(os.path.join(directory, os.path.basename(image_name)[:-4] + '_' + str(cnt_img + 1) + '_avg.png'),
                            (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int))
                    if existence_output[cnt, cnt_img] > 0.5:   # >0.5 suppose that have a line
                        #file_exist.write('%s ' % existence_output[cnt, cnt_img])
                        file_exist.write('1 ')
                        total_img += (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int)
                    else:
                        file_exist.write('0 ')

                cv2.imwrite(os.path.join(directory, os.path.basename(image_name)[:-4] + '_' + 'total_img.png'),
                            total_img)




                file_exist.close()
    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

#   if args.save_dir is not None and not ops.exists(args.save_dir):
#        log.error('{:s} not exist and has been made'.format(args.save_dir))
#        os.makedirs(args.save_dir)

#    save_dir = os.path.join(args.image_path, 'predicts')
#    if args.save_dir is not None:
#        save_dir = args.save_dir

    img_name = []
    image_path = '/Users/wutong/Pictures/grass/test_list.txt'
    with open(str(image_path), 'r') as g:
        for line in g.readlines():
            img_name.append(line.strip())
    test_lanenet(image_path="/Users/wutong/Pictures/grass/test_list.txt",
                 weights_path="/Users/wutong/Downloads/culane_lanenet_vgg_2019-06-05-13-41-40.ckpt-9000",
                 use_gpu=False, image_list=img_name, batch_size=CFG.TEST.BATCH_SIZE, save_dir='/Users/wutong/Pictures/grass/pictures')
    #test_lanenet(args.image_path, args.weights_path, args.use_gpu, args.batch_size, args.save_dir, image_list=img_name)

# args.image_path :　~/Download/model_culane-71-3/test_img.txt
# args.weights_path :  ~/Download/model_culane-71-3/culane_lanenet_vgg.ckpt
# args.use_gpu :  False

# args.batch_size : 1
# save_dir : ~/Download/model_culane-71-3/sv
