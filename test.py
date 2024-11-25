import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np

import scipy.io as sio

import MyLib as ML
import SSMSFuse
import skimage.measure
import time

# FLAGS参数设置,用来添加参数
FLAGS = tf.app.flags.FLAGS

# Mode：train, test, testAll for test all sample
tf.app.flags.DEFINE_string('mode', 'testAll',
                           'train or testAll ')
# Prepare Data: if reprepare data samples for training and testing
tf.app.flags.DEFINE_string('Prepare', 'Yes',
                           'Yes or No')
# output channel number
tf.app.flags.DEFINE_integer('outDim', 46,
                           'output channel number')

# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                           'learning_rate')
# epoch number
tf.app.flags.DEFINE_integer('epoch', 100,
                           'epoch')
# path of testing sample
tf.app.flags.DEFINE_string('test_data_name', 'TestSample',
                           'Filepattern for eval data')
# path of training result
tf.app.flags.DEFINE_string('train_dir', 'temp\houston/',
                           'Directory to keep training outputs.')
# path of the testing result
tf.app.flags.DEFINE_string('test_dir', 'TestResult\houston/',
                           'Directory to keep eval outputs.')

# the size of training samples
tf.app.flags.DEFINE_integer('image_size', 96,
                            'Image side length.')

# number of gpus used for training
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')



def testAll():
    ## test all the testing samples
    X       = tf.placeholder(tf.float32, shape=(1,512, 512, 46))
    Y       = tf.placeholder(tf.float32, shape=(1, 512, 512 ,3))
    Z       = tf.placeholder(tf.float32, shape=(1, 512/32, 512/32, 46))

    outX = SSMSFuse.model(Y,Z)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    ML.mkdir(FLAGS.test_dir)
    elapsed_time = []

    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt)

        # 计算总参数量
        total_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape().as_list()
            num_params = 1
            for dim in shape:
                num_params *= dim
            total_params += num_params
        print("Total number of parameters:", total_params / 1000000, "M")

        for root, dirs, files in os.walk(r'Houstondata\X/'):

            for i in range(8,16):
                data = sio.loadmat(r'Houstondata\X/' + files[i-1])
                gtX = data['hsi']
                gtX = np.expand_dims(gtX, axis=0)

                data = sio.loadmat(r'Houstondata\Y/' + files[i-1])
                inY  = data['rgb']
                inY  = np.expand_dims(inY, axis = 0)

                data = sio.loadmat(r'Houstondata\Z/' + files[i-1])
                inZ  = data['hsi']
                inZ  = np.expand_dims(inZ, axis = 0)

                start_time = time.time()
                pred_X = sess.run([outX], feed_dict={Y:inY, Z:inZ})
                end_time = time.time()
                gap_time = end_time - start_time
                elapsed_time.append(gap_time)
                # 保存测试结果的mat格式
                sio.savemat(r'TestResult\houston/'+ files[i-1], {'outX': pred_X})

              #保存图片
                img_X = np.squeeze(np.array(pred_X),axis=0)
                showX = ML.get3band_of_tensor(img_X, nbanch=0, nframe=[5,25,45])
                maxS = np.max(showX)
                minS = np.min(showX)
                toshow = ML.setRange(ML.get3band_of_tensor(img_X, nbanch=0, nframe=[5,25,45]), maxS, minS)
                ML.imwrite(toshow, (r'TestResult\houston/%s.png' % (files[i-1])))

                print(files[i-1] + ' done!')

            print(f"平均推理时间为：{np.mean(elapsed_time)} 秒")


if __name__ == '__main__':
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'testAll':  # test all
            testAll()
