import tensorflow as tf
#from utils.configs import *
from loss.custom_vgg16 import *
#import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16

class cal_loss(object):

    def __init__(self, img, gt, vgg_path, withtv=False):
        self.data_dict = loadWeightsData(vgg_path)

        """Build Perceptual Losses"""
        with tf.name_scope(name='vgg_16' + "_run_vgg16"):
            # content target feature
            vgg_c = custom_Vgg16(gt, data_dict=self.data_dict)
            #vgg_c = VGG16(include_top=False,input_tensor=gt, input_shape=(51, 121, 3))
            fe_generated = [vgg_c.conv1_1, vgg_c.conv2_1, vgg_c.conv3_1, vgg_c.conv4_1, vgg_c.conv5_1]

            # feature after transformation
            vgg = custom_Vgg16(img, data_dict=self.data_dict)
            #vgg = VGG16(input_tensor=img, input_shape=(51, 121, 3))
            fe_input = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]

        with tf.name_scope(name='vgg_16' + "_cal_content_L"):
            # compute feature loss
            loss = 0
            for f_g, f_i in zip(fe_generated, fe_input):
                loss += tf.reduce_mean(tf.abs(f_g - f_i))
        self.loss = loss





