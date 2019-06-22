from keras.models import Sequential
from keras.layers import Conv2D, merge, Activation, BatchNormalization, Input,Add
from keras import backend as K
import tensorflow as tf
from keras.models import Model
def net_high_layer():
    #shape = hdr.shape
    inputt = Input(shape=(512, 512, 1), name='input1_h')
    output = conv_relu_batch(inputt=inputt, name='conv1_h')
    output = conv_relu_batch(inputt=output, name='conv2_h')
    output = conv_relu_batch(inputt=output, name='conv3_h')
    output = conv_relu_batch(inputt=output, name='conv4_h')
    output = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv5_h')(output)
    output = Add(name='add1_h')([inputt, output])
    model = Model(inputt, output, name='model1_h')
    return model


def net_bottom_layer(hdr):
    shape = hdr.shape
    inputt = Input(shape=(shape[1], shape[2], shape[3]), name='input1_bt')
    output = conv_relu_batch(inputt=inputt, name='conv1_bt')
    output = conv_relu_batch(inputt=output, name='conv2_tb')
    output = conv_relu_batch(inputt=output, name='conv3_bt')
    output = conv_relu_batch(inputt=output, name='conv4_bt')
    output = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv5_bt')(output)
    output = Add(name='add1_bt')([inputt, output])
    model = Model(inputt, output, name='model1_bt')
    return model


def net_ft_layer(hdr):
    shape = hdr.shape
    inputt = Input(shape=(shape[1], shape[2], shape[3]), name='input1_ft')
    output = conv_relu_batch(inputt=inputt, name='conv1_ft')
    res1 = output
    output = conv_relu_batch(inputt=output, name='conv2_ft')
    output = conv_relu_batch(inputt=output, name='conv3_ft')
    output = Add(name='add1_ft')[output, res1]

    res2 = output
    output = conv_relu_batch(inputt=output, name='conv4_ft')
    output = conv_relu_batch(inputt=output, name='conv5_ft')
    output = Add(name='add2_ft')[output, res2]

    res3 = output
    output = conv_relu_batch(inputt=output, name='conv6_ft')
    output = conv_relu_batch(inputt=output, name='conv7_ft')
    output = Add(name='add3_ft')[output, res3]

    res4 = output
    output = conv_relu_batch(inputt=output, name='conv8_ft')
    output = conv_relu_batch(inputt=output, name='conv9_ft')
    output = Add(name='add4_ft')[output, res4]

    output = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv1_ft')(output)
    model = Model(inputt, output, name='model1_ft')
    return model


def conv_relu_batch(inputt, num=32, in_dim=3, out_dim=3, padding='same', name='conv'):
    output = Conv2D(filters=num, kernel_size=(in_dim, out_dim), strides=(1, 1), padding=padding, activation='relu', name="{}_conv".format(name))(inputt)
    output = BatchNormalization(epsilon=0.01, name="{}_batch".format(name))(output)
    return output

