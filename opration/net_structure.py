from keras.models import Sequential
from keras.layers import Conv2D, merge, Activation, BatchNormalization, Input,Add
from keras import backend as K
from keras.initializers import TruncatedNormal, Constant
import tensorflow as tf
from keras.models import Model
def net_high_layer(args):

    inputt = Input(shape=(args['patch_size'], args['patch_size'], 1), name='input1_h')
    output = conv_relu_batch(inputt=inputt, name='conv1_h')
    output = conv_relu_batch(inputt=output, name='conv2_h')
    output = conv_relu_batch(inputt=output, name='conv3_h')
    output = conv_relu_batch(inputt=output, name='conv4_h')
    output = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv5_h')(output)
    output = Add(name='add1_h')([inputt, output])
    model = Model(inputt, output, name='model1_h')
    return model


def net_bottom_layer(args):
    inputt = Input(shape=(args['size_bt'], args['size_bt'], 1), name='input1_bt')
    output = conv_relu_batch(inputt=inputt, name='conv1_bt')
    output = conv_relu_batch(inputt=output, name='conv2_tb')
    output = conv_relu_batch(inputt=output, name='conv3_bt')
    output = conv_relu_batch(inputt=output, name='conv4_bt')
    output = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv5_bt')(output)
    output = Add(name='add1_bt')([inputt, output])
    model = Model(inputt, output, name='model1_bt')
    return model


def conv_relu_batch(inputt, num=32, in_dim=3, out_dim=3, padding='same', name='conv'):
    output = Conv2D(filters=num, kernel_size=(in_dim, out_dim), strides=(1, 1), padding=padding,bias_initializer=Constant(0.1),kernel_initializer=TruncatedNormal(stddev=0.1), activation='relu', name="{}_conv".format(name))(inputt)
    output = BatchNormalization(epsilon=0.01, name="{}_batch".format(name))(output)
    return output

