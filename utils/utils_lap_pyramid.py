import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pyr(img, lev):
    img = img.astype(np.float32)
    gass_img = [img]
    cur_img = img
    for i in range(lev):
        cur_img = cv2.pyrDown(cur_img)
        gass_img.append(cur_img)
    return gass_img


# generate the laplacian pyramid from an image with specified number of levels
def lpyr_gen(img, lev=2):
    img = img.astype(np.float32)
    gass_img = gaussian_pyr(img, lev)
    lpyr_img = []
    for i in range(lev):
        g_img = gass_img[i]
        w = g_img.shape[0]
        h = g_img.shape[1]
        up_img = cv2.pyrUp(gass_img[i+1], dstsize=(h, w))
        lpyr_img.append(g_img-up_img)
    lpyr_img.append(gass_img[-1])
    return lpyr_img


def lpyr_recons(l_pyr):
    lev = len(l_pyr)
    cur_l = l_pyr[-1]
    for index in range(lev-2,-1,-1):
        img_shape = np.shape(l_pyr[index])
        next_w = img_shape[0]
        next_h = img_shape[1]
        cur_l = cv2.pyrUp(cur_l,dstsize=(next_h,next_w))
        next_l = l_pyr[index]
        cur_l = cur_l + next_l
    return cur_l


def lpyr_upsample(l_img, levels):
    lev = len(levels)-1
    cur_img = l_img
    for i in range(lev):
        h = levels[lev-1-i][0]
        w = levels[lev-1-i][1]
        cur_img = cv2.pyrUp(cur_img, dstsize=(w, h))
    return cur_img



# make all levels of pyramid to the same size as the largest one
def lpyr_enlarge_to_top(l_pyr):
    lev = len(l_pyr)
    levels = []
    cur_l=[]
    for index in range(lev):
        levels.append((l_pyr[index].shape))
        aligned = lpyr_upsample(l_pyr[index], levels)
        cur_l.append(aligned)
    return cur_l


# make all levels of pyramid to the same size as the largest one but the bottom layer
def lpyr_enlarge_to_top_but_bottom(l_pyr):
    levels = []
    cur_img = []
    for i in range(len(l_pyr)-1):
        levels.append(l_pyr[i].shape)
        up_img = lpyr_upsample(l_pyr[i], levels)
        cur_img.append(up_img)
    cur_img.append(l_pyr[-1])
    return cur_img, levels

# only upsamples one layer the bottom layer to specific size
def lpyr_enlarge_bottom_to_top(l_pyr, levels):
    levels.append((l_pyr[-1].shape))
    upsampled = lpyr_upsample(l_pyr[-1], levels)
    l_pyr[-1] = upsampled
    return l_pyr


def call2dtensorgaussfilter():
    return tf.constant([[1./256., 4./256., 6./256., 4./256., 1./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [6./256., 24./256., 36./256., 24./256., 6./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [1./256., 4./256., 6./256., 4./256., 1./256.]])


def applygaussian(imgs):
    gauss_f = call2dtensorgaussfilter()
    gauss_f = tf.expand_dims(gauss_f, axis=2)
    gauss_f = tf.expand_dims(gauss_f, axis=3)

    result = tf.nn.conv2d(imgs, gauss_f * 4, strides=[1, 1, 1, 1], padding="VALID")
    result = tf.squeeze(result, axis=0)
    result = tf.squeeze(result, axis=2)
    return result


def dilatezeros(imgs):
        zeros = tf.zeros_like(imgs)
        column_zeros = tf.reshape(tf.stack([imgs, zeros], 2), [-1, tf.shape(imgs)[1] + tf.shape(zeros)[1]])[:,:-1]
        row_zeros = tf.transpose(column_zeros)

        zeros = tf.zeros_like(row_zeros)
        dilated = tf.reshape(tf.stack([row_zeros, zeros], 2), [-1, tf.shape(row_zeros)[1] + tf.shape(zeros)[1]])[:,:-1]
        dilated = tf.transpose(dilated)

        paddings = tf.constant([[0, 1], [0, 1]])
        dilated = tf.pad(dilated, paddings, "REFLECT")

        dilated = tf.expand_dims(dilated, axis=0)
        dilated = tf.expand_dims(dilated, axis=3)

        return dilated


# funcs for tf.while_loop ====================================
def body(output_bot, i, n):
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    output_bot = dilatezeros(output_bot)
    output_bot = tf.pad(output_bot, paddings, "REFLECT")
    output_bot = applygaussian(output_bot)
    return output_bot, tf.add(i, 1), n


def cond(output_bot, i, n):
    return tf.less(i, n)


















