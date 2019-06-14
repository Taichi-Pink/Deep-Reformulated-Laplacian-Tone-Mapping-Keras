import keras
from keras.layers import Dense, Conv2D, BatchNormalization
from utils.utilities import *
from utils.utils_lap_pyramid import *
from keras.preprocessing.image import ImageDataGenerator
from loss.cal_loss import *
import tensorflow as tf
from keras import optimizers
import argparse, pickle
from opration.data_gen import *
import matplotlib.pyplot as plt
from net_structure import *
from keras.models import load_model, Model
from keras.utils import plot_model

ap = argparse.ArgumentParser()
ap.add_argument("-w","--width", required=False, default=2048, help='width of input images')
ap.add_argument("-hei","--height", required=False, default=1024, help='height of input images')
ap.add_argument("-h_p","--h_path", required=False, default='../dataset/train/hdr/', help='path of hdr images')
ap.add_argument("-l_p", "--l_path", required=False, default='../dataset/train/ldr/', help='path of ldr images')
ap.add_argument("-x_ratio","--random_patch_ratio_x", required=False, default=0.2, help='random patch ratio x of an image')
ap.add_argument("-y_ratio","--random_patch_ratio_y", required=False, default=0.6, help='random patch ratio y of an image')
ap.add_argument("-p_size","--patch_size", required=False, default=512, help='patch size of an image')
ap.add_argument("-rp_per","--random_patch_per_img", required=False, default=20, help='random_patch_per_image')
ap.add_argument("-l","--level", required=False, default=4, help='levels of laplacian pyramid')
ap.add_argument("-e","--epoch", required=False, default=6, help='epochs of training')
ap.add_argument("-b","--batch_size", required=False, default=2, help='batch size of training')
ap.add_argument("-bottom","--bottom_weight", required=False, default='../checkpoint/bot_layer/model_weights_bt.h5', help='path of bottom model')
ap.add_argument("-data_bt_h","--data_bt_hdr", required=False, default='../dataset/train/hdr_bt.pkl', help='hdr images of bottom layer')
ap.add_argument("-data_bt_l","--data_bt_ldr", required=False, default='../dataset/train/ldr_bt.pkl', help='ldr images of bottom layer')
ap.add_argument("-data_h_h","--data_h_hdr", required=False, default='../dataset/train/hdr_h.pkl', help='hdr images of high layer')
ap.add_argument("-data_h_l","--data_h_ldr", required=False, default='../dataset/train/ldr_h.pkl', help='ldr images of high layer')
ap.add_argument("-data_l","--data_ldr", required=False, default='../dataset/train/ldr.pkl', help='ldr images of fine tune layer')
ap.add_argument("-plt","--plot", required=False, default='../showprocess/plot_bt.png', help='path to output accuracy/loss plot of bottom layer')

args = vars(ap.parse_args())


def loss(gt_gray, output):
    """Build Losses"""
    loss_l1_reg = 0
    loss_l1 = tf.reduce_mean(tf.abs(output - gt_gray))

    # Calculate L2 Regularization value based on trainable weights in the network:
    weight_size = 0
    for variable in tf.trainable_variables():
        if not (variable.name.startswith('vgg_16')):
            loss_l1_reg += tf.reduce_sum(tf.cast(tf.abs(variable), tf.float32)) * 2
            weight_size += tf.size(variable)
    loss_l2_reg = loss_l1_reg / tf.to_float(weight_size)

    '''perceptual loss'''
    # duplicate the colour channel to be 3 same layers.
    output_3_channels = tf.concat([output, output, output], axis=3)
    gt_gray_3_channels = tf.concat([gt_gray, gt_gray, gt_gray], axis=3)

    losses = cal_loss(output_3_channels, gt_gray_3_channels, '../loss/pretrained/vgg16.npy')
    loss = losses.loss / 3

    losss = loss * 0.5 + loss_l1 * 0.5 + loss_l2_reg * 0.2
    return losss


# Import data
if os.path.exists(args['data_bt_hdr']):
    fr = open(args['data_bt_hdr'], 'rb')
    hdr = pickle.load(fr)
    fr.close()
    fr = open(args['data_bt_ldr'], 'rb')
    ldr = pickle.load(fr)
    fr.close()
else:
    _, _, hdr, ldr, _ = generate_train_data_from_file(args)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Model
model = net_bottom_layer(hdr)
adam = optimizers.Adam(lr=0.001)
model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
H = model.fit_generator(aug.flow(hdr, ldr, batch_size=args['batch_size']),
                    steps_per_epoch=hdr.shape[0]/args['batch_size'], epochs=args['epoch'])
model.save_weights(args['bottom_weight'])

# plot the training loss and accuracy
N = np.arange(0, args['epoch'])
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.title("Training Loss and Accuracy (Bottom Layer)")
plt.xlabel("Epoch ")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
model.summary()


