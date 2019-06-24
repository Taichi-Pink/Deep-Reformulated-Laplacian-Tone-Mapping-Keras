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
ap.add_argument("--width",  default=7768, help='width of input images')
ap.add_argument("--height",  default=3301, help='height of input images')
ap.add_argument("--h_path",  default='../dataset/train/hdr/', help='path of hdr images')
ap.add_argument("--l_path",  default='../dataset/train/ldr/', help='path of ldr images')
ap.add_argument("--random_patch_ratio_x",default=0.2, help='random patch ratio x of an image')
ap.add_argument("--random_patch_ratio_y",  default=0.6, help='random patch ratio y of an image')
ap.add_argument("--patch_size",  default=512, help='patch size of an image')
ap.add_argument("--random_patch_per_img",  default=20, help='random_patch_per_image')
ap.add_argument("--level",  default=4, help='levels of laplacian pyramid')
ap.add_argument("--epoch",  default=6, help='epochs of training')
ap.add_argument("--batch_size",  default=32, help='batch size of training')
ap.add_argument("--bottom_weight",  default='../checkpoint/bot_layer/model_weights_bt.h5', help='path of bottom model')
ap.add_argument("--data_bt_hdr",  default='../dataset/train/hdr_bt.pkl', help='hdr images of bottom layer')
ap.add_argument("--data_bt_ldr",  default='../dataset/train/ldr_bt.pkl', help='ldr images of bottom layer')
ap.add_argument("--data_h_hdr",  default='../dataset/train/hdr_h.pkl', help='hdr images of high layer')
ap.add_argument("--data_h_ldr",  default='../dataset/train/ldr_h.pkl', help='ldr images of high layer')
ap.add_argument("--data_ldr",  default='../dataset/train/ldr.pkl', help='ldr images of fine tune layer')
ap.add_argument("--plot",  default='../showprocess/plot_bt.png', help='path to output accuracy/loss plot of bottom layer')
ap.add_argument("--size_bt",  default=32, help='size of bottom image')
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
def data_gen(fr1, fr2):
    while True:
        hdr_arr = []
        ldr_arr = []
        for i in range(args['batch_size']):
            try:
                hdr = pickle.load(fr1)
                ldr = pickle.load(fr2)
            except EOFError:
                fr1 = open(args['data_bt_hdr'], 'rb')
                fr2 = open(args['data_bt_ldr'], 'rb')
            hdr_arr.append(hdr)
            ldr_arr.append(ldr)
        hdr_bt = np.array(hdr_arr)
        ldr_bt = np.array(ldr_arr)
        gen = aug.flow(hdr_bt, ldr_bt, batch_size=args['batch_size'])
        out = gen.next()
        a = out[0]
        b = out[1]
        yield a, b


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

if not os.path.exists(args['data_bt_hdr']):
    generate_train_data_from_file(args)

fr1 = open(args['data_bt_hdr'], 'rb')
fr2 = open(args['data_bt_ldr'], 'rb')
# Model
model = net_bottom_layer()
adam = optimizers.Adam(lr=0.001)
model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
H = model.fit_generator(data_gen(fr1, fr2),
                    steps_per_epoch=1300, epochs=args['epoch'])
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


