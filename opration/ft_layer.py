import glob,os,cv2
import imageio as io
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
from utils.utilities import *
from utils.utils_lap_pyramid import *
from keras.preprocessing.image import ImageDataGenerator
from loss.cal_loss import *
import tensorflow as tf
from keras import optimizers
from opration.data_gen import *
import matplotlib.pyplot as plt
from net_structure import *
from keras.models import load_model, model_from_json
from keras.utils import plot_model
import pickle, argparse
from keras.models import Model

ap = argparse.ArgumentParser()
ap.add_argument("--width",default=7768,help='width of input images')
ap.add_argument("--height", default=3301, help='height of input images')
ap.add_argument("--h_path",default='../dataset/train/hdr/', help='path of hdr images')
ap.add_argument("--l_path",default='../dataset/train/ldr/', help='path of ldr images')
ap.add_argument("--hdr_path", default='../dataset/test/hdr/', help='path of hdr images')
ap.add_argument("--random_patch_ratio_x",default=0.2, help='random patch ratio x of an image')
ap.add_argument("--random_patch_ratio_y", default=0.6, help='random patch ratio y of an image')
ap.add_argument("--patch_size",default=512, help='patch size of an image')
ap.add_argument("--random_patch_per_img", default=20, help='random_patch_per_image')
ap.add_argument("--level", default=4, help='levels of laplacian pyramid')
ap.add_argument("--epoch", default=6, help='epochs of training')
ap.add_argument("--batch_size", default=4, help='batch size of training')
ap.add_argument("--data_bottom_hdr", default='../dataset/test/hdr_bt.pkl', help='hdr images of bottom layer')
ap.add_argument("--data_high_hdr", default='../dataset/test/hdr_h.pkl', help='hdr images of high layer')
ap.add_argument("--data_bt_hdr", default='../dataset/train/hdr_bt.pkl', help='hdr images of bottom layer')
ap.add_argument("--data_h_hdr", default='../dataset/train/hdr_h.pkl', help='hdr images of high layer')
ap.add_argument("--data_ldr", default='../dataset/train/ldr.pkl', help='ldr images of fine tune layer')
ap.add_argument("--test_flag", default=True, help='Test or train fine tune layer')
ap.add_argument("--name", default='../dataset/test/name.pkl', help='Test or train fine tune layer')
ap.add_argument("--high_weight",  default='../checkpoint/high_layer/model_weights_h.h5', help='path of high model')
ap.add_argument("--bottom_weight",  default='../checkpoint/bot_layer/model_weights_bt.h5', help='path of bottom model')
ap.add_argument("--ft_weight",  default='../checkpoint/ft_layer/model_weights_ft.h5', help='path of bottom model')
ap.add_argument("--data_h_ldr", default='../dataset/train/ldr_h.pkl', help='ldr images of high layer')
ap.add_argument("--data_bt_ldr",  default='../dataset/train/ldr_bt.pkl', help='ldr images of high layer')
ap.add_argument("--plot",  default='../showprocess/plot_ft.png', help='path to output accuracy/loss plot of fine tune layer')
ap.add_argument("--bt_size",  default=(32, 32), help='path to output accuracy/loss plot of fine tune layer')
ap.add_argument("--h_size",  default=(512,512), help='path to output accuracy/loss plot of fine tune layer')
ap.add_argument("--bt_size_test", default=(243, 104), help='path to output accuracy/loss plot of fine tune layer')
ap.add_argument("--h_size_test",  default=(3884, 1650), help='path to output accuracy/loss plot of fine tune layer')

args = vars(ap.parse_args())
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


# construct the image generator for data augmentation
def data_gen(fr1, fr2, fr3):
    while True:
        hdr_h_arr = []
        hdr_bt_arr = []
        ldr_arr = []
        for i in range(args['batch_size']):
            try:
                hdr_h = pickle.load(fr1)
                hdr_bt = pickle.load(fr2)
                ldr = pickle.load(fr3)
            except EOFError:
                fr1 = open(args['data_h_hdr'], 'rb')
                fr2 = open(args['data_bt_hdr'], 'rb')
                fr3 = open(args['data_ldr'], 'rb')
            hdr_h_arr.append(hdr_h)
            hdr_bt_arr.append(hdr_bt)
            ldr_arr.append(ldr)
        h_hdr= np.array(hdr_h_arr)
        bt_hdr = np.array(hdr_bt_arr)
        ldr = np.array(ldr_arr)
        genX1 = aug.flow(h_hdr, ldr, batch_size=args['batch_size'])
        genX2 = aug.flow(bt_hdr, batch_size=args['batch_size'])
        x1i = genX1.next()
        x2i = genX2.next()
        yield [x1i[0], x2i], x1i[1]


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
if args['test_flag']:
    if not os.path.exists(args['data_high_hdr']):
        generate_test_data_from_file(args)
    fr1 = open(args['data_high_hdr'], 'rb')
    fr2 = open(args['data_bottom_hdr'], 'rb')
    fr3 = open(args['name'], 'rb')

    h = args['h_size_test']
    bt = args['bt_size_test']
else:
    if not os.path.exists(args['data_ldr']):
        generate_train_data_from_file(args)
    fr1 = open(args['data_h_hdr'], 'rb')
    fr2 = open(args['data_bt_hdr'], 'rb')
    fr3 = open(args['data_ldr'], 'rb')
    h = args['h_size']
    bt = args['bt_size']

# Model
# Construct high_layer
inputt_h = Input(shape=(h[1], h[0], 1), name='input1_h')
output_h = conv_relu_batch(inputt=inputt_h, name='conv1_h')
output_h = conv_relu_batch(inputt=output_h, name='conv2_h')
output_h = conv_relu_batch(inputt=output_h, name='conv3_h')
output_h = conv_relu_batch(inputt=output_h, name='conv4_h')
output_h = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv5_h')(output_h)
output_h = Add(name='add1_h')([inputt_h, output_h])

# Construct bottom_layer
inputt_bt = Input(shape=(bt[1], bt[0], 1), name='input1_bt')
output_bt = conv_relu_batch(inputt=inputt_bt, name='conv1_bt')
output_bt = conv_relu_batch(inputt=output_bt, name='conv2_bt')
output_bt = conv_relu_batch(inputt=output_bt, name='conv3_bt')
output_bt = conv_relu_batch(inputt=output_bt, name='conv4_bt')
output_bt = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv5_bt')(output_bt)
output_bt = Add(name='add1_bt')([inputt_bt, output_bt])
output_b = tf.reshape(output_bt, [args['batch_size'], bt[1], bt[0]])

# Expand bottom images (has the same size as high layer images)
bot_p_expand = 0
for i in range(args['batch_size']):
    bot_p = tf.squeeze(tf.slice(output_b, [i, 0, 0], [1, -1, -1]))

    index = tf.constant(0)
    n = tf.constant(args['level'])
    bot_p, index, n = tf.while_loop(cond, body, [bot_p, index, n], shape_invariants=[tf.TensorShape([None, None]),
                                                                             index.get_shape(), n.get_shape()])
    bot_p = tf.expand_dims(bot_p, axis=0)
    if i == 0:
        bot_p_expand = bot_p
    else:
        bot_p_expand = tf.concat([bot_p_expand, bot_p], axis=0)
bot_prediction = tf.expand_dims(bot_p_expand, axis=3)

# Add the output of high and bottom layer, then input to the fine tune layer
model_concat = Add(name='add')([output_h, bot_prediction])

# Construct fine tune layer
output = conv_relu_batch(inputt=model_concat, name='conv1_ft')
res1 = output
output = conv_relu_batch(inputt=output, name='conv2_ft')
output = conv_relu_batch(inputt=output, name='conv3_ft')
output = Add(name='add1_ft')([output, res1])

res2 = output
output = conv_relu_batch(inputt=output, name='conv4_ft')
output = conv_relu_batch(inputt=output, name='conv5_ft')
output = Add(name='add2_ft')([output, res2])

res3 = output
output = conv_relu_batch(inputt=output, name='conv6_ft')
output = conv_relu_batch(inputt=output, name='conv7_ft')
output = Add(name='add3_ft')([output, res3])

res4 = output
output = conv_relu_batch(inputt=output, name='conv8_ft')
output = conv_relu_batch(inputt=output, name='conv9_ft')
output = Add(name='add4_ft')([output, res4])

output = Conv2D(1, (1, 1), strides=(1, 1), padding='same', name='conv1_ft')(output)

# Construct Model(two inputs, one output)
model = Model(inputs=[inputt_h, inputt_bt], outputs=output_h)

# If test network, predict
if args['test_flag']:
    model.load_weights(args['ft_weight'])
    hdr_h = []
    hdr_bt = []
    name = []
    for i in range(args['batch_size']):
        hdr_h.append(pickle.load(fr1))
        hdr_bt.append(pickle.load(fr2))
        name.append(pickle.load(fr3))
    hdr_h_test = np.array(hdr_h)
    hdr_bt_test = np.array(hdr_bt)
    pre = model.predict([hdr_h_test, hdr_bt_test])

    for i in range(pre.shape[0]):
        predict = pre[i]
        pred = np.squeeze(predict)
        filename = name[i]
        # Normalize to [0, 255]
        pred = norm_0_to_255(pred)

        hdr_path = '../dataset/test/hdr/' + filename + '.hdr'
        hdr_img = io.imread(hdr_path)
        hdr_img = cv2.resize(hdr_img, (np.shape(pred)[1], np.shape(pred)[0]))
        hdr_gray = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)

        # bring back to RGB
        recovered_ldr = lum2rgb(pred, hdr_gray, hdr_img)
        recovered_ldr[recovered_ldr > 255] = 255
        recovered_ldr[recovered_ldr < 0] = 0

        # store results
        recovered_ldr = norm_0_to_255(recovered_ldr)
        io.imwrite('../dataset/result/' + filename + '_predict.jpg', recovered_ldr)
else:
    # Else, train network
    model.load_weights(args['bottom_weight'])
    model.load_weights(args['high_weight'])

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    H = model.fit_generator(data_gen(fr1, fr2, fr3),
                        steps_per_epoch=2600, epochs=args['epoch'])
    model.save_weights(args['ft_weight'])

    # plot the training loss and accuracy
    N = np.arange(0, args['epoch'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.title("Training Loss and Accuracy (Fine Tune Layer)")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["plot"])
    model.summary()

fr1.close()
fr2.close()
fr3.close()
