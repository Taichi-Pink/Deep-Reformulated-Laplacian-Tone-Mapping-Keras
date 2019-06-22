import glob, os, cv2
import imageio as io
from utils.utilities import *
from utils.utils_lap_pyramid import *
from loss.cal_loss import *
import tensorflow as tf
import pickle


def dualize(py_layers):
    freq_layer = 0
    bottom_layer = py_layers[-1]
    freq_layers = py_layers[:-1]
    for item in range(0, len(freq_layers)):
        freq_layer += freq_layers[item]

    dual_layers = [freq_layer, bottom_layer]
    return dual_layers


def crop_random(img, label, x, y, size, N):
    imgpatchs = []
    labelpatchs = []
    h, w = np.shape(img)

    for i in range(N):
        rand_coe_h = random.random() * (y - x) + x
        rand_coe_w = random.random() * (y - x) + x

        # get width and height of the patch
        rand_h = int(h * rand_coe_h)
        rand_w = int(w * rand_coe_w)

        # the random - generated coordinates are limited in
        # h -> [0, coor_h]
        # w -> [0, coor_w]
        coor_h = h - rand_h
        coor_w = w - rand_w

        # get x and y starting point of the patch
        coor_x = int(random.random() * coor_h)
        coor_y = int(random.random() * coor_w)

        # only create patches for the high layer
        img_patch = img[coor_x:coor_x + rand_h, coor_y:coor_y + rand_w]
        # resize the patch to [size, size]
        resize_img = cv2.resize(img_patch, (size, size))
        imgpatchs.append(resize_img)

        # Create patches for the label
        label_patch = label[coor_x:coor_x + rand_h, coor_y:coor_y + rand_w]
        # resize the patch to [size, size]
        resize_label = cv2.resize(label_patch, (size, size))
        labelpatchs.append(resize_label)

    return imgpatchs, labelpatchs


def generate_train_data_from_file(args):
    file_list = glob.glob(args["h_path"]+'*.{}'.format('hdr'))
    array_hdr_h = []
    array_ldr_h = []
    array_hdr_bt = []
    array_ldr_bt = []
    array_label = []

    fr1 = open(args['data_h_hdr'], 'wb')
    fr2 = open(args['data_bt_hdr'], 'wb')
    fr3 = open(args['data_h_ldr'], 'wb')
    fr4 = open(args['data_bt_ldr'], 'wb')
    fr5 = open(args['data_ldr'], 'wb')

    for i in range(len(file_list)):
        hdr_path = file_list[i]
        img_name = os.path.splitext(os.path.basename(hdr_path))[0]
        ldr_path = args["l_path"]+img_name+'.jpg'
        hdr_img = io.imread(hdr_path)
        ldr_img = io.imread(ldr_path)

        tf.logging.info('preprocess image {}'.format(i))

        hdr_img = cv2.resize(hdr_img, (int(args["width"]/2), int(args["height"]/2)))
        hdr_gray = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
        hdr_gray_clipped = cut_dark_end_percent(hdr_gray, 0.001)
        hdr_logged = np.log(hdr_gray_clipped + np.finfo(float).eps)
        hdr_preprocess = np.clip(hdr_logged, a_min=-50, a_max=np.max(hdr_logged))

        hdr_norm = norm_0_to_1(hdr_preprocess)
        ldr_img = cv2.resize(ldr_img, (args["width"] // 2, args["height"] // 2))
        ldr_gray = cv2.cvtColor(ldr_img, cv2.COLOR_RGB2GRAY)
        ldr_norm = norm_0_to_1(ldr_gray)

        img_rand_patches, label_rand_patches = crop_random(hdr_norm, ldr_norm, args["random_patch_ratio_x"],
                               args["random_patch_ratio_y"], args["patch_size"], args["random_patch_per_img"])
        num = len(img_rand_patches)
        for j in range(num):
            tf.logging.info('crop image {}'.format(j))

            ldr_label = label_rand_patches[j]
            ldr_label = np.expand_dims(ldr_label, axis=2)
            #array_label.append(np.array(ldr_label))
            pickle.dump(ldr_label, fr5, -1)
            img_rand_patches[j] = lpyr_gen(img_rand_patches[j], args["level"])
            label_rand_patches[j] = lpyr_gen(label_rand_patches[j], args["level"])

            img_rand_patches[j], _ = lpyr_enlarge_to_top_but_bottom(img_rand_patches[j])
            label_rand_patches[j], _ = lpyr_enlarge_to_top_but_bottom(label_rand_patches[j])

            img_rand_patches[j] = dualize(img_rand_patches[j])
            label_rand_patches[j] = dualize(label_rand_patches[j])

            for k in range(0,2):
                img_rand_patches[j][k] = np.expand_dims(img_rand_patches[j][k], axis=2)
                label_rand_patches[j][k] = np.expand_dims(label_rand_patches[j][k], axis=2)
            pickle.dump(img_rand_patches[j][0], fr1, -1)
            pickle.dump(img_rand_patches[j][1], fr2, -1)
            pickle.dump(label_rand_patches[j][0], fr3, -1)
            pickle.dump(label_rand_patches[j][1], fr4, -1)

    fr1.close()
    fr2.close()
    fr3.close()
    fr4.close()
    fr5.close()
    #         array_hdr_h.append(img_rand_patches[i][0])
    #         array_hdr_bt.append(img_rand_patches[i][1])
    #         array_ldr_h.append(label_rand_patches[i][0])
    #         array_ldr_bt.append(label_rand_patches[i][1])
    #
    # hdr_h = np.array(array_hdr_h)
    # hdr_bt = np.array(array_hdr_bt)
    # ldr_h = np.array(array_ldr_h)
    # ldr_bt = np.array(array_ldr_bt)
    # ldr = np.array(array_label)
    #
    # fr = open(args['data_h_hdr'], 'wb')
    # pickle.dump(hdr_h, fr, -1)
    # fr.close()
    #
    # fr = open(args['data_bt_hdr'], 'wb')
    # pickle.dump(hdr_bt, fr, -1)
    # fr.close()
    #
    # fr = open(args['data_h_ldr'], 'wb')
    # pickle.dump(ldr_h, fr, -1)
    # fr.close()
    #
    # fr = open(args['data_bt_ldr'], 'wb')
    # pickle.dump(ldr_bt, fr, -1)
    # fr.close()
    #
    # fr = open(args['data_ldr'], 'wb')
    # pickle.dump(ldr, fr, -1)
    # fr.close()

    #return hdr_h, ldr_h, hdr_bt, ldr_bt, ldr


def generate_test_data_from_file(args):
    file_list = glob.glob(args["h_path"] + '*.{}'.format('hdr'))
    array_hdr_h = []
    array_hdr_bt = []
    array_name = []
    for i in range(len(file_list)):
        hdr_path = file_list[i]

        name = os.path.splitext(os.path.basename(hdr_path))[0]
        array_name.append(name)

        hdr_img = io.imread(hdr_path)
        hdr_img = cv2.resize(hdr_img, (args["width"] / 2, args["height"] / 2))
        hdr_gray = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
        hdr_gray_clipped = cut_dark_end_percent(hdr_gray, 0.001)
        hdr_logged = np.log(hdr_gray_clipped + np.finfo(float).eps)
        hdr_preprocess = np.clip(hdr_logged, a_min=-50, a_max=np.max(hdr_logged))

        hdr_norm = norm_0_to_1(hdr_preprocess)
        hdr_lpyr = lpyr_gen(hdr_norm, args["level"])
        hdr_lpyr, _ = lpyr_enlarge_to_top_but_bottom(hdr_lpyr)
        hdr_lpyr_dul = dualize(hdr_lpyr)
        hdr_lpyr_dul[0] = np.expand_dims(hdr_lpyr_dul[0], axis=2)
        hdr_lpyr_dul[1] = np.expand_dims(hdr_lpyr_dul[1], axis=2)

        array_hdr_h.append(hdr_lpyr_dul[0])
        array_hdr_bt.append(hdr_lpyr_dul[1])

    hdr_h = np.array(array_hdr_h)
    hdr_bt = np.array(array_hdr_bt)
    name_list = np.array(array_name)
    fr = open(args['data_high_hdr'], 'wb')
    pickle.dump(hdr_h, fr, -1)
    fr.close()

    fr = open(args['data_bottom_hdr'], 'wb')
    pickle.dump(hdr_bt, fr, -1)
    fr.close()

    fr = open(args['name'], 'wb')
    pickle.dump(name_list, fr, -1)
    fr.close()
    return hdr_h, hdr_bt,name_list
