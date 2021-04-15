This is the implementation for Deep Reformulated Laplacian Tone Mapping. And code here is a replicate by Keras of [our previous code by Tensorflow](https://github.com/linmc86/Deep-Reformulated-Laplacian-Tone-Mapping).

# Deep-Reformulated-Laplacian-Tone-Mapping-Keras


## Introduction
<img src="https://raw.githubusercontent.com/Taichi-Pink/Deep-Reformulated-Laplacian-Tone-Mapping-Keras/master/result/9C4A1511-702551eb64_predict.jpg" width="430"> <img src="https://raw.githubusercontent.com/Taichi-Pink/Deep-Reformulated-Laplacian-Tone-Mapping-Keras/master/result/9C4A0221-feaaa06d6f_predict.jpg" width="430">

<img src="https://raw.githubusercontent.com/Taichi-Pink/Deep-Reformulated-Laplacian-Tone-Mapping-Keras/master/result/9C4A3782-70b3083cee_predict.jpg" width="430"> <img src="https://raw.githubusercontent.com/Taichi-Pink/Deep-Reformulated-Laplacian-Tone-Mapping-Keras/master/result/9C4A4301-9fd6373e60_predict.jpg" width="430">

This is the implementation for [Deep Reformulated Laplacian Tone Mapping](https://arxiv.org/abs/2102.00348)

This algorithm is developed by [I2Sense Lab](https://www.ucalgary.ca/i2sense/) of [University of Calgary](https://www.ucalgary.ca/)., we have proposed a new tone mapping method--a novel reformulated Laplacian method to 
```decompose a WDR image into a low-resolution image``` (contains the low frequency component of the WDR image) ```and a high-resolution image```(contains the remaining higher frequencies of the WDR image).


The two images are processed by a ```global compression network```(compress the global scale gradient of a WDR image) and a ```local manipulation neural network```(manipulate local features), respectively. The generated images from the two networks are further merged together, and then input merged images to ```fine tune network``` to produce the final image.

## Get Started
### Prerequistes
* Download [Pyacharm 2019](https://www.jetbrains.com/pycharm/download/#section=linux)
* Download [pretrained](https://drive.google.com/open?id=1HZel2k51rzWchYdYxdHXbE2AvRM0lyrM) file, and place it under `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/loss/pretrained/` folder.
* Download [dataset](https://drive.google.com/open?id=1da70CuYLFsVePvjimS_sUKExv6vO2xyM) file, and place it under `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/dataset/` folder.
* Tensorflow 1.9.0

### Instructions

##### Pycharm setup 
* Python 2.7
* Tensorflow-gpu 1.9.0
* Keras 2.2.4
* Opencv-python 3.4.4.19
* Scipy 1.1.0
* Matplotlib 2.2.3

#### Demo
* Set `test_flag` to `True` in `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/ft_layer.py` file
* Run this file.
  
#### Train
* Download [Laval Indoor dataset](http://indoor.hdrdb.com/)(EULA required).

* Download [Luminance HDR](https://github.com/luminancehdr/luminancehdr) for tonemapping.

* Generate the label images.
  
  >Tonemapping the [Laval Indoor dataset](http://indoor.hdrdb.com/) `.hdr` files by
  [Luminance HDR](https://github.com/luminancehdr/luminancehdr), save the tonemapped images in `.jpg`.

* Follow the data preprocessing steps specified on our paper to process the data.

* Divide the data in train set and test set.
  >Place the training `.hdr` images under `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/dataset/train/hdr/` folder, and the       corresponding label images created from step 3 under `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/dataset/train/ldr/` folder. 
  >
  >Place the testing `.hdr` images under `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/dataset/test/hdr/` folder, and the corresponding label images created from step 3 under `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/dataset/test/ldr/` folder.

* Train high layer

  >run `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/high_layer.py` file.
* Train bottom layer

  >run `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/bot_layer.py` file.
* Train fine tune layer
  >Set `test_flag` to `False` in `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/ft_layer.py` file
  >
  >Run this file.

#### Test
* Set `test_flag` to `True` in `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/ft_layer.py` file
* Run this file.

## Contact

If you have ang issue,please concat [Prof. Yadid-Pecht](https://www.ucalgary.ca/i2sense/yadid_pecht_biography) or [Dr.Jie Yang](https://jieyang1987.github.io/).
* email:

   >orly.yadid-pecht@ucalgary.ca
   >
   >yangjie@westlake.edu.cn
   >
   >mengchen.lin@ucalgary.ca
   >
   >ziyi.liu1@ucalgary.ca

