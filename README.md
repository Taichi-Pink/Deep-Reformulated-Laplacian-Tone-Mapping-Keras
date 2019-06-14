This is the implementation for Deep Reformulated Laplacian Tone Mapping. And code here is a replicate by Keras of [our previous code by Tensorflow](https://github.com/linmc86/Deep-Reformulated-Laplacian-Tone-Mapping).

# Deep-Reformulated-Laplacian-Tone-Mapping-Keras


## Introduction
![pink](https://raw.githubusercontent.com/PinkLoveyi/Deep-Reformulated-Laplacian-Tone-Mapping-Keras/master/image/hdr_show.png)
In this work, we have proposed a new tone mapping method--a novel reformulated Laplacian method to 
```decompose a WDR image into a low-resolution image``` (contains the low frequency component of the WDR image) ```and a high-resolution image```(contains the remaining higher frequencies of the WDR image).


The two images are processed by a ```global compression network```(compress the global scale gradient of a WDR image) and a ```local manipulation neural network```(manipulate local features), respectively. The generated images from the two networks are further merged together, and then input merged images to ```fine tune network``` to produce the final image.

## Get Started
### Prerequistes
* Python 2.7
* Tensorflow-gpu 1.9.0
* Keras 2.2.4
* Opencv-python 3.4.4.19
* Scipy 1.1.0
* Matplotlib 2.2.3
* [Luminance HDR](https://github.com/luminancehdr/luminancehdr) for tonemapping
* [Laval Indoor dataset](http://indoor.hdrdb.com/)(EULA required)
### Instructions
#### Demo
In Pycharm, set `test_flag` to `True` in `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/ft_layer.py` file, then run this file.
#### Train
* train high layer

  >run `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/high_layer.py` file.
* train bottom layer

  >run `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/bot_layer.py` file.
* train fine tune layer


  >run `/Deep-Reformulated-Laplacian-Tone-Mapping-Master/operation/ft_layer.py` file.


## Contact
This algorithm is developed by [I2Sense Lab](https://www.ucalgary.ca/i2sense/) of [University of Calgary](https://www.ucalgary.ca/).

If you have ang issue,please concat [Prof. Yadid-Pecht](https://www.ucalgary.ca/i2sense/yadid_pecht_biography) or [Dr.Jie Yang](https://jieyang1987.github.io/).
#### Email:
 >orly.yadid-pecht@ucalgary.ca
 
 >yangjie@westlake.edu.cn
 
 >mengchen.lin@ucalgary.ca
 
 >ziyi.liu1@ucalgary.ca

