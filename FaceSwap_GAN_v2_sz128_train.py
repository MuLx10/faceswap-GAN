
# coding: utf-8

# # Usage
# 
# **To train a model**: Run 1 ~ 10.
# 
# **To load model weights**: Run 1 and 4 ~ 7.
# 
# **To use trained model to swap a single face image**: Run "**To load model weights**" and 11.
# 
# **To use trained model to create a video clips**: Run "**To load model weights**", 12 and 13 (or 14).
# 
# 
# ## Index
# 1. [Import Packages](#1)
# 2. [Install Requirements (optional)](#2)
# 3. [Import VGGFace (optional)](#3)
# 4. [Config](#4)
# 5. [Define Models](#5)
# 6. [Load Models](#6)
# 7. [Define Inputs/outputs Variables](#7)
# 8. [Define Loss Function](#8)
# 9. [Utils for loading/displaying images](#9)
# 10. [Start Training](#10)
# 11. [Helper Function: face_swap()](#11)
# 12. [Import Packages for Making Video Clips](#12)
# 13. [Make Video Clips w/o Face Alignment](#13)
# 14. [Make video clips w/ face alignment](#14)

# <a id='1'></a>
# # 1. Import packages

# In[1]:


from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from tensorflow.contrib.distributions import Beta
import tensorflow as tf
from keras.optimizers import Adam


# In[2]:


from image_augmentation import random_transform
from image_augmentation import random_warp, random_warp128
from utils import get_image_paths, load_images, stack_images
from pixel_shuffler import PixelShuffler
from instance_normalization import InstanceNormalization


# In[3]:


import time
import numpy as np
from PIL import Image
import cv2
import glob
from random import randint, shuffle
from IPython.display import clear_output
from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='2'></a>
# # 2. Install requirements
# 
# ## ========== CAUTION ========== 
# 
# If you are running this jupyter on local machine. Please read [this blog](http://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/) before running the following cells which pip install packages.

# In[ ]:


# https://github.com/rcmalli/keras-vggface
# Skip this cell if you don't want to use perceptual loss
get_ipython().system('pip install keras_vggface')


# We only import ```face_recognition``` and ```moviepy``` when making videos. They are not necessary in training GAN models.

# In[ ]:


# https://github.com/ageitgey/face_recognition
get_ipython().system('pip install face_recognition')


# In[ ]:


get_ipython().system('pip install moviepy')


# <a id='3'></a>
# # 3. Import VGGFace
# (Skip this part if you don't want to apply perceptual loss)

# If you got error ```_obtain_input_shape(...)``` error, this is because your keras version is older than vggface requirement. 
# 
# Modify ```_obtain_input_shape(...)``` in ```keras_vggface/models.py``` will solve the problem. The following is what worked for me:
# 
# ```python
# input_shape = _obtain_input_shape(input_shape,
#                                   default_size=224,
#                                   min_size=197,
#                                   data_format=K.image_data_format(),
#                                   include_top=include_top)
# ```

# In[4]:


from keras_vggface.vggface import VGGFace


# In[5]:


vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))


# In[ ]:


vggface.summary()


# <a id='4'></a>
# # 4. Config
# 
# mixup paper: https://arxiv.org/abs/1710.09412
# 
# Default training data directories: `./faceA/` and `./faceB/`

# In[6]:


K.set_learning_phase(1)


# In[7]:


channel_axis=-1
channel_first = False


# In[8]:


IMAGE_SHAPE = (128, 128, 3)
nc_in = 3 # number of input channels of generators
nc_D_inp = 6 # number of input channels of discriminators

use_perceptual_loss = True
use_lsgan = True
use_instancenorm = False
use_mixup = True
mixup_alpha = 0.2 # 0.2

batchSize = 8
lrD = 1e-4 # Discriminator learning rate
lrG = 1e-4 # Generator learning rate

# Path of training images
img_dirA = './faceA/*.*'
img_dirB = './faceB/*.*'


# <a id='5'></a>
# # 5. Define models

# In[9]:


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization


# In[10]:


#def batchnorm():
#    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5, gamma_initializer = gamma_init)

def inst_norm():
    return InstanceNormalization()

def conv_block(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = SeparableConv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    if use_instance_norm:
        x = inst_norm()(x)
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    if use_instance_norm:
        x = inst_norm()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def res_block(input_tensor, f, dilation=1):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", dilation_rate=dilation)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", dilation_rate=dilation)(x)
    x = add([x, input_tensor])
    #x = LeakyReLU(alpha=0.2)(x)
    return x

def upscale_ps(filters, use_instance_norm=True):
    def block(x, use_instance_norm=use_instance_norm):
        x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
        if use_instance_norm:
            x = inst_norm()(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Discriminator(nc_in, input_size=128):
    inp = Input(shape=(input_size, input_size, nc_in))
    #x = GaussianNoise(0.05)(inp)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, True)
    x = conv_block_d(x, 256, True)
    x = conv_block_d(x, 512, True)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)   
    return Model(inputs=[inp], outputs=out)

def Encoder(nc_in=3, input_size=128):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(32, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = conv_block(x,64, use_instance_norm=False)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512) 
    x = conv_block(x,1024)
    x = Dense(1024)(Flatten()(x))
    x = Dense(4*4*1024)(x)
    x = Reshape((4, 4, 1024))(x)
    out = upscale_ps(512)(x)
    return Model(inputs=inp, outputs=out)

def Decoder_ps(nc_in=512, input_size=8):
    input_ = Input(shape=(input_size, input_size, nc_in))
    x = input_
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)    
    x = upscale_ps(64)(x)
    x = res_block(x, 64, dilation=2)      
    
    out64 = Conv2D(64, kernel_size=3, padding='same')(x)
    out64 = LeakyReLU(alpha=0.1)(out64)
    out64 = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(out64)
    
    x = upscale_ps(32)(x)
    x = res_block(x, 32)
    x = res_block(x, 32)
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, rgb])
    return Model(input_, [out, out64] )    


# In[11]:


encoder = Encoder()
decoder_A = Decoder_ps()
decoder_B = Decoder_ps()

x = Input(shape=IMAGE_SHAPE)

netGA = Model(x, decoder_A(encoder(x)))
netGB = Model(x, decoder_B(encoder(x)))


# In[12]:


netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)


# <a id='6'></a>
# # 6. Load Models

# In[88]:


try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
    netDA.load_weights("models/netDA.h5") 
    netDB.load_weights("models/netDB.h5") 
    print ("Model weights files are successfully loaded")
except:
    print ("Error occurs during loading weights files.")
    pass


# <a id='7'></a>
# # 7. Define Inputs/Outputs Variables
# 
#     distorted_A: A (batch_size, 64, 64, 3) tensor, input of generator_A (netGA).
#     distorted_B: A (batch_size, 64, 64, 3) tensor, input of generator_B (netGB).
#     fake_A: (batch_size, 64, 64, 3) tensor, output of generator_A (netGA).
#     fake_B: (batch_size, 64, 64, 3) tensor, output of generator_B (netGB).
#     mask_A: (batch_size, 64, 64, 1) tensor, mask output of generator_A (netGA).
#     mask_B: (batch_size, 64, 64, 1) tensor, mask output of generator_B (netGB).
#     path_A: A function that takes distorted_A as input and outputs fake_A.
#     path_B: A function that takes distorted_B as input and outputs fake_B.
#     path_mask_A: A function that takes distorted_A as input and outputs mask_A.
#     path_mask_B: A function that takes distorted_B as input and outputs mask_B.
#     path_abgr_A: A function that takes distorted_A as input and outputs concat([mask_A, fake_A]).
#     path_abgr_B: A function that takes distorted_B as input and outputs concat([mask_B, fake_B]).
#     real_A: A (batch_size, 64, 64, 3) tensor, target images for generator_A given input distorted_A.
#     real_B: A (batch_size, 64, 64, 3) tensor, target images for generator_B given input distorted_B.

# In[13]:


def cycle_variables(netG):
    distorted_input = netG.inputs[0]
    fake_output = netG.outputs[0]
    fake_output64 = netG.outputs[1]
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    masked_fake_output = alpha * rgb + (1-alpha) * distorted_input 

    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
    fn_bgr = K.function([distorted_input], [rgb])
    return distorted_input, fake_output, fake_output64, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr


# In[14]:


distorted_A, fake_A, fake_sz64_A, mask_A, path_A, path_mask_A, path_abgr_A, path_bgr_A = cycle_variables(netGA)
distorted_B, fake_B, fake_sz64_B, mask_B, path_B, path_mask_B, path_abgr_B, path_bgr_B = cycle_variables(netGB)
real_A = Input(shape=IMAGE_SHAPE)
real_B = Input(shape=IMAGE_SHAPE)


# <a id='8'></a>
# # 8. Define Loss Function
# 
# LSGAN

# In[15]:


if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))


# In[16]:


def define_loss(netD, real, fake_argb, fake_sz64, distorted, vggface_feat=None):   
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_argb)
    fake_rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_argb)
    fake = alpha * fake_rgb + (1-alpha) * distorted
    
    if use_mixup:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        # ==========
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        # ==========
        output_mixup = netD(mixup)
        loss_D = loss_fn(output_mixup, lam * K.ones_like(output_mixup)) 
        #output_fake = netD(concatenate([fake, distorted])) # dummy
        loss_G = 1 * loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
    else:
        output_real = netD(concatenate([real, distorted])) # positive sample
        output_fake = netD(concatenate([fake, distorted])) # negative sample   
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))    
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))   
        loss_D = loss_D_real + loss_D_fake
        loss_G = 1 * loss_fn(output_fake, K.ones_like(output_fake))  
    # ==========  
    loss_G += K.mean(K.abs(fake_rgb - real))
    loss_G += K.mean(K.abs(fake_sz64 - tf.image.resize_images(real, [64, 64])))
    # ==========
    
    # Perceptual Loss
    if not vggface_feat is None:
        def preprocess_vggface(x):
            x = (x + 1)/2 * 255 # channel order: BGR
            x -= [93.5940, 104.7624, 129.]
            return x
        pl_params = (0.02, 0.3, 0.5)
        real_sz224 = tf.image.resize_images(real, [224, 224])
        real_sz224 = Lambda(preprocess_vggface)(real_sz224)
        # ==========
        fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224]) 
        fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
        # ==========   
        real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
        fake_feat55, fake_feat28, fake_feat7  = vggface_feat(fake_sz224)    
        loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
        loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
        loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))
    
    return loss_D, loss_G


# In[17]:


# ========== Define Perceptual Loss Model==========
if use_perceptual_loss:
    vggface.trainable = False
    out_size55 = vggface.layers[36].output
    out_size28 = vggface.layers[78].output
    out_size7 = vggface.layers[-2].output
    vggface_feat = Model(vggface.input, [out_size55, out_size28, out_size7])
    vggface_feat.trainable = False
else:
    vggface_feat = None


# In[19]:


loss_DA, loss_GA = define_loss(netDA, real_A, fake_A, fake_sz64_A, distorted_A, vggface_feat)
loss_DB, loss_GB = define_loss(netDB, real_B, fake_B, fake_sz64_B, distorted_B, vggface_feat)

loss_GA += 3e-3 * K.mean(K.abs(mask_A))
loss_GB += 3e-3 * K.mean(K.abs(mask_B))


# In[20]:


weightsDA = netDA.trainable_weights
weightsGA = netGA.trainable_weights
weightsDB = netDB.trainable_weights
weightsGB = netGB.trainable_weights

# Adam(..).get_updates(...)
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
netDA_train = K.function([distorted_A, real_A],[loss_DA], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
netDB_train = K.function([distorted_B, real_B],[loss_DB], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)


# <a id='9'></a>
# # 9. Utils For Loading/Displaying Images

# In[21]:


def load_data(file_pattern):
    return glob.glob(file_pattern)

random_transform_args = {
    'rotation_range': 20,
    'zoom_range': 0.1,
    'shift_range': 0.05,
    'random_flip': 0.5,
    }
def read_image(fn, random_transform_args=random_transform_args):
    image = cv2.imread(fn)
    image = cv2.resize(image, (256,256)) / 255 * 2 - 1
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp128(image)
    
    return warped_img, target_img


# In[22]:


# A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None  
    shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1        
        rtn = np.float32([read_image(data[j]) for j in range(i,i+size)])
        i+=size
        tmpsize = yield epoch, rtn[:,0,:,:,:], rtn[:,1,:,:,:]       

def minibatchAB(dataA, batchsize):
    batchA = minibatch(dataA, batchsize)
    tmpsize = None    
    while True:        
        ep1, warped_img, target_img = batchA.send(tmpsize)
        tmpsize = yield ep1, warped_img, target_img


# In[23]:


def showG(test_A, test_B, path_A, path_B):
    figure_A = np.stack([
        test_A,
        np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
        np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
        ], axis=1 )
    figure_B = np.stack([
        test_B,
        np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
        np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
        ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0 )
    figure = figure.reshape((4,7) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))
    
def showG_mask(test_A, test_B, path_A, path_B):
    figure_A = np.stack([
        test_A,
        (np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
        (np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
        ], axis=1 )
    figure_B = np.stack([
        test_B,
        (np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
        (np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
        ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0 )
    figure = figure.reshape((4,7) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))


# <a id='10'></a>
# # 10. Start Training
# 
# Show results and save model weights every `display_iters` iterations.

# In[24]:


get_ipython().system('mkdir models # create ./models directory')


# In[ ]:


# Get filenames
train_A = load_data(img_dirA)
train_B = load_data(img_dirB)

assert len(train_A), "No image found in " + str(img_dirA)
assert len(train_B), "No image found in " + str(img_dirB)


# In[ ]:


t0 = time.time()
niter = 150
gen_iterations = 0
epoch = 0
errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

display_iters = 300
train_batchA = minibatchAB(train_A, batchSize)
train_batchB = minibatchAB(train_B, batchSize)

# ========== Change 10000 to desired iterations  ========== 
while gen_iterations < 50000: 
    epoch, warped_A, target_A = next(train_batchA) 
    epoch, warped_B, target_B = next(train_batchB) 
    
    # Train dicriminators for one batch
    if gen_iterations % 1 == 0:
        errDA  = netDA_train([warped_A, target_A])
        errDB  = netDB_train([warped_B, target_B])
    errDA_sum +=errDA[0]
    errDB_sum +=errDB[0]
    
    if gen_iterations == 5:
        print ("working.")

    # Train generators for one batch
    errGA = netGA_train([warped_A, target_A])
    errGB = netGB_train([warped_B, target_B])
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    gen_iterations+=1
    
    if gen_iterations % display_iters == 0 or gen_iterations == 50:
        if gen_iterations % (display_iters) == 0: # clear_output every display_iters iters
            clear_output()
        print('[%d/%d][%d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
        % (epoch, niter, gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, time.time()-t0))   
        
        # get new batch of images and generate results for visualization
        _, wA, tA = train_batchA.send(14)  
        _, wB, tB = train_batchB.send(14)
        showG(tA, tB, path_A, path_B)   
        showG(wA, wB, path_bgr_A, path_bgr_B)         
        showG_mask(tA, tB, path_mask_A, path_mask_B)           
        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        
        # Save models
        encoder.save_weights("models/encoder.h5")
        decoder_A.save_weights("models/decoder_A.h5" )
        decoder_B.save_weights("models/decoder_B.h5" )
        netDA.save_weights("models/netDA.h5")
        netDB.save_weights("models/netDB.h5")


# ## Tips for mask refinement (optional after >15k iters) 
# 
# In [Define loss function](#8), change
# ```python
# def define_loss(...)
# ...
# mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
# loss_G += K.mean(K.abs(fake_rgb - real))
# fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224])
# ...
# # cell below
# loss_GA += 3e-3 * K.mean(K.abs(mask_A))
# loss_GB += 3e-3 * K.mean(K.abs(mask_B))
# ```
# to
# ```python
# def define_loss(...)
# ...
# mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
# loss_G += K.mean(K.abs(fake - real))
# fake_sz224 = tf.image.resize_images(fake, [224, 224])
# ...
# # cell below
# loss_GA += 1e-3 * K.mean(K.square(mask_A))
# loss_GB += 1e-3 * K.mean(K.square(mask_B))
# ```
# If this gives better mask generation, then keep training.

# <a id='11'></a>
# # 11. Helper Function: face_swap()
# This function is provided for those who don't have enough VRAM to run dlib's CNN and GAN model at the same time.
# 
#     INPUTS:
#         img: A RGB face image of any size.
#         path_func: a function that is either path_abgr_A or path_abgr_B.
#     OUPUTS:
#         result_img: A RGB swapped face image after masking.
#         result_mask: A single channel uint8 mask image.

# In[21]:


def swap_face(img, path_func):
    input_size = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # generator expects BGR input    
    ae_input = cv2.resize(img, (128,128))/255. * 2 - 1        
    
    result = np.squeeze(np.array([path_func([[ae_input]])]))
    result_a = result[:,:,0] * 255
    result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
    result_a = np.expand_dims(result_a, axis=2)
    result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
       
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    result = cv2.resize(result, (input_size[1],input_size[0]))
    result_a = np.expand_dims(cv2.resize(result_a, (input_size[1],input_size[0])), axis=2)
    return result, result_a


# In[22]:


whom2whom = "BtoA" # default trainsforming faceB to faceA

if whom2whom is "AtoB":
    path_func = path_abgr_B
elif whom2whom is "BtoA":
    path_func = path_abgr_A
else:
    print ("whom2whom should be either AtoB or BtoA")


# In[23]:


input_img = plt.imread("./sh_face_img.JPG")


# In[ ]:


plt.imshow(input_img)


# In[25]:


result_img, result_mask = swap_face(input_img, path_func)


# In[ ]:


plt.imshow(result_img)


# In[ ]:


plt.imshow(result_mask[:, :, 0])


# <a id='12'></a>
# # 12. Make video clips
# 
# Given a video as input, the following cells will detect face for each frame using dlib's cnn model. And use trained GAN model to transform detected face into target face. Then output a video with swapped faces.

# In[60]:


# Download ffmpeg if need, which is required by moviepy.

#import imageio
#imageio.plugins.ffmpeg.download()


# In[61]:


import face_recognition
from moviepy.editor import VideoFileClip


# <a id='13'></a>
# # 13. Make video clips w/o face alignment
# 
# ### Default transform: face B to face A

# In[82]:


use_smoothed_mask = True
use_smoothed_bbox = True

def is_higher_than_480p(x):
    return (x.shape[0] * x.shape[1]) >= (858*480)

def is_higher_than_720p(x):
    return (x.shape[0] * x.shape[1]) >= (1280*720)

def is_higher_than_1080p(x):
    return (x.shape[0] * x.shape[1]) >= (1920*1080)

def calibrate_coord(faces, video_scaling_factor):
    for i, (x0, y1, x1, y0) in enumerate(faces):
        faces[i] = (x0*video_scaling_factor, y1*video_scaling_factor, 
                    x1*video_scaling_factor, y0*video_scaling_factor)
    return faces

def get_faces_bbox(image, model="cnn"):  
    if is_higher_than_1080p(image):
        video_scaling_factor = 4 + video_scaling_offset
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)
    elif is_higher_than_720p(image):
        video_scaling_factor = 3 + video_scaling_offset
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)  
    elif is_higher_than_480p(image):
        video_scaling_factor = 2 + video_scaling_offset
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)
    elif manually_downscale:
        video_scaling_factor = manually_downscale_factor
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)
    else:
        faces = face_recognition.face_locations(image, model=model)
    return faces

def get_smoothed_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    x0 = int(0.65*prev_x0 + 0.35*x0)
    x1 = int(0.65*prev_x1 + 0.35*x1)
    y1 = int(0.65*prev_y1 + 0.35*y1)
    y0 = int(0.65*prev_y0 + 0.35*y0)
    return x0, x1, y0, y1    
    
def set_global_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    prev_x0 = x0
    prev_x1 = x1
    prev_y1 = y1
    prev_y0 = y0

def process_video(input_img):   
    # modify this line to reduce input size
    #input_img = input_img[:, input_img.shape[1]//3:2*input_img.shape[1]//3,:] 
    image = input_img
    faces = get_faces_bbox(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1], :] = input_img
        triple_img[:, input_img.shape[1]:input_img.shape[1]*2, :] = input_img      
        triple_img[:, input_img.shape[1]*2:, :] = (input_img * .15).astype('uint8')
    
    mask_map = np.zeros_like(image)
    
    global prev_x0, prev_x1, prev_y0, prev_y1
    global frames    
    for (x0, y1, x1, y0) in faces:
        h = x1 - x0
        w = y1 - y0
        
        # smoothing bounding box
        if use_smoothed_bbox:
            if frames != 0:
                x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1)
                set_global_coord(x0, x1, y0, y1)
            else:
                set_global_coord(x0, x1, y0, y1)
                frames += 1
            
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        roi_image = cv2_img[x0+h//15:x1-h//15,y0+w//15:y1-w//15,:]
        roi_size = roi_image.shape  
        
        # smoothing mask
        if use_smoothed_mask:
            mask = np.zeros_like(roi_image)
            mask[h//15:-h//15,w//15:-w//15,:] = 255
            mask = cv2.GaussianBlur(mask,(15,15),10)
            orig_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        ae_input = cv2.resize(roi_image, (128,128))/255. * 2 - 1        
        result = np.squeeze(np.array([path_abgr_A([[ae_input]])])) # Change path_A/path_B here
        result_a = result[:,:,0] * 255
        #result_a = np.clip(result_a * 1.5, 0, 255).astype('uint8')
        result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
        result_a = cv2.GaussianBlur(result_a ,(7,7),6)
        result_a = np.expand_dims(result_a, axis=2)
        result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        mask_map[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = np.expand_dims(cv2.resize(result_a, (roi_size[1],roi_size[0])), axis=2)
        mask_map = np.clip(mask_map + .15 * input_img, 0, 255 )
        
        result = cv2.resize(result, (roi_size[1],roi_size[0]))
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        
        if use_smoothed_mask:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = mask/255*result + (1-mask/255)*orig_img
        else:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = result
            
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1]*2, :] = comb_img
        triple_img[:, input_img.shape[1]*2:, :] = mask_map
    
    # ========== Change the following line for different output type==========
    # return comb_img[:, input_img.shape[1]:, :]  # return only result image
    # return comb_img  # return input and result image combined as one
    return triple_img #return input,result and mask heatmap image combined as one


# **Description**
# ```python
#     video_scaling_offset = 0 # Increase by 1 if OOM happens.
#     manually_downscale = False # Set True if increasing offset doesn't help
#     manually_downscale_factor = int(2) # Increase by 1 if OOM still happens.
# ```

# In[85]:


# Variables for smoothing bounding box
global prev_x0, prev_x1, prev_y0, prev_y1
global frames
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0
video_scaling_offset = 0 
manually_downscale = False
manually_downscale_factor = int(2) # should be an positive integer

output = 'OUTPUT_VIDEO.mp4'
clip1 = VideoFileClip("INPUT_VIDEO.mp4")
clip = clip1.fl_image(process_video)#.subclip(11, 13) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')


# ### gc.collect() sometimes solves memory error

# In[111]:


import gc
gc.collect()

