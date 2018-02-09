
# coding: utf-8

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
from image_augmentation import random_warp
from utils import get_image_paths, load_images, stack_images
from pixel_shuffler import PixelShuffler


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


# <a id='4'></a>
# # 4. Config
# 
# mixup paper: https://arxiv.org/abs/1710.09412
# 
# Default training data directories: `./faceA/` and `./faceB/`

# In[4]:


K.set_learning_phase(1)


# In[5]:


channel_axis=-1
channel_first = False


# In[6]:


IMAGE_SHAPE = (64, 64, 3)
nc_in = 3 # number of input channels of generators
nc_D_inp = 6 # number of input channels of discriminators

use_perceptual_loss = False
use_lsgan = True
use_instancenorm = False
use_mixup = True
mixup_alpha = 0.2 # 0.2

batchSize = 32
lrD = 1e-4 # Discriminator learning rate
lrG = 1e-4 # Generator learning rate

# Path of training images
img_dirA = './faceA/*.*'
img_dirB = './faceB/*.*'


# <a id='5'></a>
# # 5. Define models

# In[7]:


from model_GAN_v2 import *


# In[8]:


encoder = Encoder()
decoder_A = Decoder_ps()
decoder_B = Decoder_ps()

x = Input(shape=IMAGE_SHAPE)

netGA = Model(x, decoder_A(encoder(x)))
netGB = Model(x, decoder_B(encoder(x)))


# In[9]:


netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)


# <a id='6'></a>
# # 6. Load Models

# In[10]:


try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
    #netDA.load_weights("models/netDA.h5") 
    #netDB.load_weights("models/netDB.h5") 
    print ("model loaded.")
except:
    print ("Weights file not found.")
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

# In[11]:


def cycle_variables(netG):
    distorted_input = netG.inputs[0]
    fake_output = netG.outputs[0]
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    masked_fake_output = alpha * rgb + (1-alpha) * distorted_input 

    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
    return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr


# In[12]:


distorted_A, fake_A, mask_A, path_A, path_mask_A, path_abgr_A = cycle_variables(netGA)
distorted_B, fake_B, mask_B, path_B, path_mask_B, path_abgr_B = cycle_variables(netGB)
real_A = Input(shape=IMAGE_SHAPE)
real_B = Input(shape=IMAGE_SHAPE)


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

# In[33]:


def swap_face(img, path_func):
    input_size = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # generator expects BGR input    
    ae_input = cv2.resize(img, (64,64))/255. * 2 - 1        
    
    result = np.squeeze(np.array([path_func([[ae_input]])]))
    result_a = result[:,:,0] * 255
    result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
    result_a = np.expand_dims(result_a, axis=2)
    result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
    
    #result = np.clip( (result + 1) * 255 / 2, 0, 255 ).astype('uint8')   
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    result = cv2.resize(result, (input_size[1],input_size[0]))
    result_a = np.expand_dims(cv2.resize(result_a, (input_size[1],input_size[0])), axis=2)
    return result, result_a


# In[34]:


whom2whom = "BtoA" # default trainsforming faceB to faceA

if whom2whom is "AtoB":
    path_func = path_abgr_B
elif whom2whom is "BtoA":
    path_func = path_abgr_A
else:
    print ("whom2whom should be either AtoB or BtoA")


# In[35]:


input_img = plt.imread("./IMAGE_FILENAME.jpg")


# In[ ]:


plt.imshow(input_img)


# In[37]:


result_img, result_mask = swap_face(input_img, path_func)


# In[ ]:


plt.imshow(result_img)


# In[ ]:


plt.imshow(result_mask[:, :, 0]) # cmap='gray'

