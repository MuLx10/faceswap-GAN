
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


# <a id='2'></a>
# # 2. Install requirements
# 
# ## ========== CAUTION ========== 
# 
# If you are running this jupyter on local machine. Please read [this blog](http://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/) before running the following cells which pip install packages.

# In[ ]:


# https://github.com/ageitgey/face_recognition
#!pip install face_recognition


# In[ ]:


#!pip install moviepy


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

# In[9]:


from model_GAN_v2 import *


# In[10]:


encoder = Encoder()
decoder_A = Decoder_ps()
decoder_B = Decoder_ps()

x = Input(shape=IMAGE_SHAPE)

netGA = Model(x, decoder_A(encoder(x)))
netGB = Model(x, decoder_B(encoder(x)))


# In[11]:


netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)


# <a id='6'></a>
# # 6. Load Models

# In[12]:


try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
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
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    masked_fake_output = alpha * rgb + (1-alpha) * distorted_input 

    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
    return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr


# In[14]:


distorted_A, fake_A, mask_A, path_A, path_mask_A, path_abgr_A = cycle_variables(netGA)
distorted_B, fake_B, mask_B, path_B, path_mask_B, path_abgr_B = cycle_variables(netGB)
real_A = Input(shape=IMAGE_SHAPE)
real_B = Input(shape=IMAGE_SHAPE)


# <a id='12'></a>
# # 12. Make video clips
# 
# Given a video as input, the following cells will detect face for each frame using dlib's cnn model. And use trained GAN model to transform detected face into target face. Then output a video with swapped faces.

# In[15]:


# Download ffmpeg if need, which is required by moviepy.

#import imageio
#imageio.plugins.ffmpeg.download()


# In[17]:


import face_recognition
from moviepy.editor import VideoFileClip


# # Define transform path: A2B or B2A

# In[20]:


whom2whom = "BtoA" # default trainsforming faceB to faceA

if whom2whom is "AtoB":
    path_func = path_abgr_B
elif whom2whom is "BtoA":
    path_func = path_abgr_A
else:
    print ("whom2whom should be either AtoB or BtoA")


# <a id='13'></a>
# # 13. Make video clips w/o face alignment
# 
# ### Default transform: face B to face A

# In[21]:


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
    
def generate_face(ae_input, path_abgr, roi_size):
    result = np.squeeze(np.array([path_abgr([[ae_input]])]))
    result_a = result[:,:,0] * 255
    result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
    result_a = cv2.GaussianBlur(result_a ,(7,7),6)
    result_a = np.expand_dims(result_a, axis=2)
    result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = cv2.resize(result, (roi_size[1],roi_size[0]))
    result_a = np.expand_dims(cv2.resize(result_a, (roi_size[1],roi_size[0])), axis=2)
    return result, result_a

def get_init_mask_map(image):
    return np.zeros_like(image)

def get_init_comb_img(input_img):
    comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
    comb_img[:, :input_img.shape[1], :] = input_img
    comb_img[:, input_img.shape[1]:, :] = input_img
    return comb_img    

def get_init_triple_img(input_img, no_face=False):
    if no_face:
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1], :] = input_img
        triple_img[:, input_img.shape[1]:input_img.shape[1]*2, :] = input_img      
        triple_img[:, input_img.shape[1]*2:, :] = (input_img * .15).astype('uint8')  
        return triple_img
    else:
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        return triple_img

def get_mask(roi_image, h, w):
    mask = np.zeros_like(roi_image)
    mask[h//15:-h//15,w//15:-w//15,:] = 255
    mask = cv2.GaussianBlur(mask,(15,15),10)
    return mask

def process_video(input_img):   
    # modify this line to reduce input size
    #input_img = input_img[:, input_img.shape[1]//3:2*input_img.shape[1]//3,:] 
    image = input_img
    faces = get_faces_bbox(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = get_init_comb_img(input_img)
        triple_img = get_init_triple_img(input_img, no_face=True)
        
    mask_map = get_init_mask_map(image)
    comb_img = get_init_comb_img(input_img)
    global prev_x0, prev_x1, prev_y0, prev_y1
    global frames    
    for (x0, y1, x1, y0) in faces:        
        # smoothing bounding box
        if use_smoothed_bbox:
            if frames != 0:
                x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1)
                set_global_coord(x0, x1, y0, y1)
            else:
                set_global_coord(x0, x1, y0, y1)
                frames += 1
        h = x1 - x0
        w = y1 - y0
            
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        roi_image = cv2_img[x0+h//15:x1-h//15,y0+w//15:y1-w//15,:]
        roi_size = roi_image.shape  
        
        ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1        
        result, result_a = generate_face(ae_input, path_abgr_A, roi_size)
        mask_map[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = result_a
        mask_map = np.clip(mask_map + .15 * input_img, 0, 255 )     
        
        if use_smoothed_mask:
            mask = get_mask(roi_image, h, w)
            roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            smoothed_result = mask/255 * result + (1-mask/255) * roi_rgb
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = smoothed_result
        else:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = result
            
        triple_img = get_init_triple_img(input_img)
        triple_img[:, :input_img.shape[1]*2, :] = comb_img
        triple_img[:, input_img.shape[1]*2:, :] = mask_map
    
    # ========== Change the following line for different output type==========
    return comb_img[:, input_img.shape[1]:, :]  # return only result image
    # return comb_img  # return input and result image combined as one
    #return triple_img #return input,result and mask heatmap image combined as one


# **Description**
# ```python
#     video_scaling_offset = 0 # Increase by 1 if OOM happens.
#     manually_downscale = False # Set True if increasing offset doesn't help
#     manually_downscale_factor = int(2) # Increase by 1 if OOM still happens.
# ```

# In[23]:


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

# In[103]:


import gc
gc.collect()

