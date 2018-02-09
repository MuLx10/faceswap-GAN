
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
# # 1. Import Packages

# In[ ]:


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


# In[ ]:


from image_augmentation import random_transform
from image_augmentation import random_warp
from utils import get_image_paths, load_images, stack_images
from pixel_shuffler import PixelShuffler


# In[4]:


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
# # 2. Install Requirements (optional)
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
# # 3. Import VGGFace (optional)
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

# In[5]:


from keras_vggface.vggface import VGGFace


# In[ ]:


vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))


# In[ ]:


vggface.summary()


# <a id='4'></a>
# # 4. Config
# 
# mixup paper: https://arxiv.org/abs/1710.09412
# 
# Default training data directories: `./faceA/` and `./faceB/`
# 
# Default `use_perceptual_loss = False`

# In[4]:


K.set_learning_phase(1)


# In[5]:


channel_axis=-1
channel_first = False


# In[14]:


IMAGE_SHAPE = (64, 64, 3)
nc_in = 3 # number of input channels of generators
nc_D_inp = 3 # number of input channels of discriminators

use_perceptual_loss = False
use_lsgan = True
use_instancenorm = False
use_mixup = True
mixup_alpha = 0.1 # 0.2

batchSize = 32
lrD = 1e-4 # Discriminator learning rate
lrG = 1e-4 # Generator learning rate

# Path of training images
img_dirA = './faceA/*.*'
img_dirB = './faceB/*.*'


# <a id='5'></a>
# # 5. Define Models

# In[15]:


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization


# In[16]:


#def batchnorm():
#    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5, gamma_initializer = gamma_init)

def conv_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def res_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    return x

# Legacy
#def upscale_block(input_tensor, f):
#    x = input_tensor
#    x = Conv2DTranspose(f, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(x) 
#    x = LeakyReLU(alpha=0.2)(x)
#    return x

def upscale_ps(filters, use_norm=True):
    def block(x):
        x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Discriminator(nc_in, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    #x = GaussianNoise(0.05)(inp)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, False)
    x = conv_block_d(x, 256, False)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)   
    return Model(inputs=[inp], outputs=out)

def Encoder(nc_in=3, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512) 
    x = conv_block(x,1024)
    x = Dense(1024)(Flatten()(x))
    x = Dense(4*4*1024)(x)
    x = Reshape((4, 4, 1024))(x)
    out = upscale_ps(512)(x)
    return Model(inputs=inp, outputs=out)

# Legacy, left for someone to try if interested
#def Decoder(nc_in=512, input_size=8):
#    inp = Input(shape=(input_size, input_size, nc_in))   
#    x = upscale_block(inp, 256)
#    x = Cropping2D(((0,1),(0,1)))(x)
#    x = upscale_block(x, 128)
#    x = res_block(x, 128)
#    x = Cropping2D(((0,1),(0,1)))(x)
#    x = upscale_block(x, 64)
#    x = res_block(x, 64)
#    x = res_block(x, 64)
#    x = Cropping2D(((0,1),(0,1)))(x)
#    x = Conv2D(3, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
#    out = Activation("tanh")(x)
#    return Model(inputs=inp, outputs=out)

def Decoder_ps(nc_in=512, input_size=8):
    input_ = Input(shape=(input_size, input_size, nc_in))
    x = input_
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh' )(x)
    return Model(input_, x )    


# In[17]:


encoder = Encoder()
decoder_A = Decoder_ps()
decoder_B = Decoder_ps()

x = Input(shape=IMAGE_SHAPE)

netGA = Model(x, decoder_A(encoder(x)))
netGB = Model(x, decoder_B(encoder(x)))


# In[ ]:


encoder.summary()


# In[ ]:


decoder_A.summary()


# In[20]:


netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)


# <a id='6'></a>
# # 6. Load Models

# In[21]:


try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
    #netDA.load_weights("models/netDA.h5") # uncomment these if you want to continue training from last checkpoint
    #netDB.load_weights("models/netDB.h5")
    print ("model loaded.")
except:
    print ("Weights file not found.")
    pass


# <a id='7'></a>
# # 7. Define Inputs/outputs Variables
# 
#     distorted_A: A (batch_size, 64, 64, 3) tensor, input of generator_A (netGA).
#     distorted_B: A (batch_size, 64, 64, 3) tensor, input of generator_B (netGB).
#     fake_A: (batch_size, 64, 64, 3) tensor, output of generator_A (netGA).
#     fake_B: (batch_size, 64, 64, 3) tensor, output of generator_B (netGB).
#     path_A: A function that takes distorted_A as input and outputs fake_A.
#     path_B: A function that takes distorted_B as input and outputs fake_B.
#     real_A: A (batch_size, 64, 64, 3) tensor, target images for generator_A given input distorted_A.
#     real_B: A (batch_size, 64, 64, 3) tensor, target images for generator_B given input distorted_B.

# In[22]:


def cycle_variables(netG):
    distorted_input = netG.inputs[0]
    fake_output = netG.outputs[0]    
    fn_generate = K.function([distorted_input], [fake_output])
    return distorted_input, fake_output, fn_generate


# In[23]:


distorted_A, fake_A, path_A = cycle_variables(netGA)
distorted_B, fake_B, path_B = cycle_variables(netGB)
real_A = Input(shape=IMAGE_SHAPE)
real_B = Input(shape=IMAGE_SHAPE)


# <a id='8'></a>
# # 8. Define Loss Function
# 
# Use LSGAN

# In[24]:


if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))


# In[25]:


def define_loss(netD, real, fake, vggface_feat=None):    
    if use_mixup:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        mixup = lam * real + (1 - lam) * fake
        output_mixup = netD(mixup)
        loss_D = loss_fn(output_mixup, lam * K.ones_like(output_mixup)) 
        output_fake = netD(fake) # dummy
        loss_G = .5 * loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
    else:
        output_real = netD(real) # positive sample
        output_fake = netD(fake) # negative sample   
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))    
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))   
        loss_D = loss_D_real + loss_D_fake
        loss_G = .5 * loss_fn(output_fake, K.ones_like(output_fake))    
    loss_G += K.mean(K.abs(fake - real))
    
    # ========== Perceptual Loss ==========
    if not vggface_feat is None:
        pl_params = (0.01, 0.1, 0.1)
        real_sz224 = tf.image.resize_images(real, [224, 224])
        fake_sz224 = tf.image.resize_images(fake, [224, 224])    
        real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
        fake_feat55, fake_feat28, fake_feat7  = vggface_feat(fake_sz224)    
        loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
        loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
        loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))
    
    return loss_D, loss_G


# In[26]:


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


# In[45]:


loss_DA, loss_GA = define_loss(netDA, real_A, fake_A, vggface_feat)
loss_DB, loss_GB = define_loss(netDB, real_B, fake_B, vggface_feat)


# In[ ]:


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


# <a id='8'></a>
# # 9. Utils for loading/displaying images

# In[48]:


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
    image = random_transform( image, **random_transform_args )
    warped_img, target_img = random_warp( image )
    
    return warped_img, target_img


# In[53]:


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


# In[ ]:


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


# <a id='10'></a>
# # 10. Start Training
# 
# Show results and save model weights every `display_iters` iterations.

# In[ ]:


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

display_iters = 50
train_batchA = minibatchAB(train_A, batchSize)
train_batchB = minibatchAB(train_B, batchSize)

#while epoch < niter: 
while gen_iterations < 20000:
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
    
    if gen_iterations % display_iters == 0:
        if gen_iterations % (3*display_iters) == 0: # clear_output every 3*display_iters iters
            clear_output()
        print('[%d/%d][%d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
        % (epoch, niter, gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, time.time()-t0))   
        
        # get new batch of images and generate results for visualization
        _, wA, tA = train_batchA.send(14)  
        _, wB, tB = train_batchB.send(14)
        showG(tA, tB, path_A, path_B)        
        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        
        # Save models
        encoder.save_weights("models/encoder.h5")
        decoder_A.save_weights("models/decoder_A.h5" )
        decoder_B.save_weights("models/decoder_B.h5" )
        netDA.save_weights("models/netDA.h5")
        netDB.save_weights("models/netDB.h5")


# <a id='11'></a>
# # 11. Helper Function: face_swap()
# This function is provided for those who don't have enough VRAM to run dlib's cnn and GAN model at the same time.
# 
#     INPUTS:
#         img: A RGB face image of any size
#         path_func: a function that is either path_A or path_B
#     OUPUT:
#         result_img: A RGB swapped face image

# In[ ]:


def swap_face(img, path_func):
    input_size = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # generator expects BGR input    
    ae_input = cv2.resize(img, (64,64))/255. * 2 - 1 # resize img to 64x64 and normalize it     
    
    result = np.squeeze(np.array([path_func([[ae_input]])]))
    
    result = np.clip( (result + 1) * 255 / 2, 0, 255 ).astype('uint8')   
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    result = cv2.resize(result, (input_size[1],input_size[0]))
    return result


# In[ ]:


whom2whom = "BtoA" # default trainsforming faceB to faceA

if whom2whom is "AtoB":
    path_func = path_B
elif whom2whom is "BtoA":
    path_func = path_A
else:
    print ("whom2whom should be either AtoB or BtoA")


# In[6]:


input_img = plt.imread("./IMAGE_FILENAME.jpg")


# In[ ]:


plt.imshow(input_img)


# In[ ]:


result_img = swap_face(input_img, path_func)


# In[ ]:


plt.imshow(result_img)


# <a id='12'></a>
# # 12. Import Packages for Making Video Clips
# 
# Given a video as input, the following cells will detect face for each frame using dlib's cnn model. And use trained GAN model to transform detected face into target face. Then output a video with swapped faces.

# In[20]:


# Download ffmpeg if need, which is required by moviepy.

#import imageio
#imageio.plugins.ffmpeg.download()


# In[29]:


import face_recognition
from moviepy.editor import VideoFileClip


# <a id='13'></a>
# # 13. Make Video Clips w/o Face Alignment

# In[77]:


use_smoothed_mask = True
use_smoothed_bbox = True

#def get_gen_output(inp, path):
#    return np.squeeze(np.array([path([[inp]])]))

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
    image = input_img
    faces = face_recognition.face_locations(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
    
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
        
        ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1        
        result = np.squeeze(np.array([path_A([[ae_input]])])) # Change path_A/path_B here
        result = np.clip( (result + 1) * 255 / 2, 0, 255 ).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = cv2.resize(result, (roi_size[1],roi_size[0]))
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        
        if use_smoothed_mask:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = mask/255*result + (1-mask/255)*orig_img
        else:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = result
    
    return comb_img


# In[78]:


# Variables for smoothing bounding box
global prev_x0, prev_x1, prev_y0, prev_y1
global frames
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0

output = 'OUTPUT_VIDEO.mp4'
clip1 = VideoFileClip("TEST_VIDEO.mp4")
clip = clip1.fl_image(process_video)#.subclip(11, 13) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')


# <a id='14'></a>
# # 14. Make video clips w/ face alignment
# 
# The code is not refined. Also I can't tell if face alignment improves the result.
# 
# Code reference: https://github.com/nlhkh/face-alignment-dlib

# In[72]:


import gc
gc.collect()


# In[140]:


use_smoothed_mask = True
apply_face_aln = True
use_poisson_blending = False # SeamlessClone is NOT recommended for video.
use_comp_video = True # output a comparison video before/after face swap
use_smoothed_bbox = True

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
    
def extract_eye_center(shape):
    xs = 0
    ys = 0
    for pnt in shape:
        xs += pnt[0]
        ys += pnt[1]
    return ((xs//6), ys//6)

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M, (xc, yc), angle

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return 90
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotated_img(img, det):
    #print (det, img.shape)
    shape = face_recognition.face_landmarks(img, det)
    pnts_left_eye = shape[0]["left_eye"]
    pnts_right_eye = shape[0]["right_eye"]
    if len(pnts_left_eye) == 0 or len(pnts_right_eye) == 0:
        return img, None, None    
    le_center = extract_eye_center(shape[0]["left_eye"])
    re_center = extract_eye_center(shape[0]["right_eye"])
    M, center, angle = get_rotation_matrix(le_center, re_center)
    M_inv = cv2.getRotationMatrix2D(center, -1*angle, 1)    
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)    
    return rotated, M, M_inv, center

def process_video(input_img):   
    image = input_img
    # ========== Decrease image size if getting memory error ==========
    #image = input_img[:3*input_img.shape[0]//4, :, :]
    #image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2))
    orig_image = np.array(image)
    faces = face_recognition.face_locations(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = np.zeros([orig_image.shape[0], orig_image.shape[1]*2,orig_image.shape[2]])
        comb_img[:, :orig_image.shape[1], :] = orig_image
        comb_img[:, orig_image.shape[1]:, :] = orig_image
        if use_comp_video:
            return comb_img
        else:
            return image
    
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
                
        if apply_face_aln:
            do_back_rot = True
            image, M, M_inv, center = get_rotated_img(image, [(x0, y1, x1, y0)])
            if M is None:
                do_back_rot = False
        
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        roi_image = cv2_img[x0+h//15:x1-h//15, y0+w//15:y1-w//15, :]
        roi_size = roi_image.shape            
        
        if use_smoothed_mask:
            mask = np.zeros_like(roi_image)
            #print (roi_image.shape, mask.shape)
            mask[h//15:-h//15,w//15:-w//15,:] = 255
            mask = cv2.GaussianBlur(mask,(15,15),10)
            roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1        
        result = np.squeeze(np.array([path_A([[ae_input]])]))
        result = np.clip( (result + 1) * 255 / 2, 0, 255 ).astype('uint8')        
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = cv2.resize(result, (roi_size[1],roi_size[0]))        
        result_img = np.array(orig_image)
        
        if use_smoothed_mask and not use_poisson_blending:
            image[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = mask/255*result + (1-mask/255)*roi_image_rgb
        elif use_poisson_blending:
            c = (y0+w//2, x0+h//2)
            image = cv2.seamlessClone(result, image, mask, c, cv2.NORMAL_CLONE)     
            
        if do_back_rot:
            image = cv2.warpAffine(image, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
            result_img[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = image[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:]
        else:
            result_img[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = image[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:]   

        if use_comp_video:
            comb_img = np.zeros([orig_image.shape[0], orig_image.shape[1]*2,orig_image.shape[2]])
            comb_img[:, :orig_image.shape[1], :] = orig_image
            comb_img[:, orig_image.shape[1]:, :] = result_img
            
    if use_comp_video:
        return comb_img
    else:
        return result_img


# In[141]:


# Variables for smoothing bounding box
global prev_x0, prev_x1, prev_y0, prev_y1
global frames
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0

output = 'OUTPUT_VIDEO.mp4'
clip1 = VideoFileClip("TEST_VIDEO.mp4")
# .subclip(START_SEC, END_SEC) for testing
clip = clip1.fl_image(process_video)#.subclip(1, 5) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')

