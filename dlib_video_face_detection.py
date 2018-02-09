
# coding: utf-8

# # Face detection for video input
# Images of detected faces will be saved to `./faces/fXroiY.png`, where `X` represents the Xth frame and `Y` the Yth face in Xth frame. 

# In[3]:


get_ipython().system('mkdir faces')


# ## 1. Install requirements
# 
# ========== CAUTION ==========
# 
# If you are running on local machine. Please read [this blog](http://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/) before pip installing packages.

# In[ ]:


get_ipython().system('pip install face_recognition')


# In[ ]:


get_ipython().system('pip install moviepy')


# In[ ]:


import imageio
imageio.plugins.ffmpeg.download()


# ## 2. Import packages

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import face_recognition
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 3. Config &  init

# In[2]:


global global_faces
global global_faces_vel
global frame_missingbox
global frames
frames = 0
global_faces = ()
global_faces_vel = ()
frame_missingbox = 0

# only those faces having size > image size / 50 will be saved.
# e.g., given 800 x 600 input, the minimum face images saved will be about 98 x 98.
min_face_scale = 1/50.


# ## 4. Crop faces
# Since dlib's cnn model performs really well on face deteciton, applying box-missing frame compensation is not needed.
# But for OpenCV's Haar-cascade Detection, compensating box-missing grame will increase the chance to crop face images in different angles.

# In[4]:


def process_video(input_img):   
    global global_faces
    global frame_missingbox
    global frames
    frames += 1
    # Resize input image if necessary.
    #img = cv2.resize(input_img, (input_img.shape[1]//3,input_img.shape[0]//3))
    img = input_img
    faces = face_recognition.face_locations(img, model="cnn")
    size_img = img.shape[0] * img.shape[1]
    
    face_detected = False
    idx = 0
    for (x0,y1,x1,y0) in faces:
        if np.abs((x1-x0) * (y1-y0)) > size_img * min_face_scale:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[x0:x1, y0:y1,:]
            fname = "./faces/f" + str(frames) + "roi" + str(idx) + ".png"
            plt.imsave(fname, roi_color)
            idx += 1            
            global_faces = (x0,y1,x1,y0)
            frame_missingbox = 0
            face_detected = True
        else:
            face_detected = False
    
    # box-missing frame compensation
    #if not face_detected and frame_missingbox <= 5 and not global_faces is ():
    #    (x0,y1,x1,y0) = global_faces
    #    roi_color = img[x0:x1, y0:y1,:]
    #    frame_missingbox += 1       
    
    return img


# Function `process_video` will be called at each frame, and image of that frame will be the argument of `process_video`.

# In[13]:


output = '_.mp4'
clip1 = VideoFileClip("INPUT_VIDEO.mp4")
clip = clip1.fl_image(process_video)#.subclip(0,10) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')


# ## 5. Pack `./faces` folder into a zip file `./faces.zip`

# In[9]:


import zipfile


# In[10]:


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


# In[15]:


zipf = zipfile.ZipFile('faces.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('faces/', zipf)
zipf.close()

