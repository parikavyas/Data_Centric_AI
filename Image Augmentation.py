#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[9]:


from tensorflow import keras


# In[14]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img


# In[11]:


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# In[45]:


file_dir = "C:/Users/pv23228/Documents/Personal/Data_Centric_AI/data/train"
a = []
for folder in os.listdir(file_dir):
    print(folder)
    
    if(folder.startswith(".") != True):
        
        #Gives the folder - i, ii, iii, iv, v, vi, etc
        folder_dir = file_dir + "/" + folder
        
        for img_file in os.listdir(folder_dir):
            
            if(img_file.startswith(".") != True):
                #Gives Image Path
                img = load_img(folder_dir + "/" + img_file)  # this is a PIL image
                
                x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                i = 0
                saved_dir = "C:/Users/pv23228/Documents/Personal/Data_Centric_AI/data/augmented"+ "/" + folder
                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=saved_dir, save_prefix=folder, save_format='jpeg'):
                    i += 1
                    if i > 5:
                        break  # otherwise the generator would loop indefinitely


# In[ ]:




