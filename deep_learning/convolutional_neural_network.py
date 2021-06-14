#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Configurations
DATA_FILE_PATH = ""
PROJECT_NAME = "project_alpha"


# In[ ]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[ ]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'deep_learning/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Convolutional Neural Network

# ### Importing the libraries

# In[ ]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Generating images for the Training set

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# ### Generating images for the Test set

# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# ### Creating the Training set

# In[ ]:


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# ### Creating the Test set

# In[ ]:


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[ ]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))


# ### Step 2 - Pooling

# In[ ]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# ### Adding a second convolutional layer

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# ### Step 3 - Flattening

# In[ ]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Step 5 - Output Layer

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[ ]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[ ]:


cnn.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)

