#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Configurations
DATA_FILE_PATH = "Churn_Modelling.csv"
PROJECT_NAME = "project_alpha"


# In[2]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[3]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'deep_learning/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Artificial Neural Network

# ### Importing the libraries

# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[5]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Importing the dataset

# In[6]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[7]:


print(X)


# In[8]:


print(y)


# ### Encoding categorical data

# Label Encoding the "Gender" column

# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[10]:


print(X)


# One Hot Encoding the "Geography" column

# In[11]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[12]:


print(X)


# ### Feature Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[14]:


print(X)


# ### Splitting the dataset into the Training set and Test set

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Part 2 - Building the ANN

# ### Initializing the ANN

# In[16]:


ann = tf.keras.models.Sequential()


# ### Adding the input layer and the first hidden layer

# In[17]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the second hidden layer

# In[18]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the output layer

# In[19]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Training the ANN

# ### Compiling the ANN

# In[20]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the ANN on the Training set

# In[21]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# ## Part 4 - Making the predictions and evaluating the model

# ### Predicting the Test set results

# In[22]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Making the Confusion Matrix

# In[23]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

