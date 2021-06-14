#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Configurations
DATA_FILE_PATH = "Data.csv"
PROJECT_NAME = "project_alpha"


# In[ ]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[ ]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'data_prep/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Data Preprocessing Template

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

