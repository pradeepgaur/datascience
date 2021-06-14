#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Configurations
DATA_FILE_PATH = "Salary_Data.csv"
PROJECT_NAME = "project_alpha"


# In[2]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[3]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'regression/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Random Forest Regression

# ## Importing the libraries

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[5]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training the Random Forest Regression model on the whole dataset

# In[7]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[8]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[9]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

