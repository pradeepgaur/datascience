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


# # Support Vector Regression (SVR)

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


# In[6]:


y = y.reshape(len(y),1)


# ## Splitting the dataset into the Training set and Test set

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Feature Scaling

# In[8]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


# ## Training the SVR model on the Training set

# In[9]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[10]:


y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[11]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

