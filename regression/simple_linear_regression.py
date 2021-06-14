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


# # Simple Linear Regression

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# ## Training the Simple Linear Regression model on the Training set

# In[7]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[8]:


y_pred = regressor.predict(X_test)


# ## Visualising the Training set results

# In[9]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# ## Visualising the Test set results

# In[10]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[10]:




