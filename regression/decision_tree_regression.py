#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Configurations
DATA_FILE_PATH = "Position_Salaries.csv"
PROJECT_NAME = "project_alpha"


# In[17]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[18]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'regression/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Decision Tree Regression

# ## Importing the libraries

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[20]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# ## Training the Decision Tree Regression model on the whole dataset

# In[21]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# ## Predicting a new result

# In[22]:


regressor.predict([[6.5]])


# ## Visualising the Decision Tree Regression results (higher resolution)

# In[23]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

