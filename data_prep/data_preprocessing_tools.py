#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Configurations
DATA_FILE_PATH = "Data.csv"
PROJECT_NAME = "project_alpha"


# In[39]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[40]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'data_prep/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Data Preprocessing Tools

# ## Importing the libraries

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[42]:


get_ipython().system(u'pwd')


# In[42]:





# ## Importing the dataset

# In[43]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[44]:


print(X)


# In[45]:


print(y)


# ## Taking care of missing data

# In[46]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[47]:


print(X)


# ## Encoding categorical data

# ### Encoding the Independent Variable

# In[48]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[49]:


print(X)


# ### Encoding the Dependent Variable

# In[50]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[51]:


print(y)


# ## Splitting the dataset into the Training set and Test set

# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[53]:


print(X_train)


# In[54]:


print(X_test)


# In[55]:


print(y_train)


# In[56]:


print(y_test)


# ## Feature Scaling

# In[57]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])


# In[58]:


print(X_train)


# In[59]:


print(X_test)

