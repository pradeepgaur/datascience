#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Configurations
DATA_FILE_PATH = "Social_Network_Ads.csv"
PROJECT_NAME = "project_alpha"


# In[9]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[10]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'classification/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Decision Tree Classification

# ## Importing the libraries

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[12]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Training the Decision Tree Classification model on the Training set

# In[15]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# ## Making the Confusion Matrix

# In[16]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

