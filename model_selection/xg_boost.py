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

DIRECTORY_PATH =  BASE_PATH + 'model_selection/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # XGBoost

# ## Importing the libraries

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[5]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[6]:


print(X)


# In[7]:


print(y)


# ## Encoding categorical data

# ### Label Encoding the "Gender" column

# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[9]:


print(X)


# ### One Hot Encoding the "Geography" column

# In[10]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[11]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training XGBoost on the Training set

# In[13]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[14]:


y_pred = classifier.predict(X_test)


# ## Making the Confusion Matrix

# In[15]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ## Applying k-Fold Cross Validation

# In[16]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

