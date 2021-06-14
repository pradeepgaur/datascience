#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Configurations
DATA_FILE_PATH = "Market_Basket_Optimisation.csv"
PROJECT_NAME = "project_alpha"


# In[49]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[50]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'association/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Apriori

# ## Importing the libraries

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Data Preprocessing

# In[52]:


dataset = pd.read_csv(DATA_FILE_PATH, header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# ## Training the Apriori model on the dataset

# In[52]:





# In[53]:


# WARNING: Make sure to upload the apyori.py file into this Colab notebook before running this cell
# from apyori import apriori
import sys
sys.path.append(DIRECTORY_PATH)
from apyori import apriori

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# ## Visualising the results

# In[54]:


results = list(rules)


# In[55]:


print(results)

