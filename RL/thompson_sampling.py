#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Configurations
DATA_FILE_PATH = "Ads_CTR_Optimisation.csv"
PROJECT_NAME = "project_alpha"


# In[7]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[8]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'RL/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Thompson Sampling

# ## Importing the libraries

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[10]:


dataset = pd.read_csv(DATA_FILE_PATH)


# ## Implementing Thompson Sampling

# In[11]:


import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward


# ## Visualising the results - Histogram

# In[12]:


plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

