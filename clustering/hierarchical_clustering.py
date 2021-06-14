#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Configurations
DATA_FILE_PATH = "Mall_Customers.csv"
PROJECT_NAME = "project_alpha"


# In[10]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[11]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'clustering/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Hierarchical Clustering

# ## Importing the libraries

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[13]:


dataset = pd.read_csv(DATA_FILE_PATH)
X = dataset.iloc[:, [3, 4]].values


# ## Using the dendrogram to find the optimal number of clusters

# In[14]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# ## Training the Hierarchical Clustering model on the dataset

# In[15]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# ## Visualising the clusters

# In[16]:


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

