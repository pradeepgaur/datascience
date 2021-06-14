#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Configurations
DATA_FILE_PATH = "Restaurant_Reviews.tsv"
PROJECT_NAME = "project_alpha"


# In[2]:


#Import drive
from google.colab import drive
drive.mount("/content/drive")


# In[3]:


BASE_PATH = '/content/drive/My Drive/' + PROJECT_NAME + '/datascience/'
DATA_FILE_PATH = BASE_PATH + 'data/' + DATA_FILE_PATH

DIRECTORY_PATH =  BASE_PATH + 'NLP/'
get_ipython().magic(u'cd {DIRECTORY_PATH}')


# # Natural Language Processing

# ## Importing the libraries

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[5]:


dataset = pd.read_csv(DATA_FILE_PATH, delimiter = '\t', quoting = 3)


# ## Cleaning the texts

# In[6]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# ## Creating the Bag of Words model

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# ## Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Training the Naive Bayes model on the Training set

# In[9]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[10]:


y_pred = classifier.predict(X_test)


# ## Making the Confusion Matrix

# In[11]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

