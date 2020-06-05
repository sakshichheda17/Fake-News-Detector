#!/usr/bin/env python
# coding: utf-8

# ## PROBLEM STATEMENT 
# To build a model to accurately classify a piece of news as REAL or FAKE.

# In[1]:


#Data manipulation
import numpy as np
import pandas as pd

#Plotting
import seaborn as sn
import matplotlib.pyplot as plt

#Splitting data to test and train
from sklearn.model_selection import train_test_split 

#Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

#Model training
from sklearn.linear_model import PassiveAggressiveClassifier

#Model accuracy evaluation
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


#Reading data
df = pd.read_csv('news.csv')

#Getting data shape and head
print(df.shape)
df.head()


# In[3]:


#Split data into test and train
x_train, x_test, y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.3)


# In[4]:


#Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

#Fit and transfrom the train set
tfidf_train=vectorizer.fit_transform(x_train)

#Transform the test set
tfidf_test=vectorizer.transform(x_test)


# In[5]:


#Initialize PassiveAggressiveClassifier
clf=PassiveAggressiveClassifier(max_iter=50)

#Fit the vectorized train set
clf.fit(tfidf_train,y_train)


# In[6]:


#Predict on vectorized test set
pred = clf.predict(tfidf_test)

#Calculate accuracy score and print it
score = accuracy_score(y_test,pred)
print('Accuracy Score : {}%'.format(round(score*100,2)))


# In[7]:


#Build Confusion matrix
cf_matrix = confusion_matrix(y_test,pred, labels=['FAKE','REAL'])

#Visualize Confusion matrix using heatmap
heat_map=sn.heatmap(cf_matrix,annot=True,fmt='',cmap='RdPu')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted Label')


# In[ ]:




