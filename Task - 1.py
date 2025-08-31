#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Necessary Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[2]:


#Loading the dataset and displaying the First Few Rows
iris_data = pd.read_csv('iris.csv')
iris_data.head()


# In[3]:


# Split the data into features (X) and labels(Y).
x = iris_data.drop(columns=['petal.width', 'variety'])
y = iris_data['variety']


# In[4]:


x.head()


# In[5]:


#Create a Ml model
model = LogisticRegression()


# In[6]:


#train the Model
model.fit(x.values, y)


# In[7]:


predictions = model.predict([[4.6, 3.5, 1.5]])


# In[8]:


print(predictions)


# In[ ]:




