#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Necessary Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[2]:


#Loading the dataset and displaying the First Few Rows
iris_data = pd.read_csv('iris.csv')
iris_data.head()


# In[3]:


# Split the data into features (X) and labels(Y).
x = iris_data.drop(columns=['petal.width', 'variety'])
y = iris_data['variety']


# In[4]:


#split the data into training and tesing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[5]:


#Standardize the features.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[6]:


#Create a Ml model
model = LogisticRegression()


# In[7]:


#train the Model
model.fit(x_train_scaled, y_train)


# In[8]:


#Evaluate the model on the testing set
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)


# In[9]:


#sample new data for predication.
new_data = np.array([[5.1, 3.5, 1.4],
                    [6.3, 2.9, 5.6],
                    [4.9, 3.0, 1.4]])


# In[10]:


#standardize the new data.
new_data_scaled = scaler.transform(new_data)


# In[11]:


predictions = model.predict(new_data_scaled)


# In[12]:


print("Predictions : ", predictions)


# In[ ]:




