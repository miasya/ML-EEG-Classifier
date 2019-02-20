#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from scipy import io
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

### GET ALL MY FEATURES AND CLASSES FROM CSV ###
### AND PERFORM TRAIN, VALID, AND TEST SPLIT ###


data = DataFrame()
data = pd.read_csv("emotions.csv", header=None, low_memory=False)

test, valid, train = np.split(data, [int(2132 * 0.2), int(2132 * 0.4)])

x_train = train.iloc[:,0:2547]
y_train = train.iloc[:,2548]

x_valid = valid.iloc[:,0:2547]
y_valid = valid.iloc[:,2548]

x_test = test.iloc[:,0:2547]
y_test = test.iloc[:,2548]


### SET UP MY MODEL ###

MAX_FEAT = 1

train_accuracy = np.zeros(40)
valid_accuracy = np.zeros(40)
features = np.zeros(40)

for i in range (40):
    RFC = RandomForestClassifier(n_estimators=100, min_samples_split=3, max_features = MAX_FEAT)
    RFC.fit(x_train, y_train)

    # Print train accuracy
    print("MAX FEATURES: ", MAX_FEAT)
    print("train accuracy: ", accuracy_score(RFC.predict(x_train), y_train))
    print("valid accuracy: ", accuracy_score(RFC.predict(x_valid), y_valid))
    
    
    train_accuracy[i] = accuracy_score(RFC.predict(x_train), y_train)
    valid_accuracy[i] = accuracy_score(RFC.predict(x_valid), y_valid)
    features[i] = MAX_FEAT
    
    MAX_FEAT = MAX_FEAT + 1
    
    
    
# Plot results
print("train accuracy plot")
plt.scatter(features, train_accuracy, s=1)
plt.show()
print("valid accuracy plot")
plt.scatter(features, valid_accuracy, s=1)
plt.show()


# In[ ]:




