#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# # Importing Dataset

# In[9]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepel_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, names=column_names)
iris_data.head(5)


# In[7]:


type(iris_data)


# In[13]:


iris_data.iloc[1:50]


# In[14]:


iris_data.iloc[51:100]


# In[15]:


iris_data.iloc[101:150]


# In[22]:


iris_data.describe()


# In[23]:


sns.pairplot(iris_data, hue="class")
plt.show()


# In[29]:


X = iris_data.drop("class", axis=1)
x


# In[28]:


y = iris_data["class"]
y


# In[40]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)


# In[58]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)


# In[42]:


y_pred = knn.predict(X_test)


# In[45]:


print("Accuracy:", accuracy_score(y_test,y_pred))


# In[47]:


print(classification_report(y_test,y_pred))


# In[61]:


X_test.head(2)

