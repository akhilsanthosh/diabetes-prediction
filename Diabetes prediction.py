#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


dataset = pd.read_csv("Diabetes.csv")
dataset


# In[8]:


dataset.isnull().sum()


# In[9]:


dataset.info()


# In[10]:


dataset.describe()


# In[12]:


plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(),annot=True,fmt=".3f",cmap="YlGnBu")
plt.title("Correlation heatmap")


# In[15]:


plt.figure(figsize=(10,8))
kde=sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1],color="Red",shade=True)
kde=sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0],color="Blue",shade=True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend(["+ve","-ve"])


# In[16]:


plt.figure(figsize=(10,8))
sns.violinplot(data=dataset,x="Outcome",y="Glucose",split=True,linewidth=2,inner="quart"),


# In[17]:


plt.figure(figsize=(10,8))
kde1=sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1],color="Red",shade=True)
kde1=sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0],color="Blue",shade=True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Density")
kde.legend(["+ve","-ve"])


# In[19]:


dataset["Glucose"]=dataset["Glucose"].replace(0,dataset["Glucose"].median())
dataset["BloodPressure"]=dataset["BloodPressure"].replace(0,dataset["BloodPressure"].median())
dataset["BMI"]=dataset["BMI"].replace(0,dataset["BMI"].mean())
dataset["Insulin"]=dataset["Insulin"].replace(0,dataset["Insulin"].mean())


# In[20]:


dataset


# In[23]:


X=dataset.drop(["Outcome"],axis=1)
y=dataset["Outcome"]


# In[24]:


y


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[29]:


X_train


# In[31]:


from sklearn.neighbors import KNeighborsClassifier


# In[32]:


training_accuracy=[]
test_accuracy=[]
for n_neighbors in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    
    training_accuracy.append(knn.score(X_train,y_train))
    test_accuracy.append(knn.score(X_test,y_test))


# In[33]:


plt.plot(range(1,11),training_accuracy,label="training_accuracy")
plt.plot(range(1,11),test_accuracy,label="test_accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[34]:


knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)   


# In[35]:


print(knn.score(X_train,y_train))
print(knn.score(X_test,y_test))


# In[37]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=0)
dt.fit(X_train,y_train)
print(dt.score(X_train,y_train),":Training accuracy")
print(dt.score(X_test,y_test),":Test accuracy")


# In[39]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
print(mlp.score(X_train,y_train),":Training accuracy")
print(mlp.score(X_test,y_test),":Test accuracy")


# In[40]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.fit_transform(X_test)


# In[41]:


X_train_scaled


# In[43]:


mlp1=MLPClassifier(random_state=0)
mlp1.fit(X_train_scaled,y_train)
print(mlp1.score(X_train_scaled,y_train),":Training accuracy")
print(mlp1.score(X_test_scaled,y_test),":Test accuracy")


# In[ ]:




