#!/usr/bin/env python
# coding: utf-8

# ### Numerical Solution Using NumPy

# In[1]:


import numpy as np
A = np.array([[1,2],[2,4],[3,1]])
B = np.array([1,3,1])
x = np.random.rand(2)
C = np.dot(A,x)


# In[2]:


def f(x):
    C = A*x
    return (np.linalg.norm(np.dot(A,x)-B))**2


# In[3]:


f(x)


# In[4]:


def grad(x):
    C = A*x
    return 2*np.dot(A.T,(np.dot(A,x)-B))


# In[5]:


grad(x)

lr = 0.02
num_iter = 10000
# In[6]:


def gD(x):
    for i in range(10000):
        x = x - 0.01*(grad(x))
        if i%1000 == 0:
            print(f"Iteration : {i} Value of f(x) = {f(x)}")
        ans = x
    print(f"Iteration : {i} = Min value of f = {f(x)} at x = {x}")
    return ans


# In[7]:


answer = gD(x)


# In[8]:


np.linalg.norm(np.dot(A,answer)-B)**2


# In[9]:


2*np.dot(A.T,A)


# In[10]:


np.dot(A.T,B)


# In[11]:


np.dot(A.T,A)


# In[12]:


np.linalg.norm(np.dot(A,answer)-B)**2


# In[ ]:




