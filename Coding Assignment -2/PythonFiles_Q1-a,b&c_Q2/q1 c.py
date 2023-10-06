#!/usr/bin/env python
# coding: utf-8

# ### Numerical Solution using NumPy

# In[1]:


import numpy as np
A = np.array([[1,2,1],[2,4,2],[3,1,9],[4,1,0],[2,1,4]])
B = np.array([1,3,1,0,9])
x = np.random.rand(3)


# In[2]:


def f(x):
    return (np.linalg.norm(np.dot(A,x)-B))**2


# In[3]:


def grad(x):
    return 2*np.dot(A.T,(np.dot(A,x)-B))


# ## Given learning rate is very high

# In[4]:


lr = 0.002 
num_iter = 10000


# In[5]:


def gD(x):
    for i in range(num_iter):
        x = x - lr*(grad(x))
        if i%1000 == 0:
            print(f"Iteration : {i} Value of f(x) = {f(x)}")
    answer = x
    print(f"Iteration : {i} = Min value of f = {f(x)} at x = {x}")
    return answer


# In[6]:


answer = gD(x)
answer


# In[7]:


np.dot(A.T,A)


# In[8]:


np.dot(A.T,B)


# In[9]:


np.linalg.norm(np.dot(A,answer)-B)**2


# In[ ]:




