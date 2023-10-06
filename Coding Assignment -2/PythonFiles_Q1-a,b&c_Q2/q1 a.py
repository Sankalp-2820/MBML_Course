#!/usr/bin/env python
# coding: utf-8

# ### Numerical Solution using NumPy

# In[1]:


import numpy as np
A = np.array([[2,-1,-1],[-1,2,0],[-1,0,1]])
b = 1
x = np.random.rand(3)


# In[2]:


def f(x):
    return np.dot(x.T,np.dot(A,x)) + b


# In[3]:


def grad(x):
    return np.dot(A+A.T,x)


# In[4]:


grad(x)


# In[5]:


lr = 0.02
num_iter = 10000


# In[6]:


def gD(x):
    for i in range(num_iter):
        x = x - lr*(grad(x))
        if i%1000 == 0:
            print(f"Iteration : {i} Value of f(x) = {f(x)}")
    print(f"Iteration : {i} = Min value of f = {f(x)} at x = {x}")


# In[7]:


gD(x)

