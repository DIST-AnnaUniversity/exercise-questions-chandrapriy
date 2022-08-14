#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
X=np.array([[0.8,0.5,0],[0.9,0.7,0.3],[1,0.8,0.5],[0,0.2,0.3],[0.2,0.1,1.3],[0.2,0.7,0.8],])    #Input values
print("Input:\n",X) 
w=np.array([[0,0,0]])         #initial weight
print("Initial weight",w)
d=np.array([1,1,1,-1,-1,-1])  #Fix teacher value of d
c=3                           #Fix learning rate c
iteration=5                   #set no.of.iterations

def dis_perceptron(X,d,w,c,iteration):  #discrete function
    for i in range(1,iteration):
         for j,n in enumerate(X):
                print("--Iteration--",i)
                net=np.dot(X[j],np.transpose(w))   #calculate net value
                if net>0:
                    out=1                          
                else:
                    out=-1
                    s=(d[j]-out)                   #calculate s
                    del_w=c*s*n                    #calculate del_w
                    weight=w+del_w                 #calculate weight and print 
                    print("Weight:",weight)
            
dis_perceptron(X,d,w,c,iteration)


# In[ ]:




