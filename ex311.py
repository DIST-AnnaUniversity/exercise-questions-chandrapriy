#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
X=np.array([[0.8,0.5,0],[0.9,0.7,0.3],[1,0.8,0.5],[0,0.2,0.3],[0.2,0.1,1.3],[0.2,0.7,0.8],])  #INPUT
print("INPUT VALUES:\n",X)    #print input
w=np.array([[0,0,0]])         #include initial weight
d=np.array([1,1,1,-1,-1,-1])  #add teacher value
c=1.5                         #Fix learning rate
iteration=3                   #no.of.iterations
l=1

#continuous function
def Continuous_perceptron(X,d,w,c,iteration):
    for i in range(1,iteration):
         for j,n in enumerate(X):
                print("--Iteration--",i)
                net=np.dot(X[j],np.transpose(w))  #calculate net value
                out=(2/(1+np.exp(-l*net)))-1      #calculate out
                z=(d[j]-out)*(1-(out*out))        
                del_w=c*z*n
                w=w+del_w                         #calculate weight
                print("Weight:",w)                #print weight
    return w
            
w1=Continuous_perceptron(X,d,w,c,iteration)      #contionuous function
X=np.array([[0.8,0.5,0],[0.9,0.7,0.3],[1,0.8,0.5],[0,0.2,0.3],[0.2,0.1,1.3],[0.2,0.7,0.8],])
c=1                                              #learning rate
l=1                                              #Fix lamda value
print("--------Testing---------")
def Test_perceptron(X,w,l):     
    for i,n in enumerate(X):
            net=np.dot(X[i],np.transpose(w))
            out=(2/(1+np.exp(-l*net)))-1
            if out>0:
                print("Class I:",X[i])            #test whether class 1
            else:
                print("Class II:",X[i])           #else belongs to class 2
Test_perceptron(X,w1,l)
            


# In[ ]:




