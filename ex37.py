#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
X=np.array([[5,7],[7,3],[3,2],[5,4],[0,0],[-1,-3],[-2,3],[-3,0],])
print(X)
w=np.array([[0,0]])
d=np.array([1,1,1,1,-1,-1,-1,-1])
c=1
iter=4
def discrete_perceptron(X,d,w,c,iter):
    for i in range(1,iter):
        for j,n in enumerate(X):
            print("Iteration",i)
            print("-------------")
            net=np.dot(X[j],np.transpose(w))
            if net>0:
                out=1
            else:
                out=-1
            print("out value:",out)
            print("Teacher value:",d[j])
            r=(d[j]-out)
            print("R:",r)
            delta_w=c*r*X[j]
            w=w+delta_w
            print("Weight:",w)
discrete_perceptron(X,d,w,c,iter)


def test_perceptron(final_out,X,w):
        for j,n in enumerate(X):
            net = np.dot(X[j],np.transpose(w))
            if net>0:
                out = 1
            else:
                out = -1
                final_out = final_out+[out]
                return final_out

new_input=np.array([[-2,3],[2,3],])
print("---TESTING---")
final_out=[]
final_output=test_perceptron(final_out,new_input,w)
print("Final output:",final_output)


# In[ ]:




