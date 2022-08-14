#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
def net(X,W):                                                          #Net function
    net= np.dot(X,W) 
return net  
def output(lam,net,func):                                              #output Function
    if(func=="bipolarcont"):
        out=(2/(1+np.exp(-lam*net)))-1 
        return out 
    elif(func=="bipolardiscrete"): 
        if net>0: 
            return 1 
        else: 
            return -1              
def upd_weight(c,d,out,X,actfunc):                                    #Update Function 
        if(actfunc=="bipolarcont"): 
            return c*(d-out)*(1-(out*out))*X 
        elif(actfunc=="bipolardiscrete"): 
            return c*(d-out)*X 
        else: 
            return 0   
def upd_weight_deltarule(c,d,out,X,func):                               #DeltaLearning 
    return 0.5*c*(d-out)*(1-(out*out))*X      
def perceptron(c,lam,X,W,d,iterations,func):                            #Perceptron Function  
    for j in range(0,iterations): 
        print("----ITERATION: ",j,"----")
        for i,x1 in enumerate(X): 
            net1=net(x1,W) 
            out1=output(lam,net1,func) 
            W=W+upd_weight(c,d[i],out1,x,func) 
            print("Weight value for step -",i,"is",W) 
            return W   
def test_perceptron(NewX,W1,func,lam):                                 #Test Perceptron Function  
    for i,x1 in enumerate(NewX): 
        print("Input",x1) 
        net1=net(x1,W1)
        out1=output(lam,net1,func) 
        print("Output",out1) 
        if(out1>0): 
            print("Output Class is Class I") 
            else: 
                print("Output Class is Class II") 
X=np.array([[1,0,0,1,0,0,1,1,1,1],[0,1,0,0,1,0,0,1,0,1]])            # Initialization of X,W,lambda,c 
d=np.array([[1],[-1]]) 
W=np.zeros(10) 
c=np.array([0.01,0.1,1,2,10]) 
lam=2 
iterations=4
actfunc='bipolardiscrete' 
print(np.dot(X,np.transpose(W))) 
print("No. of Iterations, Activation function: ",iterations) 
print("Initial Weights:",W) 
for i,c1 in enumerate(c): 
    print("LEARNING CONSTANT : ",c1,)                               #training of the perceptron 
    out=[] 
    W1=perceptron(c1,lam,X,W,d,iterations,func) 
    print("Weights after training",W1) 
    print("Expected Output:",d.reshape(2)) 
    NewX=np.array([[1,0,0,1,0,0,1,1,1,1],[0,1,0,0,1,0,0,1,0,1]]) 
    for i,x in enumerate(NewX): 
        net1=net(x,W1) 
        out1=output(lam,net1,func) 
        if(out1>0):
            out1=1 
            else: 
                out1=-1 
                out=out+[out1]             
                print("Actual Output:",out)


# In[ ]:





# In[ ]:




