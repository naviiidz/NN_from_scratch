import numpy as np
import random
import matplotlib.pyplot as plt
N=1000; M=20; 
# Train partition
p =[6,13,16,5] # Cluster centers
x=np.zeros((2*N,2)) # Pre allocation of the input and output
t =np.zeros((2*N,1)) # Target
# Synthetic Data
r =np.random.normal ( 0 , 1 , 2*N)
the=2*np.pi*np.random.rand(2*N)
x[0:N,0] = p[0] + r [ 0 :N]*np.cos(the[0:N] )
x[0:N,1] = p[1] + r [ 0 :N]*np.sin(the[0:N] )
t[0:N]=0
x[N:2*N,0] = p[2] + r[N:2*N]*np.cos(the[N:2*N] )
x[N:2*N,1] = p[3] + r[N:2*N]*np.sin(the[N:2*N] )
t[N:2*N]=1
plt.figure(figsize=(10,10))
plt.scatter(x[:,0] , x[:,1])

w1=0.5-np.random.rand() # parameter initialization
w2=0.5-np.random.rand()
b=0.5-np.random.rand()
ii=np.linspace(1,15,15)
# plt.plot(ii,(-b-w1* i*i ) / w2 )
a= range(0,2*N)
sp=random.sample(a,M+1000) ; # indeces for random selection of data for
# training and testing.
# Training
for i in range(0,M):
    y=b+w1*x[sp[i] , 0 ] + w2*x[sp[i] ,1]
    if y <0:
        y=0
    else:
        y=1
    e=t[sp[i]]- y
    w1=w1+e*x[sp[i],0]
    w2=w2+e*x[sp[i],1]
    b=b+e

# Testing
er=0
for i in range(M,M+1000):
    y=b+ w1*x[sp[i],0]+w2*x[sp[i],1]
    if y <0:
        y=0
    else :
        y=1
    e=abs(t[sp[i]]-y)
    er=er+e
er = er/1000
print(er)
plt.plot(ii,(-b-w1* ii)/w2)
plt.show()
