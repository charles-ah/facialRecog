import numpy as np
from sklearn.linear_model import LogisticRegression

out = open("sigmoid.txt","w")

pos = open("positive.txt","r").read()

neg = open("negative.txt","r").read()

pos = pos.split("\n")[0:-1]
neg = neg.split("\n")[0:-1]

#print pos

y = np.ones(len(pos)+len(neg))
y[len(pos):len(pos)+len(neg)] = np.zeros(len(neg))
for i in range(len(pos)):
    pos[i] = float(pos[i])

for i in range(len(neg)):
    neg[i] = float(neg[i])
    

#print pos
#print neg
#print pos.append(neg)
X = np.ones((len(pos) + len(neg),2),dtype=np.float128)
X[0:len(pos),1] = np.array(pos)
X[len(pos):len(pos) + len(neg),1] = np.array(neg)
#X[:,1] = np.zeros(len(pos) + len(neg))
X[:,1] = (1/np.ptp(X[:,1]))*(X[:,1] - np.sum(X[:,1])/X.shape[0])

theta = np.zeros(2)
#print X
def sigmoid(arr):
    return 1.0/(1.0 + np.exp(-1*arr))

#print np.sum( np.log( sigmoid( np.dot( X,theta ) ) )*y - ( y-1 )*np.log( 1-sigmoid( np.dot( X,theta ) ) ) )
m = (1.0/(len(pos) + len(neg)))
cost = -1*m * np.sum(np.log(sigmoid(np.dot(X,theta)))*y - (y-1)*np.log(1-sigmoid(np.dot(X,theta)))) 

print cost

alpha = 22.05
for i in range(500):
    grad = m*np.dot(np.transpose(X),sigmoid(np.dot(X,theta)) - y) 
    #print grad
    theta = theta - alpha*grad
    

print theta
'''
arr = sigmoid(np.dot(X,theta))
for x in arr:
    print x
'''
#print arr[np.argmin(arr)]
out.write(str(sigmoid(np.dot(X,theta))))
out.close()

print -1*m * np.sum(np.log(sigmoid(np.dot(X,theta)))*y - (y-1)*np.log(1-sigmoid(np.dot(X,theta))))

np.save("theta.npy",theta)
