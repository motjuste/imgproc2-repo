from PIL import Image
import random
from sets import Set
import glob
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import distance

Xtrain_file_list=[]
Xtest_file_list=[]
v=[]
l=[]
Xtrain=np.array
Xtest=np.array

uiucPos = glob.glob('../resources/uiucTrain/uiucPos*.pgm')
uiucNeg = glob.glob('../resources/uiucTrain/uiucNeg*.pgm')


""" Import the training data """   
pos_t = [Image.open(fn).convert('L') for fn in uiucPos]
pos_t = np.asarray([np.array(im) for im in pos_t])
neg_t = [Image.open(fn).convert('L') for fn in uiucNeg]
neg_t = np.asarray([np.array(im) for im in neg_t])

""" Zero mean the data """
pos_t = pos_t - pos_t.mean()
neg_t = neg_t - neg_t.mean()

""" PROCEDURE FOR RANK ro=1 """

## Labelled images in form of {(Xi,yi)}
#t_set = np.array([(pos_t, np.ones((pos_t.size,),dtype=np.int8)), 
#                  (neg_t, np.ones((neg_t.size,),dtype=np.int8)*-1)])


# X=t_set[0]
# y=t_set[1]
t_set = (np.concatenate((pos_t,neg_t)), 
         np.concatenate((np.ones((pos_t.shape[0],),dtype=np.int8),np.ones((neg_t.shape[0],),dtype=np.int8)*-1)))

print t_set[0].shape, t_set[1].shape
                  
(m,n) = pos_t[0].shape
u = np.random.rand(m)
v = np.zeros(n)
#print u

t_MAX = 10**2 # Stop at t_MAX if didn't converge
EPSILON = 10e-5 # Convergence threshold

for t in range(1, t_MAX):
    # Keep u for convergence check 
    prev_u = u
    
    # Contraction: sum u with 1-st dimension of t_set[0]
    X = np.tensordot(u, t_set[0], axes=([0], [1]) )
    #print "contraction uX ", X.shape
    
    # UPDATE v
    v = np.dot(la.pinv(np.dot(X.T,X)),np.dot(X.T,t_set[1]))
    #print v.shape
    
    # Contraction: sum u with 1-st dimension of t_set[0]
    X = np.tensordot(t_set[0], v, axes=([2], [0]) )
    #print "contraction Xv ", X.shape
    
    # UPDATE u
    u = np.dot(la.pinv(np.dot(X.T,X)),np.dot(X.T,t_set[1]))
    #print u.shape
    
    diff = abs(sum(u-prev_u))
    print EPSILON, "<-", diff
    if diff <= EPSILON:
        print "converged at t:", t
        break 

# template with rank ro=1
W = np.outer(u,v.T)
print W.shape