from PIL import Image
import random
from sets import Set
import glob
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import distance
#import gram_schmidt as orth

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
# X=t_set[0]
# y=t_set[1]
t_set = (np.concatenate((pos_t,
                         neg_t)), 
         np.concatenate((np.ones((pos_t.shape[0],),dtype=np.int8),
                         np.ones((neg_t.shape[0],),dtype=np.int8)*-1)))

print t_set[0].shape, t_set[1].shape

#R = [1,3,9]
#R = [1,2,3,4,5,6,7,8,9]
R = [1,2,3]
(m,n) = t_set[0][0].shape
u = np.empty((len(R),m))
v = np.empty((len(R),n))
print u.shape, v.shape
for r in R:
    # zero-indexify r ! 
    _r = r-1
    u[_r] = np.random.rand(m)
    v[_r] = np.zeros(n)
    
    # projection
    def proj(v1, v2):
        return map((lambda x : x * (np.dot(v2, v1) / np.dot(v1, v1))), v1)
    
    # Orthogonalize u
    if r>1:
        for i in range (2, r):
            u[i] = u[i-1] - proj(u[i],u[i-1])
            
    
    t_MAX = 10**2 # Stop at t_MAX if didn't converge
    EPSILON = 10e-5 # Convergence threshold
    
    for t in range(1, t_MAX):
        # Keep sum of u for convergence check 
        prev_u_sum = sum(u[_r])
        
        # Contraction: sum u with 1-st dimension of t_set[0]
        X = np.tensordot(u[_r], t_set[0], axes=([0], [1]) )
        #print "contraction uX ", X.shape
        
        # UPDATE v
        v[_r] = np.dot(la.pinv(np.dot(X.T,X)),np.dot(X.T,t_set[1]))
        #print v[_r].shape
        
        # Orthogonalize v
        if r>1:
            for i in range (2, r):
                v[i] = v[i-1] - orth.proj(v[i],v[i-1])
        
        # Contraction: sum u with 1-st dimension of t_set[0]
        X = np.tensordot(t_set[0], v[_r], axes=([2], [0]) )
        #print "contraction Xv ", X.shape
        
        # UPDATE u
        u[_r] = np.dot(la.pinv(np.dot(X.T,X)),np.dot(X.T,t_set[1]))
        #print u[_r].shape
        
        # Orthogonalize u
        if r>1:
            for i in range (2, r):
                u[i] = u[i-1] - orth.proj(u[i],u[i-1])
        
        #print sum(u[_r]),sum(prev_u)
        
        diff = abs(sum(u[_r])-prev_u_sum)
        print EPSILON, "<-", diff
        if diff <= EPSILON:
            print "converged at t:", t
            break 
    
    # template with rank ro=1
    W = np.outer(u[_r],v[_r].T)
    print W.shape
    
    plt.imshow(W, cmap=cm.gray)