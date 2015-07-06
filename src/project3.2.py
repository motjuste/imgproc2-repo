from PIL import Image
import random
from sets import Set
import glob
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import image_ops
from scipy import signal
from matplotlib.patches import Rectangle
from scipy import ndimage
import scipy
import pylab

Xtrain_file_list=[]
Xtest_file_list=[]
v=[]
l=[]
Xtrain=np.array
Xtest=np.array

uiucPos = glob.glob('../resources/uiucTrain/uiucPos*.pgm')
uiucNeg = glob.glob('../resources/uiucTrain/uiucNeg*.pgm')
uiucTest = glob.glob('../resources/uiucTest/TEST_*.pgm')


""" Import the training data """   
pos_t = [Image.open(fn).convert('L') for fn in uiucPos]
pos_t = np.asarray([np.array(im) for im in pos_t])
neg_t = [Image.open(fn).convert('L') for fn in uiucNeg]
neg_t = np.asarray([np.array(im) for im in neg_t])

m = (pos_t.mean() + neg_t.mean())/2
print "mean",m
np.save("t_set-mean", m)

## Labelled images in form of {(Xi,yi)}
# X <= t_set[0]
# y <= t_set[1]
Np = pos_t.shape[0]
Nn = neg_t.shape[0]
t_set = (np.concatenate((pos_t-m, # zero mean patches
                         neg_t-m)), 
         np.concatenate((np.ones(Np)/Np, # labels
                         np.ones(Nn)*-1/Nn)))

print t_set[0].shape, t_set[1].shape


# Based on modified Gram-Schmidt
# https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability    
def orthogonalize(V, r):
    def project(v1, v2):
        #return map((lambda x : x * (np.dot(v2, v1) / np.dot(v1, v1))), v1)
        return (np.dot(v2, v1) / np.dot(v1, v1))*v1    
    
    for i in range (2, r):
        V[i] = V[i-1] - project(V[i],V[i-1])
    return V
#######################

R = 3
(m,n) = t_set[0][0].shape
u = np.empty((R,m))
v = np.empty((R,n)) 
w = np.empty((R,m,n)) # template
print u.shape, v.shape
for r in range (1,R+1):
    _r = r-1 # zero-indexify r ! 
    u[_r] = np.random.rand(m)
    v[_r] = np.empty(n)
    
    # Orthogonalize u
    u = orthogonalize(u,r)
            
    
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
        v = orthogonalize(v,r)
        
        # Contraction: sum u with 1-st dimension of t_set[0]
        X = np.tensordot(t_set[0], v[_r], axes=([2], [0]) )
        #print "contraction Xv ", X.shape
        
        # UPDATE u
        u[_r] = np.dot(la.pinv(np.dot(X.T,X)),np.dot(X.T,t_set[1]))
        #print u[_r].shape
        
        # Orthogonalize u
        u = orthogonalize(u,r)
        
        #print sum(u[_r]),sum(prev_u)
        
        diff = abs(sum(u[_r])-prev_u_sum)
        print EPSILON, "<-", diff
        if diff <= EPSILON:
            print "converged at t:", t
            break 
    # endfor
            
    # add template to templates vector w
    w[_r] = np.outer(u[_r],v[_r].T)
    #print w[_r].shape
    
# endfor

W = sum(w)
print "W", W.shape
plt.imshow(W, cmap=cm.gray)
#plt.savefig("projection_w_r=9")
W.dump("projection_w.dat")


neg_m = np.empty(neg_t.shape[0])
for i in range (0, neg_t.shape[0]):
    #t_img = np.array(Image.open(uiucTest[i]).convert('L'))
    #y = classify(t_img,W,240)
    neg_m[i] = signal.convolve2d(neg_t[i],W,mode='same', fillvalue=127).max()
#neg_m = image_ops.normalize_array(neg_m)
print np.average(neg_m)
    
    
pos_m = np.empty(pos_t.shape[0])
for i in range (0, pos_t.shape[0]):
    #t_img = np.array(Image.open(uiucTest[i]).convert('L'))
    #y = classify(t_img,W,240)
    pos_m[i] = signal.convolve2d(pos_t[i],W,mode='same', fillvalue=127).max()
#pos_m = image_ops.normalize_array(pos_m)
print np.average(pos_m)

plt.figure()
plt.hist(neg_m, 10, histtype='stepfilled', stacked=True, fill=True,  color='r', label='neg_t', normed=True)
plt.hist(pos_m, 10, histtype='stepfilled', stacked=True, fill=True, color='g', alpha=0.8, label='pos_t', normed=True)
plt.xlabel("Max Value")
plt.ylabel("")
plt.legend()
plt.show()
print "min pos", pos_m.min()