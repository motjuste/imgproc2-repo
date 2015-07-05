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
    v[_r] = np.zeros(n)
    
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

# construct classifier
def classify(I, W, theta):
    res_map = signal.convolve2d(I,W,mode='same', fillvalue=127)-m # zero-meaned
    res_map = image_ops.normalize_array(res_map)
    
#    print I.shape, res_map.shape
#    p0 = (I.shape[0]-res_map.shape[0])/2
#    p1 = (I.shape[1]-res_map.shape[1])/2 
#    res_map= np.pad(res_map, ((p0,p0),(p1,p1)), 'constant', constant_values=(0,0))
#    print "after padding", I.shape, res_map.shape
    
    
    #res_map = ndimage.filters.gaussian_filter(res_map, sigma=2)
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,2)
    plt.imshow(res_map,cmap=cm.gray)
    a.set_title('Response map')    
    ###    
    a=fig.add_subplot(1,2,1)
    plt.imshow(I,cmap=cm.gray)
    a.set_title('Test image')
    
    Theta = np.ones(res_map.shape)*theta
    return res_map>=Theta # y: boolean matrix of y(i,j)
    
    #return 1 if np.dot(W,X) >= theta else -1


""" Import and classify the test data """
#for i in range (0, uiucTest):
for i in range (0, 50):
    #t_img = np.array(Image.open(uiucTest[i]).convert('L'))
    #y = classify(t_img,W,240)
    y = classify(pos_t[i],W,240)
    
    for i in range (0,y.shape[0]):
        for j in range (0,y.shape[1]):
            if y[i][j]:
                currentAxis = plt.gca()
                currentAxis.add_patch(Rectangle((j-15,i-10), 30, 20, fill=False, edgecolor="red"))

    