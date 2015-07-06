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


uiucTest = glob.glob('../resources/uiucTest/TEST_*.pgm')


# Load projection matrix W
W = np.load("projection_w_r=9.dat")
plt.imshow(W, cmap=cm.gray)
# Load mean of train set
m = np.load("t_set-mean.npy")
print "mean", m

# construct classifier
def classify(I, W, theta):
    res_map = signal.correlate2d(I,W,mode='same', fillvalue=127)-m # zero-meaned
    res_map = image_ops.normalize_array(res_map)
    
    # Smooth the response map
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

def quality_measure_img(img_i, x, y):
    points = np.empty((len(x),2))
    for i in range (0, len(x)):
        points[i][0] = x[i]
        points[i][1] = y[i]
        
    points.dump('points/'+str(img_i)+'.dat')

""" Import and classify the test data """
theta = 235
for img_i in range (0, len(uiucTest)):
#for img_i in range (0, 30):
    t_img = np.array(Image.open(uiucTest[img_i]).convert('L'))
    y = classify(t_img,W,theta)
    
    res_points_x = []
    res_points_y = []
    for i in range (0,y.shape[0]):
        for j in range (0,y.shape[1]):
            if y[i][j]:
                res_points_x = np.append(res_points_x, i)
                res_points_y = np.append(res_points_y, j)
                currentAxis = plt.gca()
                currentAxis.add_patch(Rectangle((j-15,i-10), 30, 20, fill=False, edgecolor="red"))
    
    quality_measure_img(img_i, res_points_x, res_points_y)
    
    