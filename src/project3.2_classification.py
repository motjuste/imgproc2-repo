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
from skimage.feature import peak_local_max



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
    
    coord_local_max = peak_local_max(res_map, min_distance=20, threshold_abs=theta, exclude_border=False)
    loc_max_res_map = res_map[coord_local_max[:, 0], coord_local_max[:, 1].T]
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,2)
    plt.imshow(res_map,cmap=cm.gray)
    a.set_title('Response map')    
    ###    
    a=fig.add_subplot(1,2,1)
    plt.imshow(I,cmap=cm.gray)
    a.set_title('Test image')
    
    return coord_local_max

""" Import and classify the test data """
theta = 235
#for img_i in range (0, len(uiucTest)):
for img_i in range (0, 50):
    t_img = np.array(Image.open(uiucTest[img_i]).convert('L'))
    coord_local_max = classify(t_img,W,theta)
    
    res_points_x = []
    res_points_y = []
    for r in coord_local_max:
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((r[1]-15,r[0]-10), 30, 20, fill=False, edgecolor="red"))
        
    