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
import ast


uiucPos = glob.glob('../resources/uiucTrain/uiucPos*.pgm')
uiucNeg = glob.glob('../resources/uiucTrain/uiucNeg*.pgm')


""" Import the training data """   
pos_t = [Image.open(fn).convert('L') for fn in uiucPos]
pos_t = np.asarray([np.array(im) for im in pos_t])
neg_t = [Image.open(fn).convert('L') for fn in uiucNeg]
neg_t = np.asarray([np.array(im) for im in neg_t])

m = (pos_t.mean() + neg_t.mean())/2
Np = pos_t.shape[0]
Nn = neg_t.shape[0]

t_set = np.concatenate((pos_t-m, neg_t-m))

# Load projection matrix W
W = np.load("projection_w_r=9.dat")
plt.imshow(W, cmap=cm.gray)
# Load mean of train set
m = np.load("t_set-mean.npy")
print "mean", m

## construct classifier
#def classify(I, W, theta):
#    res_map = signal.correlate2d(I,W,mode='same', fillvalue=127)-m # zero-meaned
#    res_map = image_ops.normalize_array(res_map)
#    
#    # Smooth the response map
#    #res_map = ndimage.filters.gaussian_filter(res_map, sigma=2)
#    
#    fig = plt.figure()
#    a=fig.add_subplot(1,2,2)
#    plt.imshow(res_map,cmap=cm.gray)
#    a.set_title('Response map')    
#    ###    
#    a=fig.add_subplot(1,2,1)
#    plt.imshow(I,cmap=cm.gray)
#    a.set_title('Test image')
#    
#    Theta = np.ones(res_map.shape)*theta
#    return res_map>=Theta # y: boolean matrix of y(i,j)
#    #return 1 if np.dot(W,X) >= theta else -1
    
    
def classify(test_im, w, train_mean, theta, plot_intermediates=False):
    res_map = signal.convolve2d(test_im - train_mean, w, mode='same', fillvalue=127)
    res_map_normed = image_ops.normalize_array(res_map)
    #res_map_smoothed = ndimage.filters.gaussian_filter(res_map_normed, sigma=2.0)
#     res_map_smoothed = normalize_array(res_map_smoothed)
    coord_local_max = peak_local_max(res_map_normed, min_distance=20, threshold_abs=theta, exclude_border=False)
    
    loc_max_res_map = res_map[coord_local_max[:, 0], coord_local_max[:, 1].T]
#     print loc_max_res_map  # this is the value of the stuff
    
    if plot_intermediates:
        fig = plt.figure()
        a=fig.add_subplot(1,2,2)
        plt.imshow(res_map_smoothed,cmap=cm.gray)
        a.set_title('Response map')

        a=fig.add_subplot(1,2,1)
        plt.imshow(test_im,cmap=cm.gray)
        plt.plot(coord_local_max[:, 1], coord_local_max[:, 0], 'r.')
        a.set_title('Test image')

        plt.show()
    
    return (coord_local_max, 1 if len(coord_local_max) > 0 else -1)



def euc_dist(data1, data2):
    return la.norm(np.tile(data1, (data2.shape[0], 1))
                   - np.tile(data2, (data1.shape[0], 1)), axis=1).reshape(data2.shape[0], data1.shape[0])

def quality_measures(classify_res_coords, true_top_corner_coords, wiggle_room, true_rect_size=(40, 100)):
    assert len(classify_res_coords) == len(true_top_corner_coords), "mismatch"

    true_rect_center = np.array(true_rect_size)/2.0

    true_positive = 0
    true_negative = 0  # will always be zero
    false_positive = 0
    false_negative = 0

    for image_index in xrange(len(true_top_corner_coords)):
        res_coords = np.array(classify_res_coords[image_index])
        true_corner_coords = np.array(true_top_corner_coords[image_index])
        true_center_coords = true_corner_coords + np.tile(true_rect_center, (true_corner_coords.shape[0], 1))

        dist_betw_coords = euc_dist(true_center_coords, res_coords)
        dist_assessment = dist_betw_coords <= wiggle_room

        # with appropriately small wiggle_room and appropriate true_rect_size, each res_coord should only have one True
        dist_assess_per_res = np.sum(dist_assessment, axis=1)
        if sum(dist_assess_per_res) > res_coords.shape[0]:
            print "some points are closer to more than one true positives"
            print "wiggle room and/or true rect size not appropriate"
            print image_index
        else:
            true_positive += np.sum(dist_assess_per_res)
            false_positive += res_coords.shape[0] - np.sum(dist_assess_per_res)

        dist_assess_per_true = np.sum(dist_assessment, axis=0)
        if sum(dist_assess_per_true) > true_center_coords.shape[0]:
            print "more than one points are closer to a positve"
        else :
            false_negative += true_center_coords.shape[0] - np.sum(dist_assess_per_true)


    prec = true_positive / float(true_positive + false_positive)
    rec = true_positive / float(true_positive + false_negative)
    eer = prec == rec
    acc = true_positive + true_negative / float (true_positive + true_negative + false_positive + false_negative)

    return (prec, rec, eer, acc)
    
    
def quality_measures_train(calculated_labels, Np, Nn):  # I am gonna take the global Np and Nn
    true_positive = sum(calculated_labels[0:Np] == 1)
    true_negative = sum(calculated_labels[Np:] == -1)
    false_positive = sum(calculated_labels[Np:] == 1)
    false_negative = sum(calculated_labels[0:Np] == -1)
    
    prec = true_positive / float(true_positive + false_positive)
    rec = true_positive / float(true_positive + false_negative)
    eer = prec == rec
    acc = true_positive + true_negative / float (true_positive + true_negative + false_positive + false_negative)
    
    return (prec, rec, eer, acc)

        


""" Import and classify the test data """

prec = []
rec = []
eer = []
acc = []
thetas = np.linspace(130, 240 + 1, 10)

for theta in thetas:
    calc_label = []
    for train_im in t_set:
        calc_label.append(classify(train_im, W, m, theta)[1])
#         print calc_label
    
    [prec_, rec_, eer_, acc_] = quality_measures_train(np.array(calc_label), Np, Nn)
    prec.append(prec_)
    rec.append(rec_)
    eer.append(eer_)
    acc.append(acc_)
    print theta
 

#file_uiuc_trueLocs = "CarData/trueLocations.txt"
#test_true_locs = []
#with open(file_uiuc_trueLocs, "r") as f:
#    for line in f.readlines():
#        test_true_locs.append(ast.literal_eval(line)[1:])   
#    
#prec = []
#rec = []
#eer = []
#acc = []
#
#for theta in np.linspace(int(170), int(240) + 1, 10):
#    res_coords = []
#    for test_im in uiucTest:
#        t_img = np.array(Image.open(test_im).convert('L'))
#        res_coords.append(classify(t_img, W, m, theta))
#    
#    [prec_, rec_, eer_, acc_] = quality_measures(res_coords, test_true_locs, 10)
#    prec.append(prec_)
#    rec.append(rec_)
#    eer.append(eer_)
#    acc.append(acc_)
#
#print prec, rec, eer, acc
    
    