from PIL import Image
import random
from sets import Set
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import distance

Xtrain_file_list=[]
Xtest_file_list=[]
v=[]
l=[]
Xtrain=np.array
Xtest=np.array


filenames = glob.glob('../resources/train/*.pgm')

"""2.1.2 Get randomly 90% of the files and 10% and saving them into respective lists"""
Xtrain_file_list= random.sample(filenames, int(len(filenames)*0.9))   
for x in filenames:
    if x not in Xtrain_file_list: 
        Xtest_file_list.append(x) 

"""2.1.3 Read the training data into its array """   
images_Xtrain = [Image.open(fn).convert('L') for fn in Xtrain_file_list]
Xtrain = np.asarray([np.array(im).flatten() for im in images_Xtrain])

       
""" 2.1.3 Calculate the mean from the Xtrain Matrix and subtract it"""
meanImage = Xtrain.mean(axis=0)
shiftedImages = Xtrain - meanImage

""" 2.1.4 computing the covariance matrix C"""
c = np.asmatrix(shiftedImages) * np.asmatrix(shiftedImages.T)

""" 2.1.4 computing the eigenvectors vi and eigenvalues li """
eVals,eVecs = np.linalg.eig(c)

"""2.1.5 descending sorting of eigenvalues and plotting of the C spectrum""" 
idx = np.argsort(-eVals)   
desc_eVals = eVals[idx]
desc_eVecs = eVecs[:, idx]

# Get rid of imaginary parts, there might be a way to avoid them (see Lec10).
desc_eVals = abs(desc_eVals)
 
""" Plot sorted eigenvalues """
plt.plot(desc_eVals, 'b', label='Spectrum of C')
plt.axis([0, len(desc_eVals)/4, 0, 5e7])
plt.legend()
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue')
plt.show()

""" Estimate a suitable k """
expecAcc = 0.9 # expected accuracy
sumTotal = np.sum(desc_eVals)
sumK = 0
k = 0
for k in range(0, len(desc_eVals)-1):
    sumK += desc_eVals[k]
    if sumK/sumTotal >= 0.9:
        break

# show k on the plot
plt.plot(desc_eVals, 'b', label='Spectrum of C')
plt.plot(k,desc_eVals[k], "ro", label='k')
#plt.annotate(str(k),xy=(desc_eVals[k],k))
plt.axis([0, len(desc_eVals)/4, 0, 5e7])
plt.legend()
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue')
plt.show()
 
""" Visualize the first k eigenvectors """
arr = abs(desc_eVecs) # get rid of imaginary parts
arr *= 255.0/arr.max() # normalize the image
#print(arr.shape)
#print(arr)
#plt.imshow(arr, cmap = cm.Greys_r)

""" Read the test data into its array, center them """
images_Xtest = [Image.open(fn).convert('L') for fn in Xtest_file_list]
Xtest = np.asarray([np.array(im).flatten() for im in images_Xtest])
shiftedTests = Xtest - meanImage

""" Randomly select 10 images, Compute Euc. distances to training images, plot distances in desc. order """
samples = 10
Xtrain10= random.sample(shiftedTests, samples)
dists = np.zeros(samples) # distances
for s in range(0, samples): # can be vectorized?
    dists[s] = distance.euclidean_distance(Xtrain10[s], meanImage) 

# Sort in descending order
sortedDists = dists[np.argsort(-dists)]
# Plot the distances
plt.plot(sortedDists, 'b')
plt.ylabel('Euclidean Distance')
plt.xlabel('Test image #')
plt.show()