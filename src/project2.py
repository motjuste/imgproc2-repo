from PIL import Image
import random
from sets import Set
import glob
import numpy as np
import matplotlib.pyplot as plt

Xtrain_file_list=[]
Xtest_file_list=[]
v=[]
l=[]
Xtrain=np.array
Xtest=np.array

filenames = glob.glob('../resources/train/*.pgm')

"""2.1.2 Get randomly 90% of the files and 10% and saving them into respective lists"""
Xtrain_file_list= random.sample(filenames, 2186)   
for x in filenames:
    if x not in Xtrain_file_list: 
        Xtest_file_list.append(x) 

"""2.1.3 reading the data into the respective array"""   
images_Xtrain = [Image.open(fn).convert('L') for fn in Xtrain_file_list]
Xtrain = np.asarray([np.array(im).flatten() for im in images_Xtrain])

images_Xtest = [Image.open(fn).convert('L') for fn in Xtest_file_list]
Xtest = np.asarray([np.array(im).flatten() for im in images_Xtest])

       
""" 2.1.3 Calculate the mean from the Xtrain Matrix and subtract it"""
meanImage = Xtrain.mean(axis=0)
shiftedImages = Xtrain - meanImage

""" 2.1.4 computing the covariance matrix C"""
c  = np.asmatrix(shiftedImages) * np.asmatrix(shiftedImages.T)

""" 2.1.4 computing the eigenvectors vi and eigenvalues li """
lamda,v =np.linalg.eig(c)
 
"""2.1.5 descenting sorting the eigenvalues li  and plot the C spectrum""" 
idx = np.argsort(-lamda)   
lamda = lamda[idx]
v = v[:, idx]
 
 
#tried to plot the eigenvalue spectrum, but it s not working. 
plt.plot(abs(c))
plt.show()
  
  
 