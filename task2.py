import numpy as np
import src.image_io as image_io
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def image_as_array(image):
    assert image.mode in ['L', '1'], "Only Greyscale and Binary images supported"  # TODO: @motjuste Support Color Images?
    return np.asarray(image.getdata()).reshape((image.size[1], image.size[0]))


def import_image(from_location, as_array=False):
    try:
        image = Image.open(from_location)
    except IOError:
        raise IOError  # right now only checking IOError

    if as_array:
        return image_as_array(image)
    else:
        return image

print ""

im = image_io.import_image(raw_input("Full Path to Image: "), as_array=True)  # Will ask for path
ln_im = np.log(im + 1)

def func(X, a, b, c):
    return a*X[:,0] + b*X[:,1] + c

xdata = np.array([(x, y) for y in xrange(ln_im.shape[0]) for x in xrange(ln_im.shape[1])])

res = curve_fit(func, xdata, np.ravel(ln_im))

est = func(xdata, res[0][0], res[0][1], res[0][2])
est_ = est.reshape(ln_im.shape)

# Plotting
X, Y = np.meshgrid(np.arange(est_.shape[1]), np.arange(est_.shape[0]))

fig = plt.figure()
axs = Axes3D(fig)
axs.plot_wireframe(X, Y, ln_im, rstride=20, cstride=20, color='b')
axs.plot_wireframe(X, Y, est_, rstride=20, cstride=20, color='r')
plt.show()
