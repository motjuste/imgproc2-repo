from src.image_io import import_image
from src.plot import plot_multiple_arrays

im = import_image("resources/cat.png", as_array=True)

plot_multiple_arrays([[im]], "Project 1 Demo", ["Input Image"])