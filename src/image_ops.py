import numpy as np

def normalize_array(array, new_min=0, new_max=255):
    #assert len(array.shape) == 2, "Only two dimensional arrays supported"  # TODO: @motjuste Why?
    return new_min + \
            ((array - array.min()) * ((new_max - new_min) / (array.max() - array.min())))