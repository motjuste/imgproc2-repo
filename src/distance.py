import numpy as np

def euclidean_distance(start_point, end_point):
    assert len(start_point) == len(end_point), "n-dimension mismatch"

    start_point = np.array(start_point)
    end_point = np.array(end_point)

    return np.linalg.norm(end_point - start_point)


