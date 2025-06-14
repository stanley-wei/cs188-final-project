import numpy as np

def to_cross_matrix(v):
    mat = np.asarray([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return mat

def from_cross_matrix(A):
    v = np.asarray([
        A[2, 1],
        A[0, 2],
        A[1, 0]
    ])
    return v
    

def log_so3(R):
    if np.allclose(R, np.eye(3)):
        return np.zeros(3)
    
    theta = np.arccos(0.5 * (np.trace(R) - 1))
    n = np.asarray([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    return theta * n
