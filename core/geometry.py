import numpy as np
from numba import njit
from scipy.spatial.distance import cosine

@njit
def euclideanDistance(vA, vB):
    vA = vA.astype(np.float64)
    vB = vB.astype(np.float64)
    return np.linalg.norm(vA - vB)


def safe_cosine(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0
    return cosine(u, v)

@njit
def radial_vector(position):
    norm = np.sqrt(np.sum(position**2))
    if norm == 0.0:
        return np.zeros(3)
    return position / norm


def project_to_tangential(position, velocity):
    r_hat = radial_vector(position)
    v_radial = np.dot(velocity, r_hat) * r_hat
    v_tangential = velocity - v_radial
    return v_tangential


@njit
def local_basis(position, tangential_velocity):
    r_hat = radial_vector(position)
    
    fallback_z = np.array([0.0, 0.0, 1.0])
    fallback_x = np.array([1.0, 0.0, 0.0])

    norm_vtan = np.sqrt(np.sum(tangential_velocity**2))

    if norm_vtan > 1e-8:
        v_tan_hat = tangential_velocity / norm_vtan
        t2 = np.cross(r_hat, v_tan_hat)
        t2_norm = np.sqrt(np.sum(t2**2))
        
        if t2_norm < 1e-8:
            fallback = fallback_z
            if abs(np.dot(r_hat, fallback)) > 0.99:
                fallback = fallback_x
            
            t1 = np.cross(r_hat, fallback)
            t1 = t1 / np.sqrt(np.sum(t1**2))
            t2 = np.cross(r_hat, t1)
        else:
            t2 = t2 / t2_norm
            t1 = np.cross(t2, r_hat)
    else:
        fallback = fallback_z
        if abs(np.dot(r_hat, fallback)) > 0.99:
            fallback = fallback_x
        t1 = np.cross(r_hat, fallback)
        t1 = t1 / np.sqrt(np.sum(t1**2))
        t2 = np.cross(r_hat, t1)

    basis = np.zeros((3, 3))
    basis[:, 0] = r_hat
    basis[:, 1] = t1
    basis[:, 2] = t2
    return basis