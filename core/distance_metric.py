import numpy as np
from numba import njit

from core.geometry import local_basis

@njit
def compute_radial_penalty(s, sigma_r, beta=0.1):
    val = sigma_r * np.exp(-beta * s)
    return max(val, 1e-3)

@njit
def uncertaintyDistance(pA, vA_obs, pB, vB_obs, sigma_r=3, sigma_t=0.8, use_speed_dependent_radial=True, beta=0.1):
    basisA = local_basis(pA, vA_obs)
    basisB = local_basis(pB, vB_obs)

    diff = vA_obs - vB_obs

    # Project the difference into local bases
    cA = np.dot(basisA.T, diff)
    cB = np.dot(basisB.T, diff)

    if use_speed_dependent_radial:
        sA = np.sqrt(np.sum(vA_obs**2))
        sB = np.sqrt(np.sum(vB_obs**2))
        sigma_r_A = compute_radial_penalty(sA, sigma_r, beta=beta)
        sigma_r_B = compute_radial_penalty(sB, sigma_r, beta=beta)
    else:
        sigma_r_A = sigma_r_B = sigma_r

    inv_cov_A = np.array([1.0 / sigma_r_A**2, 1.0 / sigma_t**2, 1.0 / sigma_t**2])
    inv_cov_B = np.array([1.0 / sigma_r_B**2, 1.0 / sigma_t**2, 1.0 / sigma_t**2])

    distA = np.sqrt(np.sum(cA**2 * inv_cov_A))
    distB = np.sqrt(np.sum(cB**2 * inv_cov_B))

    return 0.5 * (distA + distB)

@njit
def compute_expected_distance_matrix_from_uncertainty(positions, velocities, sigma_r=3.0, sigma_t=1.0, use_speed_dependent_radial=True, beta=0.1):
    n = positions.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = uncertaintyDistance(positions[i], velocities[i], positions[j],
                                       velocities[j], sigma_r=sigma_r, sigma_t=sigma_t, use_speed_dependent_radial=use_speed_dependent_radial, beta=beta)
            D[i, j] = D[j, i] = dist
    return D