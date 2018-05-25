import numpy as np
from scipy.linalg import eigh, eig
from scipy.spatial.distance import pdist, squareform


def get_affinity(data, **kwargs):

    keys = kwargs.keys()

    allowed_metrics = ['euclidean', 'cosine']
    if 'metric' in keys:
        metric = kwargs['metric']
    else:
        metric = 'euclidean'
    if metric not in allowed_metrics:
        metric = 'euclidean'

    if len(data.shape) == 2:
        n, m = data.shape
    elif len(data.shape) == 3:
        data = data.reshape(data.shape[0], -1)
        n, m = data.shape

    # get nXn distance matrix
    dist_matrix = squareform(pdist(data, metric=metric))

    # if no value for sigma is given, then compute sigma
    # based on input sigma_ref (default=7)
    sigma_in = None
    sigma_ref = None
    if 'sigma' in keys:
        sigma_in = kwargs['sigma']
        if sigma_in <= 0.:
            sigma_in = None

    if 'sigma_ref' in keys:
        sigma_ref = kwargs['sigma_ref']

    scaled_dist = np.zeros_like(dist_matrix)
    if sigma_in is None:
        if sigma_ref is None:
            sigma_ref = 7
        sigma_ind = np.argsort(dist_matrix,axis=1)[:,sigma_ref]
        sigma = np.array([dist_matrix[i, sigma_ind[i]] for i in range(dist_matrix.shape[0])])
        for i in range(n):
            sigma_ij = sigma[i]*sigma[:]
            scaled_dist[i,:] = -dist_matrix[i,:]**2/sigma_ij[:]
    else:
        scaled_dist = -dist_matrix**2/sigma_in**2

    # 0./0
    scaled_dist[np.where(np.isnan(scaled_dist))] = -10000.

    # affinity = exp(-d**2), and set diagonal elements to 0.
    affinity = np.exp(scaled_dist)
    for i in range(n):
        affinity[i,i] = 0.

    return affinity


def get_lap_eig(affinity):

    eps = np.finfo(np.float64).eps

    # make sure there are no NaNs
    affinity = np.nan_to_num(affinity)

    # symmetrize affinity
    a = np.tril(affinity, k=-1)
    aff = a + a.transpose()

    # compute diagonal matrix with sum of affinities
    d = np.diag(np.sum(aff, axis=0))
    inv_sqrtd = np.diag([1./np.sqrt(d[i, i]) if d[i, i] > eps else 0. for i in range(d.shape[0])])

    lap = np.dot(inv_sqrtd, np.dot(aff, inv_sqrtd))
    eigval, eigvec = eigh(lap)

    return eigval, eigvec
