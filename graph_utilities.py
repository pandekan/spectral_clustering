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

    sigma_in = None
    sigma_ref = None
    if 'sigma' in keys:
        sigma_in = kwargs['sigma']
        if sigma_in <= 0.:
            sigma_in = None

    if 'sigma_ref' in keys:
        sigma_ref = kwargs['sigma_ref']

    sigma = np.zeros((n,))
    if sigma_in is None:
        if sigma_ref is None:
            sigma_ref = n/2
        for i in range(n):
            ind = np.argsort(dist_matrix[i, :])
            sigma[i] = dist_matrix[i, ind[sigma_ref]]
            if sigma[i] <= np.finfo(np.float64).eps:
                sigma[i] = dist_matrix[i, ind[-1]]
    else:
        sigma = np.array([sigma_in]*n)

    affinity = np.zeros((n, n))
    for i in range(n):
        sij = sigma[i]*sigma[:]
        affinity[i, :] = np.exp(-(dist_matrix[i, :]**2/sij[:]))

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
