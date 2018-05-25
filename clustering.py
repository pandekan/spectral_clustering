import numpy as np
import givens_utilities
import graph_utilities
from sklearn import cluster


def data_to_eig(data, **kwargs):

    keys = kwargs.keys()
    dist_metric = None
    sigma_ref = None
    sigma_in = None

    if 'metric' in keys:
        dist_metric = kwargs['metric']

    if 'sigma_ref' in keys:
        sigma_ref = kwargs['sigma_ref']

    if 'sigma_in' in keys:
        sigma_in = kwargs['sigma_in']

    affinity = graph_utilities.get_affinity(data, metric=dist_metric, sigma_ref=sigma_ref, sigma=sigma_in)
    eigval, eigvec = graph_utilities.get_lap_eig(affinity)

    return eigval, eigvec


def spectral_clustering(eigvec, nclusters):

    # matrix of top n eigenvalues
    X = eigvec[:,-nclusters:]

    # normalize eigenvectors along rows
    Xnorm = np.zeros_like(X)
    for i in range(X.shape[0]):
        Xnorm[i,:] = X[i,:]/np.sqrt(np.sum(X[i,:]**2))

    kmeans = cluster.KMeans(nclusters)
    cluster_index = kmeans.fit_predict(Xnorm)

    return cluster_index


def self_tuning_sc(eigval, eigvec, **kwargs):

    keys = kwargs.keys()

    if 'nclusters_min' in keys:
        nclusters_min = kwargs['nclusters_min']
    else:
        nclusters_min = 2

    if 'nclusters_max' in keys:
        nclusters_max = kwargs['nclusters_max']
    else:
        nclusters_max = np.sum(eigval > 0.)
        if nclusters_max < 2:
            nclusters_max = 2

    if nclusters_max < nclusters_min:
        raise ValueError('maximum number of clusters %d less than minimum nunber of cluster %d'
                         %(nclusters_max, nclusters_min))

    rot = []
    cost = []
    for c in range(nclusters_min, nclusters_max+1):
        x = eigvec[:, -c:]
        Z, theta, fx, dfx = givens_utilities.aligning_rotation(x, c)
        cost.append(fx[-1])
        rot.append(Z)

    chosen_rot_ind = np.argmin(cost)
    chosen_rot = rot[chosen_rot_ind]
    nlabels = chosen_rot.shape[1]

    km = cluster.KMeans(nlabels)
    labels_kmeans = km.fit_predict(chosen_rot)
    labels = np.argmax(chosen_rot, axis=1)

    return labels_kmeans, labels
