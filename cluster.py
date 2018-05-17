import numpy as np
import givens_utilities
import graph_utilities


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

    return cost, rot


def cluster_data(cost, rot):

    chosen_rot_ind = np.argmin(cost)
    chosen_rot = rot[chosen_rot_ind]
    nlabels = chosen_rot.shape[1]
    labels = np.argmax(chosen_rot, axis=1)

    return nlabels, labels



