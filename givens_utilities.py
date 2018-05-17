import numpy as np
from scipy import optimize


def givens_rotation(i, j, theta, dim):

    g = np.eye(dim)
    c = np.cos(theta)
    s = np.sin(theta)

    g[i, i] = c
    g[j, j] = c

    if i > j:
        g[i, j] = s
        g[j, i] = -s
    elif i < j:
        g[i, j] = -s
        g[j, i] = s
    else:
        raise ValueError('require i, j to be different, got i=%d and j=%d'%(i,j))

    return g


def givens_rotation_gradient(i, j, theta, dim):
    """
    :param i: int, i-th index
    :param j: int, j-th index
    :param theta: float, array
    :param dim: int, size of array
    :return: array of size dimXdim
    """
    grad_g = np.zeros((dim, dim))
    c = np.cos(theta)
    s = np.sin(theta)

    grad_g[i, i] = -s
    grad_g[j, j] = -s

    if i > j:
        grad_g[i, j] = c
        grad_g[j, i] = -c
    elif i < j:
        grad_g[i, j] = -c
        grad_g[j, i] = c
    else:
        raise ValueError('require i, j to be different, got i=%d and j=%d'%(i,j))

    return grad_g


def map_ij_to_k(c):
    """
    c = number of clusters
    k_ij is the kth entry of a lexicographical list of
    (i,j) in {1,2.....c}^2 pairs with i < j
    """

    k_ij = [(i, j) for i in range(c) for j in range(c) if i < j]

    return k_ij


def get_Gk(k, k_ij, theta, dim):

    theta_k = theta[k]
    i = k_ij[k][0]
    j = k_ij[k][1]
    
    return givens_rotation(i, j, theta_k, dim)


def get_Uab(a, b, k_ij, theta, dim, K):

    Uab = np.eye(dim)
    if a == b:
        if a < K and a != 0:
            Uab = get_Gk(a, k_ij, theta, dim)

    elif a < b:
        for i in range(a,b):
            Uab = np.dot(Uab,get_Gk(i, k_ij, theta, dim))

    return Uab


def get_Uk(k, k_ij, theta, dim, K):

    return get_Uab(k, k, k_ij, theta, dim, K)


def get_Vk(k, k_ij, theta, dim):

    return givens_rotation_gradient(k_ij[k][0], k_ij[k][1], theta[k], dim)


def get_Ak(X, k_ij, theta, k, C):
    """
    Ak = X U_(1,k-1) V_k U_(k+1,K)
    where U_(a,b) = G_a,theta_a G_a+1,theta_a+1....G_b,theta_b
    :param X: float, ndarray
    :param k_ij: int, list
    :param theta: float, ndarray
    :param k: int
    :param K: int
    :return: float, ndarray
    """

    K = len(k_ij)
    Ul = get_Uab(0, k, k_ij, theta, C, K)
    Ur = get_Uab(k+1, K, k_ij, theta, C, K)
    Vk = get_Vk(k, k_ij, theta, C)

    Ak = np.dot(X, np.dot(Ul, np.dot(Vk, Ur)))

    return Ak


def cost_func(theta, *args):

    k_ij = args[0]
    C = args[1]
    X = args[2]
    K = int(C * (C - 1) / 2)
    n = X.shape[0]

    U = np.eye(C)
    for k in range(K):
        G_k = get_Gk(k, k_ij, theta, C)
        U = np.dot(U, G_k)

    # rotated matrix Z = XR
    Z = np.dot(X, U)
    # get maximum element and its index along each row
    Mi = np.max(Z, axis=1)
    mi = np.argmax(Z, axis=1)
    # compute cost function J = sum_i sum_j Z_ij **2 /Mi**2
    J = np.sum(np.sum(([Z[i, :]**2/Mi[i]**2 for i in range(n)])))

    # compute gradient as a function of theta
    gradJ = np.zeros((K,))

    for k in range(K):
        Ak = get_Ak(X, k_ij, theta, k, C)
        ZAk = Z*Ak
        for i in range(n):
            Mik = ZAk[i, mi[i]]
            for j in range(C):
                gradJ[k] += 2.*(ZAk[i, j]/Mi[i]**2 - Mik*Z[i, j]**2/Mi[i]**3)

    return J, gradJ


def aligning_rotation(X, C):
    """
    :param X: float, ndarray
    :param C: int
    :return: optimized object
             rotated matrix Z = XR
    """

    k_ij = map_ij_to_k(C)
    K = len(k_ij)
    theta_init = np.zeros((K,))
    arguments = (k_ij, C, X)

    all_fx = []
    all_dfx = []

    def store(x):

        fx, dfx = cost_func(x, *arguments)
        all_fx.append(fx)
        all_dfx.append(dfx)

    bnds = ((-np.pi/2, np.pi/2),)*K
    opt = optimize.minimize(cost_func, theta_init, arguments, method='L-BFGS-B', jac=True,\
                            bounds=bnds, options={'disp': True}, callback=store)

    # get the rotation matrix corresponding to optimized theta
    R = np.eye(C)
    for k in range(K):
        gk = get_Gk(k, k_ij, opt.x, C)
        R = np.dot(R, gk)

    Z = np.dot(X, R)

    return Z, opt.x, all_fx, all_dfx


def example_solve(x_min, x_max):

    def func(xx):

        cost = lambda xx: xx**2 - xx - 2
        grad = lambda xx: 2*xx -1
        return cost, grad

    opt = optimize.root(func,np.array([x_min, x_max]), jac=True) #,options={'disp':False})

    return opt.x