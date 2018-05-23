import numpy as np
import givens_utilities
from scipy import optimize


def compute_J(theta, *args):

    k_ij = args[0]
    C = args[1]
    X = args[2]
    K = int(C*(C-1)/2)

    n = X.shape[0]
    U = np.eye(C)
    for k in range(K):
        G_k = givens_utilities.get_Gk(k, k_ij, theta, C)
        U = np.dot(U, G_k)

    # rotated matrix Z = XR
    Z = np.dot(X, U)
    # get maximum element and its index along each row
    Mi = np.max(Z, axis=1)
    mi = np.argmax(Z, axis=1)
    # compute cost function J = sum_i sum_j Z_ij **2 /Mi**2
    J = np.sum(np.sum(([Z[i, :] ** 2 / Mi[i] ** 2 for i in range(n)])))

    return Z, J


def compute_gradJk(theta, k_in, *args):

    k_ij = args[0]
    C = args[1]
    X = args[2]
    K = int(C * (C - 1) / 2)

    n = X.shape[0]
    U = np.eye(C)
    for k in range(K):
        G_k = givens_utilities.get_Gk(k, k_ij, theta, C)
        U = np.dot(U, G_k)

    # rotated matrix Z = XR
    Z = np.dot(X, U)
    Mi = np.max(Z, axis=1)
    mi = np.argmax(Z, axis=1)
    Ak = givens_utilities.get_Ak(X, k_ij, theta, k_in, C)
    # pixel-wise multiplication between rotated matrix and gradient matrix
    ZAk = Z*Ak

    gradJk = 0.
    for i in range(n):
        Mik = Ak[i,mi[i]]
        Mi2 = Mi[i]**2
        Mi3 = Mi[i]**3
        for j in range(C):
            gradJk += 2.*(ZAk[i,j]/Mi2 - Mik*Z[i, j]**2/Mi3)

    return gradJk


def compute_J_gradJ(theta, *args):

    k_ij = args[0]
    C = args[1]
    X = args[2]
    K = int(C * (C - 1) / 2)
    n = X.shape[0]

    U = np.eye(C)
    for k in range(K):
        G_k = givens_utilities.get_Gk(k, k_ij, theta, C)
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
        Ak = givens_utilities.get_Ak(X, k_ij, theta, k, C)
        ZAk = Z*Ak
        for i in range(n):
            Mik = Ak[i, mi[i]]
            for j in range(C):
                gradJ[k] += 2.*(ZAk[i, j]/Mi[i]**2 - Mik*Z[i, j]**2/Mi[i]**3)

    return J, gradJ


def aligning_rotation_CG0(X, C, alpha=0.1, max_iter=1000):
    """
    :param X: float, ndarray
    :param C: int
    :return: optimized object
             rotated matrix Z = XR
    """

    k_ij = givens_utilities.map_ij_to_k(C)
    K = len(k_ij)
    theta = np.zeros((K,))
    arguments = (k_ij, C, X)
    print k_ij, K

    #alpha = 0.1
    iter = 0
    theta_new = theta.copy()
    Z, J = compute_J(theta, *arguments)

    J_old_1 = J
    J_old_2 = J

    while( iter < max_iter):
        iter += 1
        for k in range(K):
            gradJk = compute_gradJk(theta, k, *arguments)
            theta_new[k] = theta[k] - alpha * gradJk
            Z_new, J_new = compute_J(theta_new,*arguments)
            gradJk_new = compute_gradJk(theta_new, k, *arguments)
            print iter, k, J, J_new, gradJk, gradJk_new, theta[k], theta_new[k]
            if J_new < J:
                theta[k] = theta_new[k]
                J = J_new
            else:
                theta_new[k] = theta[k]

        if iter > 2:
            if J - J_new < 1.e-6:
                break

        #    J_old_2 = J_old_1
        #    J_old_1 = J

    Z_out, J_out = compute_J(theta_new, *arguments)

    return theta_new, Z_out, J_out

def aligning_rotation_CG(X, C, alpha=0.1, max_iter=1000):
    """
    :param X: float, ndarray
    :param C: int
    :return: optimized object
             rotated matrix Z = XR
    """

    k_ij = givens_utilities.map_ij_to_k(C)
    K = len(k_ij)
    theta = np.zeros((K,))
    arguments = (k_ij, C, X)
    print k_ij, K

    #alpha = 0.1
    iter = 0
    theta_new = theta.copy()

    J, gradJ = compute_J_gradJ(theta,*arguments)
    J_old_1 = J
    J_old_2 = J

    while( iter < max_iter):
        iter += 1
        for k in range(K):
            theta_new[k] = theta[k] - alpha*gradJ[k]
            J_new, gradJ_new = compute_J_gradJ(theta_new,*arguments)
            print iter, k, J, J_new, gradJ[k], gradJ_new[k], theta[k], theta_new[k]
            if J_new < J:
                theta[k] = theta_new[k]
                J = J_new
                gradJ = gradJ_new
            else:
                theta_new[k] = theta[k]

        #if iter > 2:
        #    if J - J_old_2 < 1.e-6:
        #        break

        #    J_old_2 = J_old_1
        #    J_old_1 = J

    Z_out, J_out = compute_J(theta_new, *arguments)

    return theta_new, Z_out, J_out


def aligning_rotation(X, C):
    """
    :param X: float, ndarray
    :param C: int
    :return: optimized object
             rotated matrix Z = XR
    """

    k_ij = givens_utilities.map_ij_to_k(C)
    K = len(k_ij)
    theta = np.zeros((K,))
    arguments = (k_ij, C, X)

    all_fx = []
    all_dfx = []

    def store(x):
        fx, dfx = compute_J_gradJ(x, *arguments)
        all_fx.append(fx)
        all_dfx.append(dfx)

    theta_bounds = ((-np.pi / 2, np.pi / 2),) * K
    opt = optimize.minimize(compute_J_gradJ, theta, arguments, method='L-BFGS-B', jac=True, \
                            bounds=theta_bounds, options={'disp': True}, callback=store)

    # get the rotation matrix corresponding to optimized theta
    R = np.eye(C)
    for k in range(K):
        gk = givens_utilities.get_Gk(k, k_ij, opt.x, C)
        R = np.dot(R, gk)

    Z = np.dot(X, R)

    return Z, opt.x, all_fx, all_dfx