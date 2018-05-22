import numpy as np
import givens_utilities


def cost_func(theta, *args):

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
            Mik = ZAk[i, mi[i]]
            for j in range(C):
                gradJ[k] += 2.*(ZAk[i, j]/Mi[i]**2 - Mik*Z[i, j]**2/Mi[i]**3)

    return J, gradJ


def aligning_rotation(X, C, max_iter=1000):
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

    alpha = 0.5
    iter = 0
    theta_new = theta.copy()
    J, gradJ = cost_func(theta, *arguments)
    J_old_1 = J.copy()
    J_old_2 = J.copy()

    while( iter < max_iter):
        iter += 1
        for k in range(K):
            theta_new[k] = theta[k] - alpha*gradJ[k]
            J_new, gradJ_new = cost_func(theta_new,*arguments)
            if J_new < J:
                theta[k] = theta_new[k]
                J = J_new
                gradJ = gradJ_new.copy()
            else:
                theta_new[k] = theta[k]

        if iter > 2:
            if J - J_old_2 < 1.e-3:
                break
        J_old_2 = J_old_1
        J_old_1 = J


    all_fx = []
    all_dfx = []

    def store(x):

        fx, dfx = cost_func(x, *arguments)
        all_fx.append(fx)
        all_dfx.append(dfx)

    theta_bounds = ((-np.pi/2, np.pi/2),)*K
    opt = optimize.minimize(cost_func, theta_init, arguments, method='L-BFGS-B', jac=True,\
                            bounds=theta_bounds, options={'disp': True}, callback=store)

    # get the rotation matrix corresponding to optimized theta
    R = np.eye(C)
    for k in range(K):
        gk = get_Gk(k, k_ij, opt.x, C)
        R = np.dot(R, gk)

    Z = np.dot(X, R)

    return Z, opt.x, all_fx, all_dfx