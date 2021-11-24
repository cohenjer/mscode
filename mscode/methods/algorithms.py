import numpy as np
from itertools import combinations
from mscode.methods.proxs import prox_ml1,prox_ml1_fast
from mscode.methods.proxs import SoftT
from mscode.methods.proxs import ml1
from mscode.methods.proxs import HardT
from mscode.methods.proxs import ls_kn_supp, ls_cg
from scipy.linalg import svdvals
import time

def admm_mix(Y, D, B, k, X0=None, itermax=1000, tol=1e-6, verbose=True, rho=None):
    '''
    Solves (approximatively, without guaranties) the mixed sparse coding problem using ADMM with hard thresholding as the proximity operator of the l0 sparsity constraint. The problem is formulated as

        :math:`\\min_X\; \\|Y - DXB\\|_F^2  \; s.t.\;   \\|X_i\\|_0 \\leq k`

    where k is the maximal number of nonzeros per column of X.

    Parameters
    ----------
    Y : numpy array
        input data, required

    D : numpy array
        input dictionary (fat), required

    B : numpy array
        input mixing matrix, required

    k : integer
        sparsity level per column, thresholded at the end, required

    rho : float
        augmented Lagrangian penality coefficient, by default it is set to ??

    Returns
    ----------
    X : numpy array
        estimated X

    e : list
        fittings along iterations

    support : numpy array
        the support of each column of X
    '''

    # Copy input
    X = np.copy(X0)

    # Input caracteristics
    n, d = D.shape
    m, r = B.shape

    # Store DtD, BtB and DtYB if possible
    DtD = D.T@D
    Bt = B.T
    BtB = Bt@B
    DtYB = D.T@Y@B

    # Initialisation of coefficients x
    # this does not matter too much since the problem is convex
    if X0 is None:
        X = np.zeros([d, r])

    # Choice of rho, use Lipschitz constant
    singvalD = np.linalg.svd(DtD)[1][0]
    singvalB = np.linalg.svd(BtB)[1][0] #why 2?
    eta = 1/singvalD/singvalB
    if rho is None:
        rho = (np.linalg.norm(D, 'fro')*np.linalg.norm(B, 'fro'))**2/(n*r)
        if verbose:
            print('The automatic value of rho is ', rho)  # heuristic based on optimal quadratic rho, here the average of squared singular values of the mixing matrix

    # Initial error
    e0 = (np.linalg.norm(Y - D@X@Bt, 'fro') ** 2)/2
    e = [np.Inf, e0]
    err_Z = [np.Inf]

    # Initial iteration count
    iter = 0

    # Main print
    if verbose:
        print('ADMM mix running\n')

    # Main loop

    # pairing and dual variables for ADMM
    Z = np.copy(X)
    nu = np.zeros([d,r])

    while iter < itermax:

        if np.abs(e[-1] - e[-2])/e[-1] < tol and np.abs(err_Z[-1] - err_Z[-2]) < 1e-2:
            break

        # printing
        if iter % 10 == 1:
            if verbose:
                print('ADMM iteration ', iter, ' cost ', e[-1])
            #else:
            #    print('.', end='')

        iter += 1

        # Prox wrt X
        rhs = DtYB + rho * (Z - nu)
        X, _ = ls_cg(rhs, DtD, BtB, X, rho, itercg = 50)  # solves linear system wrt X
        # Prox wrt Z
        Z = HardT(X + nu, k)
        # Gradient ascent step
        nu = nu + X - Z

        # error computation
        rec = np.linalg.norm(Y - D@X@Bt, 'fro')
        e.append(rec**2/2)
        err_Z.append(np.linalg.norm(X - Z, 'fro')/np.linalg.norm(X, 'fro'))

    e = e[1:]
    err_Z = err_Z[1:]

    # Get k largest entries per columns
    # Estimating support (np arrays with supports)
    support = np.argsort(np.abs(X), 0)
    # truncating the support
    support = support[-k:, :]

    # Post-processing option for k-sparse columnwise
    # Running least squares
    X, _ = ls_kn_supp(Y, D, B, support, k)

    # error computation
    rec = np.linalg.norm(Y - D@X@Bt, 'fro')
    e.append(rec**2/2)

    if verbose:
        print('\n')

    return X, e, support, Z, err_Z

def ista_mix(Y, D, B, lamb_rel, k=None, X0=None, itermax=1000, tol=1e-6, verbose=True):
    '''
    Solves the tighest convex relaxation of the mixed sparse coding problem using Fast Iterative Soft Thresholding (Fista).
    The cost function is

        :math:`\\frac{1}{2} \\|Y - DXB^T \\|_F^2 + \\lambda \\|X\\|_{1,1}`

    where :math:`\\lambda = \\lambda_{rel}\\lambda_{\\max}`

    Parameters
    ----------
    Y : numpy array
        input data, required

    D : numpy array
        input dictionary (fat), required

    B : numpy array
        input mixing matrix, required

    lamb_rel : float
        ratio of lambda_max used as a regularization, required

    k : integer
        sparsity level per column, thresholded at the end

    X0 : numpy array
        initial estimation of X

    itermax : integer (default 1000)
        maximum number of proximal iterations

    tol : float (default 1e-6)
        relative error threshold for stopping the algorithm

    verbose : boolean (default True)
        Set to False to remove prints

    Returns
    ----------
    X : numpy array
        estimated X

    e : list
        fittings along iterations

    rec : list
        reconstruction errors along iterations

    support : numpy array
        the support of each column of X
    '''

    # Copy input
    X = np.copy(X0)

    # Input caracteristics
    n, d = D.shape
    m, r = B.shape

    # Store DtD, BtB and DtYB if possible
    DtD = D.T@D
    Bt = B.T
    BtB = Bt@B
    DtYB = D.T@Y@B

    # Computing lambda_max
    DtYBabs = np.abs(DtYB)
    lambda_max = np.sum(np.max(DtYBabs, axis=0))
    lamb = lamb_rel*lambda_max
    if verbose:
        print('lambda max is', lambda_max, ' \n')

    # Initialisation of coefficients x
    # this does not matter too much since the problem is convex
    if X0 is None:
        X = np.zeros([d, r])

    # Choice of stepsize, use Lipschitz constant
    singvalD = np.linalg.svd(DtD)[1][0]
    singvalB = np.linalg.svd(BtB)[1][0] #why 2?
    eta = 1/singvalD/singvalB

    # Initial error
    rec0 = np.linalg.norm(Y - D@X@Bt, 'fro')
    e0 = rec0**2/2 + lamb*ml1(X)
    # e_old = 0
    rec = [rec0]
    e = [np.Inf, e0]

    # Initial iteration count
    iter = 0

    # Main print
    if verbose:
        print('ISTA l11 running\n')

    # Main loop with proximal gradient

    # pairing variable for Fista
    Z = np.copy(X)
    beta = 1

    while np.abs(e[-1] - e[-2])/e[-1] > tol and iter < itermax:

        # printing
        if iter % 10 == 1:
            if verbose:
                print('ISTA iteration ', iter, ' cost ', e[-1], '\n')
            #else:
            #   print('.', end='')

        iter += 1

        # compute the gradient
        X_old = np.copy(X)
        #X, _, _ = prox_ml1(Z - eta * (DtD@Z@BtB - DtYB), lamb*eta, tol=1e-6)
        X = prox_ml1_fast(Z - eta * (DtD@Z@BtB - DtYB), lamb*eta)
        # Extrapolation
        beta_old = beta
        beta = (1+np.sqrt(1+4*beta**2))/2
        Z = X + ((beta_old-1)/beta) * (X-X_old)

        # error computation
        rec_new = np.linalg.norm(Y - D@X@Bt, 'fro')
        rec.append(rec_new)
        e.append(rec_new**2/2 + lamb*ml1(X))

    e = e[1:]

    # Get k largest entries per columns
    # Estimating support (np arrays with supports)
    support = np.argsort(np.abs(X), 0)
    # truncating the support
    support = support[-k:, :]

    # Post-processing option for k-sparse columnwise
    if k is not None:
        # Running least squares
        X, _ = ls_kn_supp(Y, D, B, support, k)

        # error computation
        rec_new = np.linalg.norm(Y - D@X@Bt, 'fro')
        rec.append(rec_new)
        e.append(rec_new**2/2 + lamb*ml1(X))

    if verbose:
        print('\n')

    return X, e, rec, support


def ista(Y, D, B, lamb_rel, k=None, X0=None, itermax=1000, tol=1e-6, verbose=True, samereg=False, warning=False, DtD=None, DtY=None, DtYB=None, BtB=None, return_old=False, eta=None):
    '''
    Solves a simple convex relaxation of the mixed sparse coding problem using Fast Iterative Soft Thresholding (Block-Fista). Each columns has its own regularization parameter.
    The cost function is

        :math:`\\frac{1}{2} \\|Y - DXB^T \\|_F^2 + \\sum_i \\lambda_i \\|X_i\\|_{1}`

    where :math:`\\lambda_i = \\lambda_{rel,i}\\lambda_{\\max,i}`

    Parameters
    ----------
    Y : numpy array
        input data, required

    D : numpy array
        input dictionary (fat), required

    B : numpy array
        input mixing matrix, required

    lamb_rel : float or list of floats
        ratio of lambda_max used as a regularization, required.
        If float is provided, the same regularization ratio is used in all columns.
        If list is provided, it must be of length X.shape[1] and provide regularization level for each columns.

    k : integer
        sparsity level per column, thresholded at the end. Use None to ignore.

    samereg : boolean
        if True, all lamb values are equal to lambda_max*lamb_rel, which yields the usual Lasso problem.

    X0 : numpy array
        initial estimation of X

    itermax : integer (default 1000)
        maximum number of proximal iterations

    tol : float (default 1e-6)
        relative error threshold for stopping the algorithm

    verbose : boolean (default True)
        Set to False to remove prints

    warning : boolean (default False)
        If True, will prompt a warning when the final estimation before debiaising is sparser than the desired sparsity level.

    return_old : boolean (default False)
        If True, adds a fith output which is the estimated X before debiaising

    eta : float (default None)
        the stepsize for ISTA. If None, the Lipschitz constant is computed and used.

    DtD, BtB (...) : numpy arrays
        pre-computed cross products of the inputs if available.

    Returns
    ----------
    X : numpy array
        estimated X

    e : list
        fittings along iterations

    rec : list
        reconstruction errors along iterations

    support : numpy array
        the support of each column of X
    '''

    # Copy input
    X = np.copy(X0)

    # Input caracteristics
    n, d = D.shape
    m, r = B.shape

    # Store DtD, BtB and DtYB if possible
    if DtD is None:
        DtD = D.T@D
    if DtY is None:
        DtY = D.T@Y
    Bt = B.T
    if BtB is None:
        BtB = Bt@B
    if DtYB is None:
        DtYB = DtY@B

    # Computing lambda_max
    DtYBabs = np.abs(DtYB)
    lambda_max = np.max(DtYBabs, axis=0)
    if samereg:
        # We get the usual Lasso lambda_max
        lambda_max = np.max(lambda_max)
    lamb = lamb_rel*lambda_max
    if verbose:
        print('lambda max is', lambda_max, ' \n')

    # Initialisation of coefficients x
    # this does not matter too much since the problem is convex
    if X0 is None:
        X = np.zeros([d, r])

    # Choice of stepsize, use Lipschitz constant (improvable by power iteration or randomization)
    if eta is None:
        singvalD = np.linalg.svd(DtD)[1][0]
        singvalB = np.linalg.svd(BtB)[1][0]
        eta = 1/singvalD/singvalB

    # Initial error
    rec0 = np.linalg.norm(Y - D@X@Bt, 'fro')
    e0 = rec0**2/2 + np.sum(lamb*np.sum(np.abs(X), axis=0))
    # e_old = 0
    rec = [rec0]
    e = [np.Inf, e0]

    # Initial iteration count
    iter = 0

    # Main print
    if verbose:
        print('ISTA l11 running\n')

    # Main loop with proximal gradient

    # pairing variable for Fista
    Z = np.copy(X)
    beta = 1

    while np.abs(e[-1] - e[-2])/e[-1] > tol and iter < itermax:

        # printing
        if iter % 10 == 1:
            if verbose:
                print('ISTA iteration ', iter, ' cost ', e[-1], '\n')
            #else:
            #    print('.', end='')

        iter += 1

        # compute the gradient
        X_old = np.copy(X)
        X = SoftT(Z - eta * (DtD@Z@BtB - DtYB), lamb*eta)
        # Extrapolation
        beta_old = beta
        beta = (1+np.sqrt(1+4*beta**2))/2
        Z = X + ((beta_old-1)/beta) * (X-X_old)

        # error computation
        rec_new = np.linalg.norm(Y - D@X@Bt, 'fro')
        rec.append(rec_new)
        e.append(rec_new**2/2 + np.sum(lamb*np.sum(np.abs(X), axis=0)))

    e = e[1:]

    # Get k largest entries per columns
    # Estimating support (np arrays with supports)
    support = np.argsort(np.abs(X), 0)
    if k is not None:
        # truncating the support
        support = support[-k:, :]
        if warning:
            # check if there are too many zeroes already
            for i in range(r):
                Xabs = np.abs(X)
                if np.min(Xabs[support[:,i],i])==0:
                    print('Warning: regularization may be too strong')

    if verbose:
        print('\n')

    if return_old:
        X_old=np.copy(X)
    # Post-processing option for k-sparse columnwise
    if k is not None:
        # Running least squares
        X, _ = ls_kn_supp(Y, D, B, support, k)

        # error computation
        rec_new = np.linalg.norm(Y - D@X@Bt, 'fro')
        rec.append(rec_new)
        e.append(rec_new**2/2 + np.sum(lamb*np.sum(np.abs(X), axis=0)))

    if return_old:
        return X, e, rec, support, X_old
    else:
        return X, e, rec, support

def ista_nn(Y, D, B, lamb_rel, k=None, X0=None, itermax=1000, tol=1e-6, verbose=True, samereg=False, return_old=False, DtD=None, DtY=None, DtYB=None, BtB=None,eta=None, warning=False):
    '''
    Solves a simple convex relaxation of the mixed sparse coding problem using Fast Iterative Soft Thresholding (Fista) under nonnegativity constraints. Each columns has its own regularization parameter.
    The cost function is

        :math:`\\frac{1}{2} \\|Y - DXB^T \\|_F^2 + \\sum_i \\lambda_i \\|X_i\\|_{1}\; s.t.\; X\\geq 0`

    where :math:`\\lambda_i = \\lambda_{rel,i}\\lambda_{\\max,i}`

    Parameters
    ----------
    Y : numpy array
        input data, required

    D : numpy array
        input dictionary (fat), required

    B : numpy array
        input mixing matrix, required

    lamb_rel : float or list of floats
        ratio of lambda_max used as a regularization, required.
        If float is provided, the same regularization ratio is used in all columns.
        If list is provided, it must be of length X.shape[1] and provide regularization level for each columns.

    k : integer
        sparsity level per column, thresholded at the end. Use None to ignore.

    samereg : boolean
        if True, all lamb values are equal to lambda_max*lamb_rel, which yields the usual Lasso problem.

    X0 : numpy array
        initial estimation of X

    itermax : integer (default 1000)
        maximum number of proximal iterations

    tol : float (default 1e-6)
        relative error threshold for stopping the algorithm

    verbose : boolean (default True)
        Set to False to remove prints

    warning : boolean (default False)
        If True, will prompt a warning when the final estimation before debiaising is sparser than the desired sparsity level.

    return_old : boolean (default False)
        If True, adds a fith output which is the estimated X before debiaising

    eta : float (default None)
        the stepsize for ISTA. If None, the Lipschitz constant is computed and used.

    DtD, BtB (...) : numpy arrays
        pre-computed cross products of the inputs if available.

    Returns
    ----------
    X : numpy array
        estimated X

    e : list
        fittings along iterations

    rec : list
        reconstruction errors along iterations

    support : numpy array
        the support of each column of X
    '''

    # Copy input
    X = np.copy(X0)

    # Input caracteristics
    n, d = D.shape
    m, r = B.shape

    # Store DtD, BtB and DtYB if possible
    if DtD is None:
        DtD = D.T@D
    if DtY is None:
        DtY = D.T@Y
    Bt = B.T
    if BtB is None:
        BtB = Bt@B
    if DtYB is None:
        DtYB = DtY@B

    # Computing lambda_max
    DtYBabs = np.abs(DtYB)
    lambda_max = np.max(DtYBabs, axis=0)
    if samereg:
        # We get the usual Lasso lambda_max
        lambda_max = np.max(lambda_max)
    lamb = lamb_rel*lambda_max
    if verbose:
        print('lambda max is', lambda_max, ' \n')

    # Initialisation of coefficients x
    # this does not matter too much since the problem is convex
    if X0 is None:
        X = np.zeros([d, r])

    # Choice of stepsize, use Lipschitz constant (improvable by power iteration or randomization)
    if eta is None:
        singvalD = np.linalg.svd(DtD)[1][0]
        singvalB = np.linalg.svd(BtB)[1][0]
        eta = 1/singvalD/singvalB

    # Initial error
    rec0 = np.linalg.norm(Y - D@X@Bt, 'fro')
    e0 = rec0**2/2 + np.sum(lamb*np.sum(np.abs(X), axis=0))
    # e_old = 0
    rec = [rec0]
    e = [np.Inf, e0]

    # Initial iteration count
    iter = 0

    # Main print
    if verbose:
        print('ISTA l11 running\n')

    # Main loop with proximal gradient

    # pairing variable for Fista
    Z = np.copy(X)
    beta = 1

    while np.abs(e[-1] - e[-2])/e[-1] > tol and iter < itermax:

        # printing
        if iter % 10 == 1:
            if verbose:
                print('ISTA iteration ', iter, ' cost ', e[-1], '\n')
            #else:
            #    print('.', end='')

        iter += 1

        # compute the gradient
        X_old = np.copy(X)
        X = np.maximum(Z - eta * (DtD@Z@BtB - DtYB + lamb*np.ones(Z.shape)) ,0)
        # Extrapolation
        beta_old = beta
        beta = (1+np.sqrt(1+4*beta**2))/2
        Z = X + ((beta_old-1)/beta) * (X-X_old)

        # error computation
        rec_new = np.linalg.norm(Y - D@X@Bt, 'fro')
        rec.append(rec_new)
        e.append(rec_new**2/2 + np.sum(lamb*np.sum(np.abs(X), axis=0)))

    e = e[1:]


    # Get k largest entries per columns
    # Estimating support (np arrays with supports)
    support = np.argsort(np.abs(X), 0)
    if k is not None:
        # truncating the support
        support = support[-k:, :]
        if warning:
            # check if there are too many zeroes already
            for i in range(r):
                Xabs = np.abs(X)
                if np.min(Xabs[support[:,i],i])==0:
                    print('Warning: regularization may be too strong')

    if verbose:
        print('\n')

    if return_old:
        X_old=np.copy(X)
    # Post-processing option for k-sparse columnwise
    if k is not None:
        # Running nonnegative least squares
        X, _ = ls_kn_supp(Y, D, B, support, k, nonnegative=True)

        # error computation
        rec_new = np.linalg.norm(Y - D@X@Bt, 'fro')
        rec.append(rec_new)
        e.append(rec_new**2/2 + np.sum(lamb*np.sum(np.abs(X), axis=0)))

    if return_old:
        return X, e, rec, support, X_old
    else:
        return X, e, rec, support



def iht_mix(Y, D, B, k , X_in, tol=1e-6, itermax=1000, verbose=False, DtD=None, DtY=None, eta=None):
    '''
    An adaptation of the (Extrapolated) Iterated Hard Thresholding Algorithm for the mixed sparse coding problem. At each iteration, a Nesterov Fast Gradient step is performed with projection on the set of columnwise k-sparse matrices.

    Parameters
    ----------
    Y : numpy array
        input data, required

    D : numpy array
        input dictionary (fat), required

    B : numpy array
        input mixing matrix, required

    k : integer
        Sparsity level, must be the number of terms in each subset of S (not checked)

    Xin : numpy array
        Initial guess for solution X

    itermax : integer (default 1000)
        maximum number of proximal iterations

    tol : float (default 1e-6)
        relative error threshold for stopping the algorithm

    verbose : boolean (default True)
        Set to False to remove prints

    eta : float (default None)
        the stepsize for ISTA. If None, the Lipschitz constant is computed and used.

    DtD, DtY : numpy arrays
        pre-computed cross products of the inputs if available.

    Returns
    ----------
    X : numpy array
        Solution of the mixed sparse coding problem with fixed column sparsity

    err : float
        Reconstruction error / residuals.

    support : numpy array
        the support of each column of X
    '''
    X = np.copy(X_in)
    if DtD is None:
        DtD = D.T@D
    if DtY is None:
        DtY = D.T@Y
    Bt = B.T
    BtB = Bt@B
    DtY = D.T@Y
    DtYB = DtY@B
    if eta is None:
        step = 1/np.linalg.svd(DtD)[1][0]/np.linalg.svd(BtB)[1][0] #why 2?
    else:
        step = eta
    if verbose:
        print('IHT stepsize is fixed to ', step)
    err = np.linalg.norm(Y - D@X@Bt, 'fro')
    err = [np.Inf, err]

    #Fast version
    Z = np.copy(X)
    beta = 1
    iter = 0

    while np.abs(err[-1] - err[-2])/err[-1] > tol and iter < itermax:

        # printing
        if iter % 10 == 1:
            if verbose:
                print('IHT iteration ', iter, ' cost ', err[-1], '\n')
            #else:
            #    print('.', end='')

        iter += 1
        X_old = np.copy(X)
        X = HardT(Z - step * (DtD@Z@BtB - DtYB), k)
        beta_old = beta
        beta = (1+np.sqrt(1+4*beta**2))/2
        Z = X + ((beta_old-1)/beta) * (X-X_old)
        err.append(np.linalg.norm(Y - D@X@Bt, 'fro')) # suboptimal

    #for iter in range(itermax):
        #X = ht_op(X - step * (DtD@X@BtB - DtYB), k)
        #err.append(np.linalg.norm(Y - D@X@B.T, 'fro')) # suboptimal

    err = err[1:]

    # Get k largest entries per columns
    # Estimating support (np arrays with supports)
    support = np.argsort(np.abs(X), 0)
    # truncating the support
    support = support[-k:, :]

    # Post-processing option for k-sparse columnwise
    if k is not None:
        # Running least squares
        X, _ = ls_kn_supp(Y, D, B, support, k)

        # error computation
        err.append(np.linalg.norm(Y - D@X@Bt, 'fro'))

    if verbose:
        print('\n')

    return X, err, support

def homp(Y, D, B, k, Xin=None, tol=1e-6, itermax=1000):
    '''
    Hierarchical Orthogonal Matching Pursuit for the mixed sparse coding
    problem. Computes k-sparse approximations of column sub-problems until
    convergence using OMP (modified) as a routine.

    Parameters
    ----------
    Y : numpy array
        Input data, required

    D : numpy array
        Input dictionary, required

    B : numpy array
        Input mixing matrix, required

    k : int
        Sparsity level, required

    Xin : numpy array (default: random)
        Initial X

    itermax : integer (default 1000)
        maximum number of proximal iterations

    tol : float (default 1e-6)
        relative error threshold for stopping the algorithm

    Returns
    ----------
    X : numpy array
        Final estimated X

    err : list
        Error after each pass.

    S : numpy array
        Support of the solution
    '''
    n, m = Y.shape
    _, r = B.shape
    _, d = D.shape
    # todo: checks on sizes
    # todo: init X random
    X = np.copy(Xin)
    # Supports in numpy 1/0 format
    S = np.zeros(X.shape)
    # Supports in list of list format (col per col)
    Slist = [[0 for i in range(k)] for j in range(r)]
    err = np.linalg.norm(Y-D@X@B.T, 'fro')
    err = [np.Inf, err]
    #itermax = 50
    it = 0

    while it < itermax:
        if np.abs(err[-1] - err[-2])/err[-1] < tol:
            break
        it += 1
        for i in range(r):
            reject = 0
            V = Y - D@X@B.T + D@np.outer(X[:,i],B[:,i])
            normb = np.linalg.norm(B[:,i],2) ** 2
            Vb = V@B[:,i]/normb
            x, s_idx = omp(Vb, D, k)
            if np.linalg.norm(Vb - D@x) > np.linalg.norm(Vb - D@X[:,i]):
                #print('bad, bad omp !! Rejecting step')
                x = np.zeros(d)
                z = np.linalg.lstsq(D[:,Slist[i]], Vb, rcond=None)
                x[Slist[i]] = z[0]
                reject += 1
            else:
                Slist[i] = s_idx
            X[:,i] = x

        err.append(np.linalg.norm(Y - D@X@B.T, 'fro'))
        if reject == r:
            print('why, why homp !!! you were such a good boy. Stopping algorithm to avoid infinity loop.')
            break

    err = err[1:]
    # At the end, do a least squares with the final support
    X, err_temp = ls_kn_supp(Y, D, B, Slist, k)
    err.append(err_temp)

    return X, err, np.transpose(np.array(Slist))

def omp(V, D, k):
    '''
    Orthogonal Matrix Pursuit modified, in a naive implementation.

    Parameters
    ----------
    V : numpy column vector
        input data, required

    D : numpy array
        input dictionary, required

    b : numpy column vector
        input mixing vector, required

    k : integer
        sparsity level, required

    Returns
    ----------
    x : numpy column vector
        estimated sparse coefficients

    s : numpy column vector
        binary vector with ones at the support position of x

    TODO: write for matrix input
    '''
    _, d = D.shape
    x = np.zeros(d)
    list_idx = []
    res = V

    for p in range(k):
        temp = D.T@res
        j = np.argmax(np.abs(temp))
        list_idx.append(j)
        z = np.linalg.lstsq(D[:,list_idx], V, rcond=None)
        x[list_idx] = z[0]
        res = V - D@x

    return x, list_idx


def pseudo_trick(Y,D,B,k):
    '''
    Tries to solve the mixed sparse coding problem by looking at the distorted problem
    :math:`min_{\\|X_i\\|_0\\leq k} \\|Y(B^T)^\\dagger - DX \\|_F^2`,
    which is solved in parallel, column by column, using the omp algorithm.
    As a post-processing step, a least squares fitting is done with the identified support.

    Parameters
    ----------
    Y : numpy array
        Input data, required

    D : numpy array
        Input dictionary, required

    B : numpy array
        Input mixing matrix, required

    k : int
        Sparsity level, required

    Returns
    ----------
    X : numpy array
        Final estimated X

    err : list
        Error after each pass.

    S : numpy array
        Support of the solution
    '''
    _, d = D.shape
    _, r = B.shape
    C = np.linalg.pinv(B.T)
    V = Y@C
    X_trick = np.zeros([d, r])
    Slist = [[0 for i in range(k)] for j in range(r)]
    for i in range(r):
        z, s_idx = omp(V[:, i], D, k)
        Slist[i] = s_idx
        X_trick[:, i] = z

    # Debiaising
    X_trick, err = ls_kn_supp(Y, D, B, Slist, k)

    return X_trick, err, np.transpose(np.array(Slist))

def brute_trick(Y,D,B,k):
    '''
    A brute force version of the pseudo_trick, for checking counter examples. Returns the error in the B domain.
    '''
    _, d = D.shape
    _, r = B.shape
    C = np.linalg.pinv(B.T);
    V = Y@C
    X_trick = np.zeros([d,r])
    Slist = [[0 for i in range(k)] for j in range(r)]
    for i in range(r):
        print(i)
        err = 10 ** 16
        count=0
        listcombi = combinations([i for i in range(d)],k)
        for s in listcombi:
            count = count+1
            if count%100==0:
                print(count)
            Ds = D[:,s]
            z = np.linalg.lstsq(Ds,V[:,i], rcond=None)[0]
            errnew = np.linalg.norm(V[:,i]-Ds@z)
            if errnew < err:
                err = errnew
                store_s = s
                store_z = z
        Slist[i] = list(store_s)
        X_trick[store_s,i] = store_z

    # Debiaising
    #X_trick, err = ls_kn_supp(Y,D,B,Slist,k)
    err = np.linalg.norm(Y - D@X_trick@B.T, 'fro')

    return X_trick, err, Slist
