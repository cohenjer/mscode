import numpy as np
from scipy.optimize import nnls
from numpy.matlib import repmat as repmat
from mscode.methods.prox_ind_l1_norm import prox_l1inf

def ml1(X):
    '''
    Computes the induced matrix l1 norm of X
    '''
    return np.max(np.sum(np.abs(X), axis=0))

def SoftT(x, lamb):
    '''
    Computes the Soft-Thresholding of input vector x, with coefficient lamb
    '''
    return np.maximum(np.abs(x) - lamb, 0)*np.sign(x)


def ls_kn_supp(Y, D, B, S, k, nonnegative=False):
    '''
    Solves the mixed sparse coding problem once the support has been fixed. This is a least-squares problem with a lot of structure, so let's be careful not to waste too much computation time. If the support is large, better not to use this function as it will form a k**2 * r**2 system

    We solve a linear system :math:`M^T y = M^T M z` where
    :math:`M = [D_{S_1} \\odot b_1, \\ldots, D_{S_i} \\odot b_i, \\ldots]`
    which is yields the unique solution to the overcomplete least squares problem
    :math:`\\min_z \\| y - Mz \\|_2^2`. We use the structure of M to compute :math:`M^T y` and :math:`M^T M`.

    Can also handle nonnegative least squares with scipy nnls active set solver.

    Parameters
    ----------
    Y : numpy array
        input data, required

    D : numpy array
        input dictionary (fat), required

    B : numpy array
        input mixing matrix, required

    S : list of list of integers (or numpy array)
        Known support for each column of the solution, required
        example: [[0,3],[1,3]] means columns one has support S_1 = {0,3} and column 2 has support S_2={1,3}.

    k : integer
        Sparsity level, must be the number of terms in each subset of S (not checked)

    nonnegative : boolean
        Default to False. Set to True to impose nonnegativity constraints on the solution, the NNLS solver is active set from scipy.

    Returns
    -------
    X : numpy array
        Solution of the mixed sparse coding problem with fixed column sparsity

    err : float
        Reconstruction error / residuals.

    '''
    #TODO: check correctness, seen some weird things with error not being minimal.
    #TODO: if support is too large, solve with conjugate gradient instead.
    m, n = Y.shape
    _, d = D.shape
    _, r = B.shape
    klist = np.array([i for i in range(k)])

    if isinstance(S, np.ndarray):
        # Conversion to list of list from ndarray numpy
        S = list(S.T)
        S = [list(i) for i in S]

    # Computing Mty
    # YBt = [N_1, ..., N_i, ..., N_r]
    YB = Y@B  # m*r
    Mty = np.zeros(k*r)
    for i in range(r):
        D_S = D[:, S[i]] #size should be k
        Mty[i*k + klist] = D_S.T@YB[:,i]

    # Computing MtM
    MtM = np.zeros([k*r, k*r])
    for i in range(r):
        for j in range(r):
            D_Si = D[:,S[i]] #size should be k
            D_Sj = D[:,S[j]] #size should be k
            temp = (D_Si.T@D_Sj)*(B[:,i].T@B[:,j]) # size should be k^2
            for p in range(k):
                MtM[klist + i*k, j*k+p] = temp[:,p]

    # Let's solve the small square system
    if nonnegative==False:
        try:
            z = np.linalg.solve(MtM, Mty)
        except np.linalg.LinAlgError:
            MtM = MtM + 1e-6*np.eye(MtM.shape[0],MtM.shape[1])
            z = np.linalg.solve(MtM, Mty)
    else:
        MtM = MtM + 1e-6*np.eye(MtM.shape[0],MtM.shape[1])
        z,_ = nnls(MtM, Mty, 1000)

    # Putting back the values in X
    X = np.zeros([d,r])
    for i in range(r):
        X[S[i],i] = z[i*k + klist]

    # error computed in the original domain, check if efficient version
    err = np.linalg.norm(Y - D@X@B.T, 'fro')

    return X, err

def HardT(X, k):
    '''
    Truncates to the k largest values of X columnwise, arbitrarily so if necessary. This is somewhat the proximal operator of the l0 "norm".
    '''
    if X.ndim == 1:
        m=X.shape[0]
        Z = np.copy(X)
        idx = np.argsort(np.abs(X), axis = 0)
        Z[idx[0:m-k]] = 0
    else:
        m, n = X.shape
        Z = np.copy(X)
        idx = np.argsort(np.abs(X), axis = 0)
        for p in range(n): # may be vectorized
            Z[idx[0:m-k,p],p] = 0

    return Z


def ls_cg(Y, DtD, BtB, Xinit, rho, itercg = 50):
    '''
    Solves linear systems of the form

         :math:`(D^TD X B^TB + \\rho X) = Y`

    using conjugate gradient. The important point is that the large mixing matrix (DtD otimes BtB + rho I) is never formed explicitely. However the DtD matrix is formed, as in the other methods.

    Parameters
    ----------
    Y : numpy array
        Input data. It corresponds to DtYB + rho (Z - mu) in ADMM.

    DtD : numpy array
        dictionary

    BBt : numpy array
        mixing matrix

    Xinit : numpy array
        current estimate for the coefficients X

    rho : float
        parameter of the ADMM

    itercg : int (default 50)
        maximum number of conjugate gradient iterations

    Returns
    -------
    X : numpy array
        estimated least squares solution

    out : internal output
    '''
    X = np.copy(Xinit)
    d, r = np.shape(Y)

    # getting this next part right is the whole point of the CG here
    AX = (DtD@X)@BtB + rho*X

    R = Y - AX
    tol = 1e-16*np.linalg.norm(Y,'fro')
    P = np.copy(R)
    stock = []

    for i in range(itercg):
        if np.dot(R.flatten(), R.flatten()) < tol:
            break
        # Classical CG (vectorized for understanding but we could do all in matrix format), with clever Ap product.
        AP = DtD@P@BtB + rho*P
        alpha = np.dot(R.flatten(), R.flatten())/np.dot(P.flatten(), AP.flatten())
        X = X + alpha*P
        R_old = np.copy(R)
        R = R - alpha*AP
        beta = np.dot(R.flatten(),R.flatten())/np.dot(R_old.flatten(), R_old.flatten())
        P = R + beta * P
        stock.append(np.dot(R.flatten(), R.flatten()))


    return X, stock


def prox_ml1_fast(X,lamb):
    '''
    A faster proximal operator algorithm for l1infty norm, exact after a few steps.

    Reference:
    The fastest l1oo prox in the west, Benjamin Bejar, Ivan Dokmanic and Rene Vidal, 2019

    Note: This is simply a wrapper for their code.

    Be careful of the following bug: if X is an integer array, the output will always be 0.
    '''
    # output variable
    X = np.asfortranarray(X)
    V = np.asfortranarray(np.zeros(X.shape))

    # Catch exceptions or bugs
    if lamb==0:
        V = X
    else:
        prox_l1inf(X,V,lamb)
    return V



def prox_ml1(X, lamb, tol=1e-10):
    '''
    Computes the proximal operator of the matrix induced l1 norm.

    Parameters
    ----------
    X : numpy array
        input of the proximal operator

    lamb : float
        regularization parameter

    tol : float
        small tolerance on the value of the maximal columnwise 1 norm

    Returns
    -------
    U : numpy array
        the proximal operator applied to X

    t : float
        maximum l1 norm of the columns of U

    nu_t : list of floats
        optimal dual parameters

    Credits to Jeremy E. Cohen
    Reference: "Computing the proximal operator of the l1 induced matrix norm, J.E.Cohen, arxiv:2005.06804v2".
    '''

    # Lambda cannot be Zero
    if lamb == 0:
        out = np.copy(X)
        return out, np.max(np.sum(np.abs(X), axis=0)), 0

    # Remove single column case
    if X.ndim==1:
        out = SoftT(B, lamb)
        return out, np.sum(np.abs(out)), 1

    #%% Precomputations

    # Constants
    n, m  = np.shape(X)
    ps    = np.linspace(1,n,n)
    Xsort = np.sort(np.abs(X), axis=0)
    Xsort = np.flip(Xsort, axis=0)
    Xcumsum=np.cumsum(Xsort, axis=0)

    # Maximal value of t (prox is identity)
    Xsum  = np.sum(Xsort, axis=0)  # Xsort for abs value
    tmax = np.max(Xsum)
    tmin = 0
    t = tmax/2  # initial value in the middle of the admissible interval

    # Find the order of visited columns in X
    sorted_sums = np.flip(np.sort(Xsum))
    order = np.flip(np.argsort(Xsum))

    # Deal with the lamb>lamb_max case in advance to avoid bug
    if lamb>=np.sum(Xsort[0,:]):
        U = np.zeros([n,m])
        t = 0
        nu_t = Xsort[0,:]/lamb
        return U, t, nu_t

    #%% Starting bisection
    while tmax-tmin > tol:
        # Compute the current active set and its size
        I = order[sorted_sums>t]
        l = len(I)

        # Process commulative sums
        Xcumsum_t = (Xcumsum[:, I]-t)/lamb

        # Compute the candidate nu values
        Ps = np.transpose(repmat(ps, l, 1))  # matrix of sequences from 1 to n in with i columns
        nu = Xcumsum_t/Ps

        nu_t = np.zeros(m)
        N = Xsort[:, I] - lamb*nu
        # temp is reverse sorted
        N = np.flip(N, axis=0)  # temp is sorted
        for j in range(l):
            # Find the largest index for which the condition described in the paper is satisfied
            # i.e.  xsort(p) - \lambda\nu(p) >= 0
            idx = np.searchsorted(N[:,j], 0, side='left')
            idx = len(N[:,j]) - idx - 1  # counter effect flip
            nu_t[I[j]] = nu[idx, j]

        # end j loop

        # Checking dual condition 1< or 1> to move t
        if np.sum(nu_t) < 1:
            # t must be decreased
            tmax = t
            t = (t + tmin)/2
        else:
            # t must be increased
            tmin = t
            t = (t + tmax)/2

    # Final step, thresholding vectors that need to be
    U = SoftT(X, lamb*nu_t)

    return U, t, nu_t
