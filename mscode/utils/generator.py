import numpy as np
import tensorly as tl


def gen_mix(dims, k, snr=20, distr='Gaussian', cond=1, cond_rand=False, decrease_coeff = 0.7):
    '''
    Generates simulated dataset for experiments according to the mixed sparse coding model.

    Parameters
    ----------
    dims : list of length 4
        [m, n, d, r]

    k : integer
        sparsity constraint

    snr : integer
        signal to noise ratio, controls noise level

    distr : string
        Default is 'Gaussian', but 'Uniform' also works. 'Decreasing' is Gaussian D,B and Uniform X with artificially decreasing weights for X.

    cond : float
        Controls the conditionning of the B matrix

    cond_rand : boolean
        If True, the singular values of a random gaussian matrix are scaled by cond. If False, the singular values are linearly spaced and conditionning is indeed cond.

    decrease_coeff : float (default 0.7)
        In the 'Decreasing setup', multiplicative factor for decrease

    Returns
    -------
    Y : nd numpy array
        noised data

    Ytrue : nd numpy array
        noiseless data

    D : nd numpy array
        dictionary normalized columnswise in l2 norm

    B : nd numpy array
        mixing matrix

    X : nd numpy array
        true unknown sparse coefficients

    S : nd numpy array
        support of X

    sig : float
        noise variance used in practice

    condB : float
        the true conditionning of B
    '''
    m, n, d, r = dims

    if distr == 'Gaussian' or 'Decreasing':
        D = np.random.randn(n, d)
        B = np.random.randn(m, r)
    elif distr == 'Uniform':
        D = np.random.rand(n, d)
        B = np.random.rand(m, r)
    else:
        print('Distribution not supported')

    for i in range(d):
        D[:, i] = D[:, i]/np.linalg.norm(D[:, i])

    # add poor B conditionning
    u, s, v = np.linalg.svd(B)
    if cond_rand:
        a = (s[0] - s[-1]/100)/(s[0]-s[-1])
        b = s[0] - a*s[0]
        s = a*s + b  # conditionning is exactly 100, sv are linearly spaced
    else:
        s = np.linspace(1, 1/cond, r)
    B = u[:, :r]@np.diag(s)@v.T

    # X k-sparse columnwise generation
    X = np.zeros([d, r])
    S = []
    for i in range(r):
        pos = np.random.permutation(d)[0:k]
        if distr == 'Uniform':
            X[pos, i] = np.random.rand(k)
        elif distr == 'Gaussian':
            X[pos, i] = np.random.randn(k)
        elif distr == 'Decreasing':
            for l, npos in enumerate(pos):
                X[npos,i] = np.random.choice((-1,1))*np.random.rand(1)*(decrease_coeff ** l)
        else:
            print('Distribution not supported')
        S.append(pos)

    # Formatting to np array
    S = np.transpose(np.array(S))

    # Noise and SNR
    Ytrue = D@X@B.T
    noise = np.random.rand(n, m)

    spower = np.linalg.norm(Ytrue, 'fro')**2
    npower = np.linalg.norm(noise, 'fro')**2
    old_snr = np.log10(spower/npower)
    sig = 10**((old_snr - snr/10)/2)
    noise = sig*noise  # scaling to get SNR right

    Y = Ytrue + noise

    return Y, Ytrue, D, B, X, S, sig, s[0]/s[-1]

def gen_mix_tensor(dims, dims_D, k, snr=20, distr='Gaussian', cond=1, cond_rand=False, decrease_coeff = 0.7):
    '''
    Generates simulated dataset for experiments according to the mixed sparse coding model.

    Parameters
    ----------
    dims : list
        [m, n, l]
        #todo: implement for arbriratry order

    dims_D : list with the number of atoms and rank
        [d, r]

    k : integer
        sparsity constraint

    snr : integer
        signal to noise ratio, controls noise level

    distr : string
        Default is 'Gaussian', but 'Uniform' also works. 'Decreasing' is Gaussian D,B and Uniform X with artificially decreasing weights for X.

    cond : float
        Controls the conditionning of the B matrix

    cond_rand : boolean
        If True, the singular values of a random gaussian matrix are scaled by cond. If False, the singular values are linearly spaced and conditionning is indeed cond.

    decrease_coeff : float (default 0.7)
        In the 'Decreasing setup', multiplicative factor for decrease

    Returns
    -------
    Y : nd numpy array
        noised data

    Ytrue : nd numpy array
        noiseless data

    D : nd numpy array
        dictionary normalized columnswise in l2 norm

    B : nd numpy array
        mixing matrix

    X : nd numpy array
        true unknown sparse coefficients

    S : nd numpy array
        support of X

    sig : float
        noise variance used in practice

    condB : float
        the true conditionning of B
    '''
    n,m,l = dims
    d, r = dims_D

    if distr == 'Gaussian' or 'Decreasing':
        D = np.random.randn(n, d)
        B = np.random.randn(m, r)
        C = np.random.randn(l, r)
    elif distr == 'Uniform':
        D = np.random.rand(n, d)
        B = np.random.rand(m, r)
        C = np.random.rand(l, r)
    else:
        print('Distribution not supported')

    for i in range(d):
        D[:, i] = D[:, i]/np.linalg.norm(D[:, i])

    # add poor B conditionning
    u, s, v = np.linalg.svd(B)
    if cond_rand:
        a = (s[0] - s[-1]/100)/(s[0]-s[-1])
        b = s[0] - a*s[0]
        s = a*s + b  # conditionning is exactly 100, sv are linearly spaced
    else:
        s = np.linspace(1, 1/cond, r)
    B = u[:, :r]@np.diag(s)@v.T

    # X k-sparse columnwise generation
    X = np.zeros([d, r])
    S = []
    for i in range(r):
        pos = np.random.permutation(d)[0:k]
        if distr == 'Uniform':
            X[pos, i] = np.random.rand(k)
        elif distr == 'Gaussian':
            X[pos, i] = np.random.randn(k)
        elif distr == 'Decreasing':
            for l, npos in enumerate(pos):
                X[npos,i] = np.random.choice((-1,1))*np.random.rand(1)*(decrease_coeff ** l)
        else:
            print('Distribution not supported')
        S.append(pos)

    # Formatting to np array
    S = np.transpose(np.array(S))

    # Noise and SNR
    #Ytrue = D@X@B.T
    Ytrue = tl.cp_tensor.cp_to_tensor((None,[D@X,B,C]))
    noise = np.random.rand(n, m, l)

    spower = tl.norm(Ytrue, 2)**2
    npower = tl.norm(noise, 2)**2
    old_snr = np.log10(spower/npower)
    sig = 10**((old_snr - snr/10)/2)
    noise = sig*noise  # scaling to get SNR right

    Y = Ytrue + noise

    return Y, Ytrue, D, B, C, X, S, sig, s[0]/s[-1]

def initialize(dims, distr='Gaussian'):
    '''
    Provides an initial guess for X in the mixed sparse coding problem given dimensions are an elementwise standard distribution.

    Parameters
    ----------
    dims : list
        [d,r], where d is the dictionary size, and r the mixing size

    distr : string
        "Gaussian", "Uniform", "Zeros"

    Returns
    -------
    X : numpy array
        Initial guess for Mixed Sparse Coding
    '''
    if distr == 'Gaussian':
        X = np.random.randn(dims[0], dims[1])
    elif distr == 'Uniform':
        X = np.random.rand(dims[0], dims[1])
    elif distr == 'Zeros':
        X = np.zeros(dims)

    return X
