import numpy as np
from mscode.methods.algorithms import ista_mix, ista, ista_nn
from mscode.utils.generator import gen_mix, initialize

def count_support(Strue, Sest):
    '''
    Counts the number of corrected estimated atoms

    Parameters
    ----------
    Strue : list of list (or numpy nd array)
        True support

    Sest  : list of list (or numpy nd array)
        Estimated support

    Returns
    -------
    score : float
        Percentage of correctly estimated atoms
    '''


    if isinstance(Strue, np.ndarray):
        # Conversion to list of list from ndarray numpy
        Strue = list(Strue.T)
        Strue = [list(i) for i in Strue]

    if isinstance(Sest, np.ndarray):
        # Conversion to list of list from ndarray numpy
        Sest = list(Sest.T)
        Sest = [list(i) for i in Sest]

    r = len(Strue)
    maxscore = r*len(Strue[0])  # all same size

    cnt = 0
    for i in range(r):
        for el in Strue[i]:
            if el in Sest[i]:
                cnt += 1

    return cnt/maxscore*100

def redundance_count(S):
    '''
    Checks how many times the same columns are chosen in various support.
    Gives the total of repeated columns with multiplicities.

    Parameters
    ----------
    S : list of list or numpy array
        Support

    Returns
    -------
    out : int
        Number of repeated columns
    '''

    if isinstance(S, np.ndarray):
        # Conversion to list of list from ndarray numpy
        Strue = list(S.T)
        Strue = [list(i) for i in S]

    cnt=0
    r = len(S)

    # unfolded uniqued list
    Sunfold = []
    for i in S:
        for j in i:
            Sunfold.append(j)
    Sunfold = list(set(Sunfold))

    for el in Sunfold:
        for i in S:
            for j in i:
                if el == j:
                    cnt += 1

    return cnt - len(Sunfold)

def find_lambda(dims, grid, ista_type):
    '''
    Finds a good lambda value by running a fixed (Mixed) Lasso problem of fixed size with varying regularization.

    Parameters
    ----------
    dims : list
        (m,n,d,k,r,SNR,cond)

    grid : list
        tested lambda values

    ista_type : string
        choose between 'Fista', 'Fista_m' and 'Fista_nn'

    Returns
    -------
    lamb : float
        estimated good regularization ratio
    '''

    m,n,d,k,r,SNR,cond = dims
    Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond)
    X0 = initialize([d,r], distr = 'Zeros')
    # initial recovery rates and best lambdas
    recov_ista = 0
    lamb_best_ista = 0
    cnt = 1

    for lamb in grid:
        # Running FIstas
        if ista_type == 'Fista_m':
            _, _, _, S_ista = ista_mix(Y, D, B, lamb, k=k, X0=X0, tol=1e-5, verbose=False)
        elif ista_type == 'Fista':
            _, _, _, S_ista = ista(Y, D, B, lamb, k=k, X0=X0, tol=1e-5, verbose=False)
        elif ista_type == 'Fista_nn':
            _, _, _, S_ista = ista_nn(Y, D, B, lamb, k=k, X0=X0, tol=1e-5, verbose=False)

        # Computing recovery rates, store if larger
        temp_recov_ista = count_support(S, S_ista)
        if temp_recov_ista > recov_ista:
            lamb_best_ista = lamb
            recov_ista = temp_recov_ista
        elif temp_recov_ista == recov_ista:
            # store for averaging
            cnt += 1
            lamb_best_ista += lamb

    lamb_best_ista = lamb_best_ista/cnt
    return lamb_best_ista
