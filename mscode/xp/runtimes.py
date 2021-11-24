# Average 5 runs (problems) for various n,m dimensions. k,d fixed
# Same wrt k,d and n,m fixed large
# For all algorithms
# Check av. number of iterations and time per iteration as well when applicable
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations, product
import shelve
from mscode.utils.utils import count_support, redundance_count
from mscode.methods.algorithms import iht_mix, homp, omp, ls_kn_supp, pseudo_trick, brute_trick, ista_mix, ista, admm_mix
from mscode.utils.generator import gen_mix, initialize
import time
import pandas as pd

# Seeding
np.random.seed(0)

## 1: fixed k,d, runtime wrt n,m
# Generation
k = 5
r = 6
d = 100
SNR = 20  # dB
cond = 2*1e2
tol = 1e-6
distr = 'Gaussian'

# vary n,m
grid_n = [10, 50, 1000]
grid_m = [10, 50, 1000]

Nbruns = 10

# Storing in DataFrame
store_pd = pd.DataFrame(columns=["xp", "n", "m", "d", "k", "algorithm", "iteration", "time"])

for i in range(3):
    for j in range(3):
        for l in range(Nbruns):
            n = grid_n[i]
            m = grid_n[j]
            print('d,k,n,m,r',d,k,n,m,r)

            Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr=distr)
            X0 = initialize([d,r], distr = 'Gaussian')

            # Running HOMP
            tic = time.time()
            X_homp, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)
            time_homp = time.time()-tic

            # Running ISTA_mix (init 0)
            lamb = 0.0005
            tic = time.time()
            _, cost_istam, err_istam, S_istam = ista_mix(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)
            time_istam = time.time()-tic

            # Test Ista
            lamb = 0.001
            tic = time.time()
            _, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)
            time_ista = time.time()-tic

            # Test IHTmix
            tic = time.time()
            X_iht, err_iht, S_iht = iht_mix(Y, D, B, k, X0, tol=tol)
            time_iht = time.time()-tic

            # Comparison with trick
            tic = time.time()
            X_trick, err_trick, S_trick = pseudo_trick(Y, D, B, k)
            time_trick = time.time()-tic

            # Compare with best solution of true support (S)
            tic = time.time()
            X_S, err_S = ls_kn_supp(Y, D, B, S, k)
            time_supportest = time.time()-tic

            # Storing
            dic = {
                "xp":"XP1",
                'n':6*[n], 'm':6*[m], 'd':6*[d], 'k':6*[k],
                'algorithm': ['HOMP', 'Mixed-FISTA', 'Block-FISTA', 'IHT', 'TrickOMP', 'Fixed Support'],
                'iteration': [len(err_homp), len(err_istam), len(err_ista), len(err_iht), 1, 1],
                'time': [time_homp, time_istam, time_ista, time_iht, time_trick, time_supportest],
            }
            store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

# Second test: fix n, m vary d,k
n = 50
m = 50
grid_d = [50, 100, 1000]
grid_k = [5, 10, 30]

Nbruns = 10

for i in range(3):
    for j in range(3):
        for l in range(Nbruns):
            d = grid_d[i]
            k = grid_k[j]
            print('d,k,n,m,r',d,k,n,m,r)

            Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond)
            X0 = initialize([d,r], distr = 'Gaussian')


            ### Running HOMP
            tic = time.time()
            X_homp, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)
            time_homp =  time.time()-tic

            # Running ISTA_mix (init 0)
            lamb = 0.0005
            tic = time.time()
            _, cost_istam, err_istam, S_istam = ista_mix(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)
            time_istam = time.time()-tic

            # Test Ista
            lamb = 0.001
            tic = time.time()
            _, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)
            time_ista = time.time()-tic

            # Test IHTmix
            tic = time.time()
            X_iht, err_iht, S_iht = iht_mix(Y, D, B, k, X0, tol=tol)
            time_iht = time.time()-tic

            # Comparison with trick
            tic = time.time()
            X_trick, err_trick, S_trick = pseudo_trick(Y, D, B, k)
            time_trick = time.time()-tic

            # Compare with best solution of true support (S)
            tic = time.time()
            X_S, err_S = ls_kn_supp(Y, D, B, S, k)
            time_supportest = time.time()-tic

            # Storing
            dic = {
                "xp":"XP2",
                'n':6*[n], 'm':6*[m], 'd':6*[d], 'k':6*[k],
                'algorithm': ['HOMP', 'Mixed-FISTA', 'Block-FISTA', 'IHT', 'TrickOMP', 'Fixed Support'],
                'iteration': [len(err_homp), len(err_istam), len(err_ista), len(err_iht), 1, 1],
                'time': [time_homp, time_istam, time_ista, time_iht, time_trick, time_supportest],
            }
            store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

#todo automate processing of values

# Uncomment for storing outputs
# path designed for running from mscode root (where setup is located)
#year = 2021
#month = 10
#day = 21
#path = '../..'
#stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP7/{}_results'.format(path,stor_name))
