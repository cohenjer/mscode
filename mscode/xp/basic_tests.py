import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations, product
import shelve
from mscode.utils.utils import count_support, redundance_count
from mscode.methods.algorithms import iht_mix, homp, omp, ls_kn_supp, pseudo_trick, brute_trick, ista_mix, ista, admm_mix, ista_nn
from mscode.utils.generator import gen_mix, initialize

# Generation
k = 5 #2
r = 6 #2
n = 50 #10
m = 50 #20
d = 100 #50
#noise = 0.03 # 0.03
SNR = 20  # dB
cond = 2*1e2
distr = 'Uniform'
tol = 1e-6

Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, cond_rand=False, distr=distr)

X0 = initialize([d,r], distr = 'Uniform')

# Running HOMP
print('Using HOMP\n')
X_homp, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)

# Running ISTA_mix (init 0)
print('Using Ista(s)\n')
Xinit = X0
lamb_rel = 0.001
X_ista, cost_ista, err_ista, S_ista = ista_mix(Y, D, B, lamb_rel, k=k, X0=Xinit, verbose=False, tol=tol)
# checkup sparsity level
#nz_level = np.sum(np.abs(X_ista) > 1e-16, axis=0)
#print('There are ', nz_level, 'nonzeros in columns of X_ista')

## Same with larger lambda
#lamb_rel = 0.2
#X_ista2, cost_ista_m, err_ista2, S_ista2 = ista_mix(Y, D, B, lamb_rel, k=k, X0=Xinit, tol=tol, verbose=False)
## checkup sparsity level
##nz_level2 = np.sum(np.abs(X_ista2) > 1e-16, axis=0)
##print('There are ', nz_level2, 'nonzeros in columns of X_ista')

# Test Ista (nn)
lamb_rel = 0.001
X_ista0, cost_ista0, err_ista0, S_ista0 = ista_nn(Y, D, B, lamb_rel, k=k, X0=Xinit, tol=tol, verbose=False)
#X_ista0, cost_ista0, err_ista0, S_ista0 = ista(Y, D, B, lamb_rel, k=k, X0=Xinit, tol=tol, verbose=False)
# checkup sparsity level
#nz_level0 = np.sum(np.abs(X_ista0) > 1e-16, axis=0)
#print('There are ', nz_level0, 'nonzeros in columns of X_ista')

# Test IHTmix
print('Using IHTmix\n')
eta = 1/np.linalg.norm(D.T@D)
X_iht, err_iht, S_iht = iht_mix(Y, D, B, k, X0, eta=eta, tol=1e-8, itermax=1000)

# Test ADMM
print('using ADMM')
X_adm, err_adm, S_adm, Z_adm, err_Z = admm_mix(Y, D, B, k, X0)

# Comparison with trick
print('Using PseudoTrick\n')
X_trick, err_trick, S_trick = pseudo_trick(Y, D, B, k)

# HOMP with Trick init test
print('Using HOMP Trick init')
X_homp2, err_homp2, S_homp2 = homp(Y, D, B, k, X_trick)

# Compare with best solution of true support (S)
X_S, err_S = ls_kn_supp(Y, D, B, S, k)


# Plotting error curves of iterative algorithms
plt.figure()
plt.semilogy(cost_ista)
plt.semilogy(cost_ista0)
plt.semilogy(err_homp)
plt.semilogy(err_iht)
plt.semilogy(err_adm)
plt.legend(['Mixed Fista','Block Fista', 'HOMP', 'IHT', 'ADMM'])

# checking support
#for i in range(r):
#    print('True', np.sort(S[i]))
#    print('IHT', np.sort(S_iht[:,i]))
#    print('ADMM', np.sort(S_adm[:,i]))
#    print('HOMP', np.sort(S_homp[:,i]))
#    print('ISTAm', np.sort(S_ista[:,i]))
#    print('ISTAm large', np.sort(S_ista2[:,i]))
#    print('ISTA', np.sort(S_ista0[:,i]))
#    print('OMPcol', np.sort(S_trick[:,i]))
#    print('HOMPtrickinit', np.sort(S_homp2[:,i]), '\n')
#
# Error prints
print('Reconstruction errors \n')
print('TrueS error', err_S)
print('IHT   error', err_iht[-1])
print('ADMM   error', err_adm[-1])
print('FISTA_mix (small lambda)   error', err_ista[-1])
#print('FISTA_mix (large lambda)   error', err_ista2[-1])
print('FISTA error', err_ista0[-1])
print('HOMP  error', err_homp[-1])
print('Trick error', np.linalg.norm(Y-D@X_trick@B.T, 'fro'))
print('HOMPi error', err_homp2[-1])

# Support evaluations
print('support scores')
print('Fista_m ', count_support(S, S_ista))
#print('Fista_m large', count_support(S, S_ista2))
print('Fista', count_support(S, S_ista0))
print('HOMP ', count_support(S, S_homp))
print('HOMP init trick ', count_support(S, S_homp2))
print('OMP trick ', count_support(S, S_trick))
print('FIHT ', count_support(S, S_iht))
print('ADMM ', count_support(S, S_adm))

# Checking redudancy levels
print('True redundance', redundance_count(S))
print('Fista_m redundance', redundance_count(S_ista))
#print('Fista_m large redundance', redundance_count(S_ista2))
print('Fista redundance', redundance_count(S_ista0))
print('HOMP redundance', redundance_count(S_homp))
print('HOMPinit redundance', redundance_count(S_homp2))
print('Trick redundance', redundance_count(S_trick))
print('FIHT redundance', redundance_count(S_iht))
print('ADMM redundance', redundance_count(S_adm))
