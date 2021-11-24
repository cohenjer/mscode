# Check convergence for k>1
# Same for iht ?
import numpy as np
from matplotlib import pyplot as plt
#from itertools import combinations, product
#import shelve
from mscode.utils.utils import count_support
from mscode.methods.algorithms import iht_mix, homp
from mscode.utils.generator import gen_mix, initialize

# Problem parameters
# todo: implement a loader to choose these globally
k = 5 #2
r = 6 #2
n = 50 #10
m = 50 #20
d = 100 #50
#noise = 0.03 # 0.03
SNR = 20  # dB
cond = 2*1e2
tol = 1e-6
distr = 'Gaussian'

Nbruns = 4
Nbinit = 10  # not a working parameter at the moment
store_err = []
store_rec = []


for i in range(Nbruns):
    print(i)
    Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond)
    for j in range(Nbinit):
        X0 = initialize([d,r], distr = 'Gaussian')

        _, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)
        store_rec.append(count_support(S_homp, S))
        store_err.append(err_homp)

    X0 = initialize([d,r], distr= 'Zeros')
    _, err_homp, S_homp = homp(Y, D, B, k, X0)
    store_rec.append(count_support(S_homp, S))
    store_err.append(err_homp)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
plt.figure()
Nbinit = Nbinit+1
for i in range(Nbruns):
    for j in range(Nbinit):
        plt.semilogy(store_err[i*Nbinit+j], color=colors[i])
#plt.legend(['run1','run2', 'run3', 'run4', 'run5'])

plt.show()

# Note : zero init seems good for small and large values of k, but its nice to also try random init.

# Uncomment for storing outputs
# path designed for running from mscode root (where setup is located)
#year = 2021
#month = 3
#day = 16
#path = '.'
#stor_name = '{}-{}-{}'.format(year,month,day)
#np.save('{}/data/XP6/{}_{}_support_err'.format(path,stor_name,distr), store_err)
#np.save('{}/data/XP6/{}_{}_rec_err'.format(path,stor_name,distr), store_rec)
