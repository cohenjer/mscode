# For all functions, boxplot with normal and uniform inits, only error. Keep problem constant, Fista should not change results.
import numpy as np
from matplotlib import pyplot as plt
#from itertools import combinations, product
#import shelve
from mscode.utils.utils import count_support
from mscode.methods.algorithms import iht_mix, homp, omp, ls_kn_supp, pseudo_trick, brute_trick, ista_mix, ista, admm_mix
from mscode.utils.generator import gen_mix, initialize
import pandas as pd
import plotly.express as px

# Problem parameters
k = 5 #2
r = 6 #2
n = 50 #10
m = 50 #20
d = 100 #50
#noise = 0.03 # 0.03
SNR = 20  # dB
cond = 2*1e2
tol = 1e-6
lamb_m = 0.001
lamb= 0.005
distr='Uniform' #always uniform here
init_distr='Gaussian'

# Store results in Pandas DataFrame
store_pd = pd.DataFrame(columns=["value", "algorithm", "init type", "data number"])

# Generate a few data matrices, and for each run 10 init
Nbdata = 10
Nbinit = 10

for i in range(Nbdata):
    print(i)
    Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr=distr)

    # Zero initialization test
    X0z = initialize([d,r], distr= 'Zeros')

    # IHT
    _, err_iht, S_iht = iht_mix(Y, D, B, k, X0z, tol=tol)

    # Comparison with trick (independent of init)
    _, err_trick, S_trick = pseudo_trick(Y, D, B, k)

    # Running HOMP
    _, err_homp, S_homp = homp(Y, D, B, k, X0z, tol=tol)

    # Running ISTA_mix (init 0)
    _, _, err_ista_m, S_ista_m = ista_mix(Y, D, B, lamb_m, k=k, X0=X0z,  verbose=False, tol=tol)

    # Test Ista
    _, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0z,  verbose=False, tol=tol)


    dic = {
    'value':[count_support(S, S_ista_m), count_support(S, S_ista), count_support(S, S_homp), count_support(S, S_iht), count_support(S, S_trick)],
    'algorithm': ['Mixed-FISTA', 'Block-FISTA', 'HOMP', 'IHT', 'TrickOMP'],
    "init type": 5*['zero init'],
    "data number": 5*[i]
    }
    store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

    # Initialise Loop
    for j in range(Nbinit):
        if init_distr=='Gaussian':
            X0 = initialize([d,r], distr = 'Gaussian')
        else:
            X0 = initialize([d,r], distr = 'Uniform')

        # Running all algorithms
        # IHT
        _, err_iht, S_iht = iht_mix(Y, D, B, k, X0, tol=tol)

        # Running HOMP
        _, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)

        # Running ISTA_mix (init 0)
        _, _, err_ista_m, S_ista_m = ista_mix(Y, D, B, lamb_m, k=k, X0=X0,  verbose=False, tol=tol)

        # Test Ista
        _, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)

        dic = {
        'value':[count_support(S, S_ista_m), count_support(S, S_ista), count_support(S, S_homp), count_support(S, S_iht)],
        'algorithm': ['Mixed-FISTA', 'Block-FISTA', 'HOMP', 'IHT'],
        "init type": 4*['random init'],
        "data number": 4*[i]
        }
        store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)


fig = px.box(store_pd, x='data number', y='value', facet_col='algorithm', color='init type', title="Robustess to random initializations for 10 problem instances",  labels={'value':'Support recovery', 'data number': 'problem index'}, notched=True)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_xaxes(type='category')
fig.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1500,
    height=500,
    #paper_bgcolor="white",#'rgb(233,233,233)',
    #plot_bgcolor="white",#'rgb(233,233,233)',
)
fig.show()

# Note: ISta algorithms are convex and should not depend too much on the initialization (they still do, maybe run longer?).
# Uncomment for storing outputs
# path designed for running from mscode root (where setup is located)
#year = 2021
#month = 10
#day = 22
#path = '../..'
#stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP4/{}_results'.format(path,stor_name))
#fig.write_image('{}/data/XP4/{}_plot.pdf'.format(path,stor_name))
