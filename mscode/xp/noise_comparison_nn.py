# recovery (and error) vs noise for block-FISTA and its nonnegative version

# Questions:
# - test two distributions for X: Gaussian, and decreasing

# - to choose lambda(s), we fix according to average best one from a set of experiments using the same settings on the fly. The grid is very coarse. In practice, use cross-validation.
# - We initialize with 1 zero init, cf init tests for more details

# Reasonable dimensions for reasonable runtime
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations, product
import shelve
from mscode.utils.utils import count_support, redundance_count, find_lambda
from mscode.methods.algorithms import ista, ista_nn
from mscode.utils.generator import gen_mix, initialize
import plotly.express as px
import pandas as pd

# Problem parameters
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

# We run the tests several times since performances are very problem-dependent
Nbdata = 50


# Recovery and error versus noise
grid_SNR = [1000, 100, 50, 40, 30, 20, 15, 10, 5, 2, 0] #[40, 20]
grid_lambda = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Storage variables
l_SNR = len(grid_SNR)
# Store results in Pandas DataFrame
store_pd = pd.DataFrame(columns=["value", "algorithm",'SNR'])
#stock_ista_nn = np.zeros([l_SNR,Nbdata,2])
#stock_ista = np.zeros([l_SNR,Nbdata,2])


for (i,SNR) in enumerate(grid_SNR):
    print('SNR', SNR, 'dB')
    # run 3 checks for lambda, to find a reasonable value
    store_lamb = []
    store_lamb_nn = []
    for iter in range(3):
        #store_lamb_nn.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista_nn'))
        store_lamb.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista'))
    store_lamb_nn = store_lamb
    lamb = np.median(store_lamb)
    lamb_nn = np.median(store_lamb_nn)
    print('lambda ratio is', lamb, 'and for nonneg', lamb_nn)
    for j in range(Nbdata):
        # Generate data
        Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr = distr)
        # The default zero init
        X0 = initialize([d,r], distr = 'Zeros')

        # Fista and Fista_mix
        # Running ISTA_mix (init 0)
        X_istann, _, err_ista_nn, S_ista_nn = ista_nn(Y, D, B, lamb_nn, k=k, X0=X0, verbose=False, tol=tol)
        #stock_ista_nn[i, j, 0]=count_support(S, S_ista_nn)
        #stock_ista_nn[i, j, 1]=np.linalg.norm(X - X_istann, 'fro')

        # Test Ista
        X_ista, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0, verbose=False, tol=tol)
        #stock_ista[i, j, 0]=count_support(S, S_ista)
        #stock_ista[i, j, 1]=np.linalg.norm(X - X_ista, 'fro')

        dic = {
        'value':[count_support(S, S_ista), count_support(S, S_ista_nn)],
        'algorithm': ['Block-FISTA', 'nonnegative Block-FISTA'],
        'SNR': 2*[SNR]
        }
        store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)


# Plots
fig = px.box(store_pd, x='SNR', y='value', facet_col='algorithm', color='algorithm', title="",  labels={'value':'Support recovery'} )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_xaxes(type='category')
fig.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1000,
    height=400,
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False)
fig.show()

# Uncomment for storing outputs
# path designed for running from mscode root (where setup is located)
#year = 2021
#month = 10
#day = 25
#path = '../..'
#stor_name = '{}-{}-{}'.format(year,month,day)
#stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XPnn/{}_results'.format(path,stor_name))
#fig.write_image('{}/data/XPnn/{}_plot.pdf'.format(path,stor_name))
