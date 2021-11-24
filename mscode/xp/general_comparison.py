# recovery (and error) vs noise for all algorithms
# recovery (and error) vs condB for all algorithms
# recovery vs (k,d) for all algorithms (heatmap)

# todo: also condD?

# Questions:
# - test two distributions for X: Gaussian, and decreasing

# - to choose lambda(s), we fix according to average best one from a set of experiments using the same settings on the fly. The grid is very coarse. In practice, use cross-validation.
# - We initialize with 1 zero init, cf init tests for more details

# Reasonable dimensions for reasonable runtime
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations, product
import shelve
import pandas as pd
from mscode.utils.utils import count_support, redundance_count, find_lambda
from mscode.methods.algorithms import iht_mix, homp, omp, ls_kn_supp, pseudo_trick, brute_trick, ista_mix, ista, admm_mix
from mscode.utils.generator import gen_mix, initialize
import plotly.express as px

# Random seeding
np.random.seed(seed=0)

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

# Store results in Pandas DataFrame
store_pd = pd.DataFrame(columns=["xp", "value", "algorithm", "error type", "SNR", "lambda", "k", "r", "d", "m", "n", "cond"])


for SNR in grid_SNR:
    print('SNR', SNR, 'dB')
    # run 3 checks for lambda, to find a reasonable value
    store_lamb = []
    store_lamb_m = []
    for iter in range(3):
        store_lamb_m.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista_m'))
        store_lamb.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista'))
    lamb = np.median(store_lamb)
    lamb_m = np.median(store_lamb_m)
    print('lambda ratio is', lamb, 'and for mixed', lamb_m)
    for j in range(Nbdata):

        # Generate data
        Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr = distr)

        # The default zero init
        X0 = initialize([d,r], distr = 'Zeros')

        # Running algorithms
        X_istam, _, err_ista_m, S_ista_m = ista_mix(Y, D, B, lamb_m, k=k, X0=X0, verbose=False, tol=tol)
        X_ista, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0, verbose=False, tol=tol)
        X_homp, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)
        X_iht, err_iht, S_iht = iht_mix(Y, D, B, k, X0, tol=tol)
        X_trick, err_trick, S_trick = pseudo_trick(Y, D, B, k)

        # Storing results
        dic={
            'xp':10*['XP1'],
            'value':[count_support(S, S_ista_m), count_support(S, S_ista), count_support(S, S_homp), count_support(S, S_iht), count_support(S, S_trick)]+
            [np.linalg.norm(X - X_istam), np.linalg.norm(X - X_ista), np.linalg.norm(X - X_homp), np.linalg.norm(X - X_iht), np.linalg.norm(X - X_trick)],
            'algorithm': 2*['Mixed-FISTA', 'Block-FISTA', 'HOMP', 'IHT', 'TrickOMP'],
            "error type": 5*['support recovery']+5*['reconstruction error'],
            "SNR":10*[SNR], "lambda":2*[lamb, lamb_m,0,0,0],
            "k":10*[k], "r":10*[r], "d":10*[d], "m":10*[m], "n":10*[n], "cond":10*[condB],
        }
        store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

## Recovery and error versus conditionning
SNR = 20
grid_cond = [1, 10, 50, 100, 5*1e2, 1e3, 5*1e3, 1e4, 5*1e4, 1e5]

for cond in grid_cond:
    print('cond', cond)
    # run 3 checks for lambda, to find a reasonable value
    store_lamb = []
    store_lamb_m = []
    for iter in range(3):
        store_lamb_m.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista_m'))
        store_lamb.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista'))
    lamb = np.median(store_lamb)
    lamb_m = np.median(store_lamb_m)
    print('lambda ratio is', lamb, 'and for mixed', lamb_m)
    for j in range(Nbdata):
        # Generate data
        Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr=distr)
        # The default zero init
        X0 = initialize([d,r], distr = 'Zeros')

        # Running algorithms
        X_istam, _, err_ista_m, S_ista_m = ista_mix(Y, D, B, lamb_m, k=k, X0=X0,  verbose=False, tol=tol)
        X_ista, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)
        X_homp, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)
        X_iht, err_iht, S_iht = iht_mix(Y, D, B, k, X0, tol=tol)
        X_trick, err_trick, S_trick = pseudo_trick(Y, D, B, k)

        dic={
            'xp':10*['XP2'],
            'value':[count_support(S, S_ista_m), count_support(S, S_ista), count_support(S, S_homp), count_support(S, S_iht), count_support(S, S_trick)]+
            [np.linalg.norm(X - X_istam), np.linalg.norm(X - X_ista), np.linalg.norm(X - X_homp), np.linalg.norm(X - X_iht), np.linalg.norm(X - X_trick)],
            'algorithm': 2*['Mixed-FISTA', 'Block-FISTA', 'HOMP', 'IHT', 'TrickOMP'],
            "error type": 5*['support recovery']+5*['reconstruction error'],
            "SNR":10*[SNR], "lambda":2*[lamb, lamb_m,0,0,0],
            "k":10*[k], "r":10*[r], "d":10*[d], "m":10*[m], "n":10*[n], "cond":10*[np.round(condB,3)],
        }
        store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

## Recovery and error versus (k,d)
cond = 5*1e2
grid_k = [1, 2, 5, 10, 20]
grid_d = [20, 50, 100, 200, 400]

for d in grid_d:
    for k in grid_k:
        print('(d,k) is', d, k)
        # run 3 checks for lambda, to find a reasonable value
        store_lamb = []
        store_lamb_m = []
        for iter in range(3):
            store_lamb_m.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista_m'))
            store_lamb.append(find_lambda((m,n,d,k,r,SNR,cond), grid_lambda, 'Fista'))
        lamb = np.median(store_lamb)
        lamb_m = np.median(store_lamb_m)
        print('lambda ratio is', lamb, 'and for mixed', lamb_m)
        for j in range(Nbdata):
            # Generate data
            Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr=distr)
            # The default zero init
            X0 = initialize([d,r], distr = 'Zeros')

            # Running algorithms
            X_istam, _, err_ista_m, S_ista_m = ista_mix(Y, D, B, lamb_m, k=k, X0=X0,  verbose=False, tol=tol)
            X_ista, _, err_ista, S_ista = ista(Y, D, B, lamb, k=k, X0=X0,  verbose=False, tol=tol)
            X_homp, err_homp, S_homp = homp(Y, D, B, k, X0, tol=tol)
            X_iht, err_iht, S_iht = iht_mix(Y, D, B, k, X0, tol=tol)
            X_trick, err_trick, S_trick = pseudo_trick(Y, D, B, k)

            # Storing results
            dic={
                'xp':10*['XP3'],
                'value':[count_support(S, S_ista_m), count_support(S, S_ista), count_support(S, S_homp), count_support(S, S_iht), count_support(S, S_trick)]+
                [np.linalg.norm(X - X_istam), np.linalg.norm(X - X_ista), np.linalg.norm(X - X_homp), np.linalg.norm(X - X_iht), np.linalg.norm(X - X_trick)],
                'algorithm': 2*['Mixed-FISTA', 'Block-FISTA', 'HOMP', 'IHT', 'TrickOMP'],
                "error type": 5*['support recovery']+5*['reconstruction error'],
                "SNR":10*[SNR], "lambda":2*[lamb, lamb_m,0,0,0],
                "k":10*[k], "r":10*[r], "d":10*[d], "m":10*[m], "n":10*[n], "cond":10*[condB],
            }
            store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

df1 = store_pd[store_pd.xp=='XP1']
df2 = store_pd[store_pd.xp=='XP2']
df3 = store_pd[store_pd.xp=='XP3']

fig = px.box(df1[df1['error type']=='support recovery'], x='SNR', y='value', facet_col='algorithm', color='algorithm', title="Support recovery versus SNR",  labels={'value':'Support recovery'})
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_xaxes(type='category')
fig.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1000,
    height=400,
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis2=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis3=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis4=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis5=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False,
)
fig.show()

fig2 = px.box(df2[df2['error type']=='support recovery'], x='cond', y='value', color='algorithm', facet_col='algorithm', title="Support recovery versus conditionning of B", labels={'value':'Support recovery'})
fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig2.update_xaxes(type='category')
fig2.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1000,
    height=400,
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis2=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis3=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis4=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    yaxis5=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False,
)
fig2.show()


# Normalizing the support recovery scores
fig3=px.density_heatmap(df3[df3['error type']=='support recovery'], x='d', y='k', z='value', facet_col='algorithm', color_continuous_scale='Viridis', histfunc="avg", labels={'value':'Support recovery'}, title='Recovery for varying sparsity and dictionary size')
fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig3.update_xaxes(type='category')
fig3.update_yaxes(type='category')
fig3.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1000,
    height=310,
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
)
fig3.show()


year = 2021
month = 10
day = 20
path = '../..'
stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP1/{}_results'.format(path,stor_name))
#fig.write_image('{}/data/XP1/{}_plot1.pdf'.format(path,stor_name))
#fig2.write_image('{}/data/XP1/{}_plot2.pdf'.format(path,stor_name))
#fig3.write_image('{}/data/XP1/{}_plot3.pdf'.format(path,stor_name))
# for Frontiers export
#fig.write_image('{}/data/XP1/{}_plot1.jpg'.format(path,stor_name))
#fig2.write_image('{}/data/XP1/{}_plot2.jpg'.format(path,stor_name))
#fig3.write_image('{}/data/XP1/{}_plot3.jpg'.format(path,stor_name))

# to load data
#store_pd = pd.read_pickle('{}/data/XP1/{}_results'.format(path,stor_name))
