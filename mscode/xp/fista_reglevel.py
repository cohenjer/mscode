# Recov vs best lamb/lamb_max  (generate randomly, find best, report scores, show histogram of realizations in background with second origin)
# Show av. Recovery vs deviations from best (0 dev means average from previous curve)
# Do for all Fistamix and Fista
# Check redundancy rate
import numpy as np
from matplotlib import pyplot as plt
#from itertools import combinations, product
#import shelve
from mscode.utils.utils import count_support
from mscode.methods.algorithms import ista_mix, ista
from mscode.utils.generator import gen_mix, initialize
import pandas as pd
import plotly.express as px

# Problem parameters
k = 5
r = 6 #2
n = 50 #10
m = 50 #20
d = 100 #50
#noise = 0.03 # 0.03
SNR = 20  # dB
cond = 2*1e2
tol = 1e-6
distr = 'Gaussian'

# Grid on lambda/lambda_max (11 values)
#grid = np.linspace(0, 0.1, 11, endpoint=True)
grid = [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
print(grid)
# Storage
Ntest = 200
labels=['-90%', '-50%', '-20%', 'optimal', '+100%', '+400%', '+900%']
# Storing with Pandas DataFrame
store_pd = pd.DataFrame(columns=["xp", "value", "algorithm", "alpha", "alpha_mod"])

# Loop on tensors, test 100
for i in range(Ntest):
    print(i)
    # Data generation
    Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, distr=distr)
    X0 = initialize([d,r], distr = 'Zeros')
    # initial recovery rates and best lambdas
    recov_istam = 0
    recov_ista = 0
    lamb_best_istam = 0
    lamb_best_ista = 0
    cnt = 1
    cnt_m = 1

    for lamb in grid:
        # Running FIstas
        _, _, _, S_istam = ista_mix(Y, D, B, lamb, k=k, X0=X0, verbose=False, tol=tol)
        _, _, _, S_ista = ista(Y, D, B, lamb, k=k, X0=X0, verbose=False, tol=tol)

        # Computing recovery rates, store if larger
        temp_recov_istam = count_support(S, S_istam)
        if temp_recov_istam > recov_istam:
            lamb_best_istam = lamb
            recov_istam = temp_recov_istam
        elif temp_recov_istam == recov_istam:
            # store for averaging
            cnt_m += 1
            lamb_best_istam += lamb

        temp_recov_ista = count_support(S, S_ista)
        if temp_recov_ista > recov_ista:
            lamb_best_ista = lamb
            recov_ista = temp_recov_ista
        elif temp_recov_ista == recov_ista:
            # store for averaging
            cnt += 1
            lamb_best_ista += lamb


    lamb_best_ista = lamb_best_ista/cnt
    lamb_best_istam = lamb_best_istam/cnt_m

    dic = {
        "xp":2*['xp1'],
        "alpha": [lamb_best_ista, lamb_best_istam],
        "alpha_mod": [lamb_best_ista, lamb_best_istam],
        "value": [recov_ista, recov_istam],
        "algorithm": ["Block-FISTA","Mixed-FISTA"]
    }
    store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)


    # Robustness to lambda trials
    # log'ish' scale
    grid2 = np.array([-9*lamb_best_ista/10, -5*lamb_best_ista/10, -2*lamb_best_ista/10, 0, lamb_best_ista, 5*lamb_best_ista, 10*lamb_best_ista]) + lamb_best_ista


    for (cnt, lamb) in enumerate(grid2):
        # Running Fista on a local optimal grid
        _, _, _, S_ista = ista(Y, D, B, lamb, k=k, X0=X0, verbose=False, tol=tol)
        recov_ista = count_support(S, S_ista)
        # TODO
        dic = {
            "xp":1*['xp2'],
            "alpha": [lamb_best_ista],
            "alpha_mod": [labels[cnt]],
            "value": [recov_ista],
            "algorithm": ["Block-FISTA"]
        }
        store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)


    for (cnt, lamb) in enumerate(grid2):
        # Running Fista mix on a local optimal grid
        _, _, _, S_istam = ista_mix(Y, D, B, lamb, k=k, X0=X0, verbose=False, tol=tol)
        recov_istam = count_support(S, S_istam)
        #storage_onedatam.append(recov_istam)
        dic = {
            "xp":1*['xp2'],
            "alpha": [lamb_best_istam],
            "alpha_mod": [labels[cnt]],
            "value": [recov_istam],
            "algorithm": ["Mixed-FISTA"]
        }
        store_pd = store_pd.append(pd.DataFrame(dic), ignore_index=True)

    #storage_robustm.append(storage_onedatam)


# end loop on data matrices

# Starting plots
# lambda effect, maybe todo improve?
#storage = np.array(storage)
#storage_m = np.array(storage_m)
#plt.figure()
#plt.scatter(np.log10(storage[:,0]),storage[:,1], color='r')
#plt.scatter(np.log10(storage_m[:,0]),storage_m[:,1], color='b')
#plt.legend(['Block Fista','Mixed Fista'])
df1 = store_pd[store_pd.xp == 'xp1']
df2 = store_pd[store_pd.xp == 'xp2']

fig = px.scatter(df1, x='alpha', y='value', color='algorithm', title="",  labels={'value': 'Support recovery', 'alpha': 'regularization strength alpha'},log_x=True)
#fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
#fig.update_xaxes(type='category')
fig.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1000,
    height=400,
    #paper_bgcolor="white",#'rgb(233,233,233)',
    #plot_bgcolor="white",#'rgb(233,233,233)',
)
fig.show()


fig2 = px.box(df2, x='alpha_mod', y='value', color='algorithm', facet_col='algorithm', title="",  labels={'value': 'Support recovery', 'alpha_mod': 'variation wrt optimal  alpha'})#, log_x=True)
#fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig2.update_xaxes(type='category')
fig2.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    autosize=False,
    width=1000,
    height=400,
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
)
fig2.show()

# note: because the grid for finding the optimal lambda percentage is grosser than the second refinement grid, also the grids are different, it is possible that around the optimal value we still improve. Optimal is only wrt the first grid search.


# Uncomment for storing outputs
# path designed for running from mscode root (where setup is located)
#year = 2021
#month = 10
#day = 25
#path = '../..'
#stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP5/{}_results'.format(path,stor_name))
#fig.write_image('{}/data/XP5/{}_plot.pdf'.format(path,stor_name))
#fig2.write_image('{}/data/XP5/{}_plot2.pdf'.format(path,stor_name))
