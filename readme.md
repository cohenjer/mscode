# MScode

This repository contains the mscode package.

It can be easily installed from this location by
```
pip install . -r requirements.txt
```

The full documentation is available under doc/build/html/index.html

It contains implementations of several algorithms to solve the Mixed Sparse Coding problem as described in "Dictionary-based low-rank approximation and the mixed sparse coding problem", Jeremy E. Cohen, 2021, [arxiv link](https://arxiv.org/abs/2111.12399). Please cite this reference if you use this package. The exact version of the code used for the publication is commit 40dadb43742e4536fe719fbc91c691c3ae5e6148, tagged as version v0.0.

It also contains the numerical results of the Mixed Sparse Coding experiments from the above reference stored as pandas dataframes, figures plotted from these data, and the script to reproduce these results. To run these scripts, set the root to ./mscode/xp/ and run
`python xxx.py`
where xxx stands for the name of the script to run. These scripts will write their output to ./mscode/data/ so please change the relevant paths in the scripts to avoid overwritting existing results.

This repository also includes a copy of the prox-l1oo distributed at this repository:
https://github.com/bbejar/prox-l1oo
and which was developped in Bejar, Benjamin, Ivan Dokmanic, and Rene Vidal. "The fastest L1, oo prox in the west." IEEE transactions on pattern analysis and machine intelligence (2021).

credits: J.E.Cohen, 2021
