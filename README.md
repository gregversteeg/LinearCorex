# Linear Total Correlation Explanation (CorEx)

Linear CorEx finds latent factors that are as informative as possible about relationships in the data. 
The approach is described in this paper:
[Low Complexity Gaussian Latent Factor Models and a Blessing of Dimensionality](https://arxiv.org/abs/1706.) 
This is useful for covariance estimation, clustering related variables, and dimensionality reduction. 

Previous versions of CorEx worked for discrete data, and could be applied to continuous data with some hacks. 
This is the first truly continuous version of CorEx. While this extension required assumptions of linearity, the 
advantage is that the code is pretty fast since it only relies on matrix algebra. In principle it could be 
further accelerated using GPUs. 

Instead of lower bounding TC_L(X;Y) as we do for discrete CorEx, 
we are able to optimize it exactly in the linear case. It turns out that the optima of this objective 
may have an undesirable property: information about the X_i's can be stored "synergistically" in the latent factors. 
In other words, to predict a single variable you need to combine info from all the latent factors. Therefore, we 
add a constraint that the solutions should be non-synergistic (latent factors are individually informative about each variable X_i). 
This also recovers the property of the original lower bound formulation from AISTATS that each latent factor
has a non-negative added contribution towards TC. (And it helps recover/surpass the results of the information sieve.) 

Note that by default, we constrain solutions to eliminate synergy. 
But, you can turn it off by setting eliminate_synergy=False in the python API or -a from the command line. 
For making nice trees, it should be left on (e.g. for personality data or ADNI data). 

We are maximizing TC_L(X;Y) for y = W x, where x is in R^n, y is in R^m and W is an m by n weight matrix. 

In the tests folder, try:
```
python test_weak_correlations.py
python vis_corex.py tests/data/test_big5.csv --layers=5,1 --verbose=1 --no_row_names -o big5
python vis_corex.py tests/data/adni_blood.csv --layers=30,5,1 --missing=-1e6 --verbose=1 -o adni
python vis_corex.py tests/data/matrix.tcga_ov.geneset1.log2.varnorm.RPKM.txt --layers=30,5,1 --delimiter=' ' --verbose=1 --gaussianize="outliers" -o gene
```
Each of these examples generates pairwise plots of relationships and a graph. 

Missing values can be specified, but are just imputed in a naive way. 

## Papers

See [Sifting Common Info...](https://arxiv.org/abs/1606.02307) and 
[Maximally informative representations...](https://arxiv.org/abs/1410.7404) for work building up to this method. 
The main paper describing the method is [Low Complexity Gaussian Latent Factor Models and a Blessing of Dimensionality](https://arxiv.org/abs/1706.)
The connections with the idea of "synergy" will be described in future work. 


### Troubleshooting visualization
To get the visualization of the hierarchy looking nice sometimes takes a little effort. To get graphs to compile correctly do the following. 
Using "brew" to install, you need to do "brew install gts" followed by "brew install --with-gts graphviz". 
The (hacky) way that the visualizations are produced is the following. The code, vis_corex.py, produces a text file called "graphs/graph.dot". This just encodes the edges between nodes in dot format. Then, the code calls a command line utility called sfdp that is part of graphviz, "sfdp graph.dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=False -o graph_sfdp.pdf".
These dot files can also be opened with OmniGraffle if you would like to be able to manipulate them by hand. 
If you want, you can try to recompile graphs yourself with different options to make them look nicer. Or you can edit the dot files to get effects like colored nodes, etc. 



