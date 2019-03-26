# Latent Factor Models Based on Linear Total Correlation Explanation (CorEx)

Linear CorEx finds latent factors that are as informative as possible about relationships in the data. 
The approach is described in this paper:
[Low Complexity Gaussian Latent Factor Models and a Blessing of Dimensionality](https://arxiv.org/abs/1706.03353).
This is useful for covariance estimation, clustering related variables, and dimensionality reduction, especially 
in the high-dimensional, under-sampled regime. 

To install:
```
pip install linearcorex
```

Mathematically, the objective is to find factors, y, where y = W x and  
x in R^n is the data and W is an m by n weight matrix. 
We are minimizing TC(X|Y) + TC(Y) where TC is the "total correlation" or multivariate mutual information. This objective
is optimized when X's are independent after conditioning on Y's, and the Y's themselves are independent. 
Instead of heuristically upper bounding this objective as we do for discrete CorEx, 
we are able to optimize it exactly in the linear case. 
While this extension required assumptions of linearity, the 
advantage is that the code is pretty fast since it only relies on matrix algebra. In principle it could be 
further accelerated using GPUs. 


Without further constraints, the optima of this objective 
may have an undesirable property: information about the X_i's can be stored "synergistically" in the latent factors. 
In other words, to predict a single variable you need to combine info from all the latent factors. Therefore, we 
add a constraint that the solutions should be non-synergistic (latent factors are individually informative about each variable X_i). 
This also recovers the property of the original lower bound formulation from AISTATS that each latent factor
has a non-negative added contribution towards TC.
Note that by default, we constrain solutions to eliminate synergy. 
But, you can turn it off by setting eliminate_synergy=False in the python API or -a from the command line. 
For making nice trees, it should be left on (e.g. for personality data or ADNI data). 

To test the command line interface, try:
```
cd $INSTALL_DIRECTORY/linearcorex/
python vis_corex.py ../tests/data/test_big5.csv --layers=5,1 --verbose=1 --no_row_names -o big5
python vis_corex.py ../tests/data/adni_blood.csv --layers=30,5,1 --missing=-1e6 --verbose=1 -o adni
python vis_corex.py ../tests/data/matrix.tcga_ov.geneset1.log2.varnorm.RPKM.txt --layers=30,5,1 --delimiter=' ' --verbose=1 --gaussianize="outliers" -o gene
```
Each of these examples generates pairwise plots of relationships and a graph. 

The python API uses the sklearn conventions of fit/transform.  
```python
import linearcorex as lc
import numpy as np

out = lc.Corex(n_hidden=5, verbose=True)  # A Corex model with 5 factors
X = np.random.random((100, 50))  # Random data with 100 samples and 50 variables
out.fit(X)  # Fit the model on data
y = out.transform(X)  # Transform data into latent factors
print(out.clusters)  # See the clusters
cov = out.get_covariance()  # The covariance matrix
```


Missing values can be specified, but are just imputed in a naive way. 

## Papers

See [Sifting Common Info...](https://arxiv.org/abs/1606.02307) and 
[Maximally informative representations...](https://arxiv.org/abs/1410.7404) for work building up to this method. 
The main paper describing the method is 
[Low Complexity Gaussian Latent Factor Models and a Blessing of Dimensionality](https://arxiv.org/abs/1706.03353).
The connections with the idea of "synergy" will be described in future work. 


### Troubleshooting visualization
For Mac users: 

To get the visualization of the hierarchy looking nice sometimes takes a little effort. To get graphs to compile correctly do the following. 
Using "brew" to install, you need to do "brew install gts" followed by "brew install --with-gts graphviz". 
The (hacky) way that the visualizations are produced is the following. The code, vis_corex.py, produces a text file called "graphs/graph.dot". This just encodes the edges between nodes in dot format. Then, the code calls a command line utility called sfdp that is part of graphviz, 

```
sfdp graph.dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=False -o graph_sfdp.pdf
```

These dot files can also be opened with OmniGraffle if you would like to be able to manipulate them by hand. 
If you want, you can try to recompile graphs yourself with different options to make them look nicer. Or you can edit the dot files to get effects like colored nodes, etc.

For Ubuntu users:

Credits: https://gitlab.com/graphviz/graphviz/issues/1237

1. Remove any existing installation with `conda uninstall graphviz`. (If you did not install with Conda, you might need to do `sudo apt purge graphviz` and/or `pip uninstall graphviz`).
    
2. run `sudo apt install libgts-dev`

3. run `sudo pkg-config --libs gts`
    
4. run `sudo pkg-config --cflags gts`

5. Download `graphviz-2.40.1.tar.gz` from [here](https://graphviz.gitlab.io/pub/graphviz/stable/SOURCES/graphviz.tar.gz)

6. Navigate to directory containing download, and extract with `tar -xvf graphviz-2.40.1.tar.gz` (or newer whatever the download is named.)

7. `cd` into extracted folder (ie `cd graphviz-2.40.1`) and run `sudo ./configure --with-gts`

8. Run `sudo make` in the folder

9. Run `sudo make install` in the folder

10. Reinstall library using `pip install graphviz`
    
 



