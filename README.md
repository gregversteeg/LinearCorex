# Linear Total Correlation Explanation (CorEx)

While this version of CorEx makes some strong assumptions: latent factors are linear functions of the data and we assume that the input 
 is drawn from a multivariate Gaussian, the algorithm is fast and scalable. 
  It avoids heuristic assumptions about estimating connectivity (called alpha in previous corex work). 
 
Instead of lower bounding TC_L(X;Y), we are able to optimize it exactly in the linear case. It turns out that the optima of this objective 
may have an undesirable property: information about the X_i's can be stored in a super-additive way in the latent factors. 
In other words, to predict a single variable you need to combine info from all the latent factors. Therefore, we 
add a constraint that the solutions should be additive (information in latent factors about each variable X_i adds). 
This also recovers the property of the original lower bound formulation from AISTATS that each latent factor
has a non-negative added contribution towards TC. (And it helps recover/surpass the results of the information sieve,
currently under submission. Please ask if you'd like to see the draft of that.) 

Note that by default, we add this additivity constraint. The additivity constraint is that I(X_i;Y)  = sum_j I(X_i;Y_j), for each i. 
But, you can turn it off by setting additive=False in the python API or -a from the command line. 
For making nice trees, it should be left on (e.g. for personality data or ADNI data). 

We are maximizing TC_L(X;Y) for y = W x, where x is in R^n, y is in R^m and W is an m by n weight matrix. 

In the tests folder, try:
```
python test_weak_correlations.py
python vis_corex.py tests/data/test_big5.csv --layers=5,1 -v --no_row_names -o big5
python vis_corex.py tests/data/adni_blood.csv --layers=30,5,1 --missing=-1e6 -v -o adni
python vis_corex.py tests/data/matrix.tcga_ov.geneset1.log2.varnorm.RPKM.txt --layers=30,5,1 --delimiter=' ' -v -o gene
```
Each of these examples generates pairwise plots of relationships and a graph. 

Note that missing values are imputed in vis_sieve beforehand. If you are using the python API,
you should impute missing values manually (as the mean within each column). 