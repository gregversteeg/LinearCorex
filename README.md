# Linear Total Correlation Explanation (CorEx)

While this version of CorEx makes some strong assumptions: latent factors are linear functions of the data and we assume that the input 
 is drawn from a multivariate Gaussian, the algorithm is superfast, stable, and scalable. It also avoids some assumptions of previous versions. 
 
Instead of lower bounding TC_L(X;Y), we are able to optimize it exactly. It turns out that the optima of this objective 
tend to have an undesirable property: information about the X_i's is stored in a super-additive way in the latent factors. 
In other words, to predict a single variable you need to combine info from all the latent factors. Therefore, we 
add a constraint that the solutions should be additive (information in latent factors about each variable X_i adds). 
This also recovers the property of the original lower bound formulation from AISTATS that each latent factor
has a non-negative added contribution towards TC. 

We are maximizing TC_L(X;Y) for y = W x, where x is in R^n, y is in R^m and W is an m by n weight matrix. 
The additivity constraint is that I(X_i;Y)  = sum_j I(X_i;Y_j), for each i. 

In the tests folder, try:
```
python test_weak_correlations.py
python test_faces.py
python vis_corex.py tests/data/test_big5.csv --n_hidden=5 -v --no_row_names -o big5
python vis_corex.py tests/data/adni_blood.csv --n_hidden=30 --missing=-1e6 -v -o adni
```
Each of these examples generates pairwise plots of relationships and a graph. 

Note that missing values are imputed in vis_sieve beforehand. If you are using the command line API,
you should impute missing values manually (as the mean within each column). 