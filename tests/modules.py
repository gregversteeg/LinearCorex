import numpy as np
import sklearn as sk
import sklearn.decomposition as skd
import sklearn.manifold as skm
import sklearn.cluster as skc
import sklearn.neural_network as skn
import sys
sys.path.append('/Users/gregv/deepcloud/LinearSieve')
import linearsieve as sieve
sys.path.append('/Users/gregv/deepcloud/LinearCorex')
import linear_corex as lc
from scipy.stats import kendalltau, pearsonr, spearmanr

methods = [ 
   #(name,type,f)
    ("LC", lambda m, q: lc.Corex(n_hidden=m, verbose=True, max_iter=5000).fit_transform(q)),
   ("Sieve", lambda m, q: sieve.Sieve(n_hidden=m, max_iter=100).fit_transform(q)),
   #("Spectral","c",cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack',affinity="nearest_neighbors")),
   #("LSA","d",TruncatedSVD(n_components=2)),
   # ("RBM", lambda m, q: skn.BernoulliRBM(n_components=m, learning_rate=0.01, n_iter=50).fit_transform(binarize(q))),
   #("NMF", lambda m, q: skd.ProjectedGradientNMF(n_components=m, init='random', random_state=0).fit_transform(-np.min(q) + q)),
   #("Factor Analysis", lambda m, q: skd.FactorAnalysis(n_components=m).fit_transform(q)),
   #("PCA", lambda m, q: skd.PCA(n_components=m).fit_transform(q)),
   #("ICA", lambda m, q: skd.FastICA(n_components=m).fit_transform(q)),
   #("KMeans", lambda m, q: skc.MiniBatchKMeans(n_clusters=m+1).fit_transform(q)[:,:m]),
   #("LLE", lambda m, q: skm.LocallyLinearEmbedding(n_components=m).fit_transform(q)),
   #("Isomap", lambda m, q: skm.Isomap(n_components=m).fit_transform(q)),
   # ("NMF2",ProjectedGradientNMF(n_components=1, init='random', random_state=0)),
   # ("LSA2",TruncatedSVD(n_components=1)),
   # ("PCA2",PCA(n_components=1)),
   # ("ICA3",FastICA(n_components=1)),
   # ("ICA4",FastICA(n_components=1)),
   # ("LLE2",LocallyLinearEmbedding(n_neighbors=3,n_components=1)),
   # ("Isomap2",Isomap(n_neighbors=40,n_components=1)),
   # #("MDSe", MDS(n_components=1,metric=False)),
   # ("Knn",KNeighborsClassifier(3)),
   # ("Linear SVM",SVC(kernel="linear", C=0.025)),
   # ("Decision Tree",DecisionTreeClassifier(max_depth=5)),
   # ("Random Forest",RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
   # ("Adaboost",AdaBoostClassifier()),
   # ("NB",BernoulliNB()),
   # ("Lin. Disc.","s",LDA()),
   # ("Log. Reg.","s",linear_model.LogisticRegression())
   ]

def binarize(q):
  return np.vstack([(x - np.min(x)) / np.ptp(x) for x in q.T]).T

def shannon_k(noise):
  return 1. / (0.5 * np.log(1 + 1./noise))

def shannon_noise(c):
  # Give the noise level compatible with a certain capacity (assuming standard signal)
  return 1. / (np.exp(2. * c) - 1)

def random_frac(k):
  # q = np.random.random(k)
  # return q / np.sum(q)
  return np.random.dirichlet(np.ones(k))

def random_unit(k):
  z = np.random.randn(k)
  return z / np.sqrt(np.dot(z, z))

def gen_data(n_sources=1, k=10, n_samples=10000, noise=0.): # , extra=0):
  """Generate data. The model is that there are n_sources, Z_j, normal with unit variance.
     There are k observed vars per source, X_i = Z_j + E_i. E_i is iid normal noise with variance 'noise' (AWGN). 
     There is a Shannon threshold relating k and noise, defined by shannon_ functions
  """
  sources = np.random.randn(n_samples, n_sources)
  observed = np.vstack([source + np.sqrt(noise) * np.random.randn(n_samples) for source in sources.T for i in range(k)]).T
  # observed = np.hstack([observed, np.random.randn(n_samples, extra)])
  return observed, sources


def gen_data_cap(n_sources=1, k=10, n_samples=1000, capacity=4., scale=False): 
  """Generate data. The model is that there are n_sources, Z_j, normal with unit variance.
     There are k observed vars per source, X_i = Z_j + E_i. E_i is iid normal noise with variance 'noise' (AWGN). 
     Noise is chosen randomly so that the capacity for each source is fixed. 
     There is a Shannon threshold relating k and noise, defined by shannon_ functions
  """
  sources = np.random.randn(n_samples, n_sources)
  capacities = [capacity * random_frac(k) for source in sources.T]
  noises = [shannon_noise(c) for c in capacities]
  observed = np.vstack([source + np.sqrt(noises[j][i]) * np.random.randn(n_samples) for j, source in enumerate(sources.T) for i in range(k)]).T
  if scale:
    scales = 2**(20 * (np.random.random(observed.shape[1]) - 0.5))
    observed = scales * observed  # N by n
  return observed, sources

def gen_data_s(n_sources=1, k=10, n_samples=10000, rho=1): 
  """Generate data. The model is that there are n_sources, Z_j, normal with unit variance.
     There are k observed vars per source, X_i = s_i Z_j + E_i. E_i is iid normal noise with unit variance 'noise' (AWGN). 
  """
  sources = np.random.randn(n_samples, n_sources)
  s = [rho * random_unit(k)  for source in sources.T]
  observed = np.vstack([s[j][i] * source + np.random.randn(n_samples) for j, source in enumerate(sources.T) for i in range(k)]).T
  return observed, sources

def score(true, predicted, method=pearsonr):
    """Compare n true signals to some number of predicted signals.
    For each true signal take the best match in terms of 'method' correlation."""
    rs = []
    for t in true.T:
        rs.append(max(np.abs(method(t, p)[0]) for p in predicted.T))
    return np.mean(rs)

def multi_score(m, C, k, f, n_samples, scale=False):
  results = []
  for i in range(10):
    x, z = gen_data_cap(n_sources=m, k=k, n_samples=n_samples, capacity=C, scale=scale)
    results.append(score(z, f(m, x)))
  return np.mean(results), np.std(results)

def test_solvers(m, C, k, n_samples=1000, scale=False):
  """Use a variety of solvers to recover m sources from data, x."""
  return [(name,) + multi_score(m, C, k, f, n_samples, scale=scale) for name, f in methods]
