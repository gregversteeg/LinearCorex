from __future__ import print_function
import matplotlib.pyplot as plt  
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
import linear_corex as lc
import os
if not os.path.isdir('figs'):
    os.makedirs('figs')


def shannon_noise(c):
  # Give the noise level compatible with a certain capacity (assuming standard signal)
  return 1. / (np.exp(2. * c) - 1)

def random_frac(k):
  # q = np.random.random(k)
  # return q / np.sum(q)
  return np.random.dirichlet(np.ones(k))

def gen_data_cap(n_sources=1, k=10, n_samples=1000, capacity=4.):
  """Generate data. The model is that there are n_sources, Z_j, normal with unit variance.
     There are k observed vars per source, X_i = Z_j + E_i. E_i is iid normal noise with variance 'noise' (AWGN).
     Noise is chosen randomly so that the capacity for each source is fixed.
     There is a Shannon threshold relating k and noise, defined by shannon_ functions
  """
  sources = np.random.randn(n_samples, n_sources)
  capacities = [capacity * random_frac(k) for source in sources.T]
  noises = [shannon_noise(c) for c in capacities]
  observed = np.vstack([source + np.sqrt(noises[j][i]) * np.random.randn(n_samples) for j, source in enumerate(sources.T) for i in range(k)]).T
  return observed, sources

def relative_error(tc_history):
  y = tc_history[1:]
  err = y[-1] - y[:-1]
  return err / y[-1]

seed = 1
errs = []
max_iter = 10**4
tol = 1e-8
N = 500
gpu = False
C = 1
big = False
if big:
    max_iter = 10**5
eliminate_synergy = True

np.random.seed(seed)
x, z = gen_data_cap(n_sources=1, k=10, n_samples=N, capacity=C) # , noise=0.1)
out = lc.Corex(n_hidden=1, verbose=True, max_iter=max_iter, tol=tol, seed=seed, anneal=False, gpu=gpu, eliminate_synergy=eliminate_synergy).fit(x)
errs.append(relative_error(out.history['TC']))
print(np.max(np.abs(out.ws)))
print('TC', out.tc)

np.random.seed(seed)
x, z = gen_data_cap(n_sources=10, k=100, n_samples=N, capacity=C) #, noise=10)
out = lc.Corex(n_hidden=10, verbose=True, max_iter=max_iter, tol=tol, seed=seed, anneal=False, gpu=gpu, eliminate_synergy=eliminate_synergy).fit(x)
errs.append(relative_error(out.history['TC']))
print(np.max(np.abs(out.ws)))
print('TC', out.tc)

np.random.seed(seed)
out = lc.Corex(n_hidden=10, verbose=True, max_iter=max_iter, tol=tol, seed=seed, anneal=False, gpu=gpu, eliminate_synergy=eliminate_synergy).fit(np.random.randn(N,1000))
errs.append(relative_error(out.history['TC']))
print(np.max(np.abs(out.ws)))
print('TC', out.tc)

np.random.seed(seed)
x, z = gen_data_cap(n_sources=50, k=10, n_samples=100, capacity=C) # , noise=100)
out = lc.Corex(n_hidden=50, verbose=True, max_iter=max_iter, tol=tol, seed=seed, anneal=False, gpu=gpu, eliminate_synergy=eliminate_synergy).fit(x)
errs.append(relative_error(out.history['TC']))
print(np.max(np.abs(out.ws)))
print('TC', out.tc)

if big:
    np.random.seed(seed)
    x, z = gen_data_cap(n_sources=100, k=10, n_samples=100, capacity=C) # , noise=100)
    out = lc.Corex(n_hidden=100, verbose=True, max_iter=max_iter, tol=tol, seed=seed,gpu=gpu).fit(x)
    errs.append(relative_error(out.history['TC']))
    print(np.max(np.abs(out.ws)))
    print('TC', out.tc)

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (255, 127, 14),     
             (44, 160, 44), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

fig = plt.figure(figsize=(8,5))
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()    

y0, y1, dy = np.log10(tol) + 2, 0, 1
n = len(out.history['TC']) - 1
#x0, x1, dx = np.log10(1), np.log10(n), 0.5
x0, x1, dx = 1, np.log10(max_iter), 1

plt.ylim(y0, y1)    
plt.xlim(x0, x1)  

# Make sure your axis ticks are large enough to be easily read.    
# You don't want your viewers squinting to read your plot.    
plt.yticks(np.arange(y0+1, y1, dy), ['$10^{%d}$' % x for x in np.arange(y0 + 1, y1, dy)], fontsize=16)
#plt.xticks(np.log10(np.arange(1, n+1, 3)), ['%d'%x for x in np.arange(1, n+1, 3)], fontsize=14)    
plt.xticks(np.arange(x0, x1+1, dx), ['$10^{%d}$' % x for x in np.arange(x0, x1+1, dx)], fontsize=16)
plt.ylabel('Relative Error', fontsize=18, fontweight='bold')
plt.xlabel('# Iterations', fontsize=18, fontweight='bold')

for y in np.arange(y0+1, y1, dy):    
    plt.plot([x0, x1], [y, y], "--", lw=0.5, color="black", alpha=0.3)    

for x in np.arange(x0, x1+1):
    plt.plot([x,x], [y0, y1], "--", lw=0.5, color="black", alpha=0.3)    

# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")  

#plt.plot(np.log10(1 + np.arange(len(err))), np.log10(err), 'o-', lw=2.5, color=tableau20[0]) 
pos = [(55, -5), (152, -4), (205, -3), (600, -2), (600, 0), (700,-1)]
labels = ["m=1, k=10", "m=10, k=100", 'ind (m=10)', "m=50, k=10", 'm=100']
for j, err in enumerate(errs):
  plt.plot(np.log10(np.arange(1, 1 + len(err))), np.log10(np.abs(err)), '-', lw=2.5, color=tableau20[j])
  plt.text(np.log10(pos[j][0]), pos[j][1], labels[j], fontsize=18,fontweight='bold', color=tableau20[j])

plt.savefig("figs/convergence.pdf", bbox_inches="tight")
