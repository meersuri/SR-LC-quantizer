# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:52:00 2020

@author: Meer Suri
"""

import numpy as np
import gensim.downloader as api
import gensim as gs
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sr_lc_integer_rate_quantizer import* 
from utils import*


mpl.rcParams['grid.color'] = '#dddddd'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

wv = api.load('word2vec-google-news-300')

w = gs.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

vectors = w.vectors

x = preProcess(vectors, center = True, unit_norm = True, pca = True, toAngle = True)

n, d = x.shape
num_bins = 100
hist, be = np.histogram(x[:, 0], bins = num_bins, density = True)
plt.figure()
plt.plot(be[:-1], hist, label = 'Data')
mu, loc, scale = stats.invgauss.fit(x[:, 0])
invnorm_dist = stats.invgauss.pdf(be[:-1], mu, loc, scale)
plt.plot(be[:-1], invnorm_dist, label = 'Inverse Gaussian fit')
plt.title("Magnitude histogram - normalized", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc = 1, fontsize = 12, frameon = True)
plt.tight_layout()
##plt.savefig('magnitude_distribution.jpg', dpi = 300, bbox_inches = 'tight')


phase_hist = []
phase_bin_edges = []
phase_fit_hist = []
rnd_idx = np.random.choice(range(1, d), 30)
plt.figure()
plt.title("Phase histogram (normalized)", fontsize = 14)
plt.xlabel("Angle", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
for val in range(1, d):
    hist, be = np.histogram(x[:, val], bins = num_bins, density = True)
    phase_hist.append(hist)
    phase_bin_edges.append(be)
    plt.plot(be[:-1], hist)
    mu, s = stats.norm.fit(x[:, val])
    norm_dist = stats.norm.pdf(be[:-1], mu, s)
    phase_fit_hist.append((mu, s))
    plt.plot(be[:-1], norm_dist)

plt.tight_layout()
##plt.savefig('phase_distribution_rnd.jpg', dpi = 300, bbox_inches = 'tight')


plt.figure()
plt.plot(np.array(phase_fit_hist)[:,0],'-x')
plt.title('Per dimension normal distribution mean', fontsize = 14)
plt.xlabel('Dimension index', fontsize = 14)
plt.ylabel('Mean', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
##plt.savefig('dimension_means.jpg', dpi = 300, bbox_inches = 'tight')


plt.figure()
plt.plot(np.array(phase_fit_hist)[:,1]**2,'-x')
plt.title('Per dimension normal distribution variance', fontsize = 14)
plt.xlabel('Dimension index', fontsize = 14)
plt.ylabel('Variance', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
##plt.savefig('dimension_variances.jpg', dpi = 300, bbox_inches = 'tight')

dim_std = np.array(phase_fit_hist)[:,1]
min_dist_rates = rateAllocation(dim_std)
total_rates = list(min_dist_rates.keys())
plt.figure()
plt.plot(min_dist_rates[total_rates[0]], label = 'Total rate = {} bits'.format(int(total_rates[0])))
plt.plot(min_dist_rates[total_rates[5]], label = 'Total rate = {} bits'.format(int(total_rates[5])))
plt.plot(min_dist_rates[total_rates[50]], label = 'Total rate = {} bits'.format(int(total_rates[50])))
plt.title('Optimal rate allocation', fontsize = 14)
plt.xlabel('Dimension index', fontsize = 14)
plt.ylabel('Rate - bits/sample', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc = 1, fontsize = 12, frameon = True)
plt.tight_layout()
##plt.savefig('rate_allocation.jpg', dpi = 300, bbox_inches = 'tight')
   
lc_coeff = getLCCoeff()
c_scale = np.load("./c_scale").item()


n, d = x.shape
rate = np.array(min_dist_rates[1611], dtype = np.int)
memory = [list(c_scale[r].keys())[-1] for r in rate]

x_comp, sqnr = compress(x[:, 1:], rate, memory, lc_coeff, c_scale)

mu = np.array(phase_fit_hist)[:, 0]
s = np.array(phase_fit_hist)[:, 1]

x_rxn = decompress(x_comp, rate, memory, lc_coeff, mu, s, c_scale)

rxn_error = np.abs((x[:, 1:] - x_rxn)/x[:, 1:])

# check average and max distortion in cosine similarity for a random sample
abs_error = distCosSim(x[:, 1:], x_rxn, 5000)

# visualize quantization and reconstruction
plt.figure()
plt.plot(x[:200, 1])
plt.plot(x_rxn[:200, 0])
    









