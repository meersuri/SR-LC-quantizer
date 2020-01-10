# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:37:47 2020

@author: Meer Suri
"""
import numpy as np
from scipy import stats
from sr_lc_integer_rate_quantizer import* 

def toSpherical(X):
    """ 
    Convert vector to spherical coordinates where first dimension is the
    norm and the remaining d-1 dimensions constitue the phase
    """
    d = X.shape[1]
    X1 = np.zeros_like(X)
    X1[:,0] = np.sqrt(np.sum(X**2, axis = 1))
    for i in range(1, d):
#        X1[:, i] = np.arccos(X[:, i-1]/np.sqrt(np.sum(X[:, i-1: ]**2, axis = 1)))
        X1[:, i] = X[:, i]/X1[:, 0]
    return X1

def optScale(memory, rate, lc_coeff, mu, s, c_lb = 0.5, c_ub = 1.5, n_points = 50):
    """
    Optimize the scale factor of the reproducer values using a simple line search
    from c_lb to c_ub split into n_points
    """
    c = np.linspace(c_lb, c_ub, n_points)
    n_frames = 10
    n = 10000
    all_sqnr = []
    for c_val in c:
        x = np.random.normal(mu, s, n)
        quant = SR_LC_Int_Quantizer(memory, rate, lc_coeff, mu, s, c_scale = c_val)
        quant.encode(x)
        #testing frames
        sqnr = []
        for i in range(n_frames):
            x = np.random.normal(mu, s, n)
            quant.encode(x)
            sqnr.append(quant.sqnr)
        
        print("c_scale = {:.4}".format(c_val))
        print("mean SQNR = {:.4} {}".format(np.mean(sqnr), "dB"))
        CI = stats.t.interval(0.95, len(sqnr)-1, loc=np.mean(sqnr), scale=stats.sem(sqnr))
        print("95% CI size = {:.4} {}".format(CI[1] - CI[0], "dB")) 
        all_sqnr.append(np.mean(sqnr))
        quant.reset()
    print("Best SQNR = {:.4} {}, c_scale = {}".format(np.max(all_sqnr), "dB",c[np.argmax(all_sqnr)] ))
    best_c = c[np.argmax(all_sqnr)]
    
    return best_c

def sqnr(memory, rate, lc_coeff, mu, s, c_scale, frame_len = 1e3, n_frames = 1000):
    """
    calculate and return the SQNR of the quantizer based on n_frames of 
    length frame_len
    """
    n = int(frame_len)
    #start-up frame
    x = np.random.normal(mu, s, n)
    quant = SR_LC_Int_Quantizer(memory, rate, lc_coeff, mu, s, c_scale)
    quant.encode(x)
    #testing frames
    sqnr = []
    for i in range(n_frames):
        x = np.random.normal(mu, s, n)
        quant.encode(x)
        sqnr.append(quant.sqnr)
    
    print("mean SQNR = {:.4} {}".format(np.mean(sqnr), "dB"))
    CI = stats.t.interval(0.95, len(sqnr)-1, loc=np.mean(sqnr), scale=stats.sem(sqnr))
    print("95% CI size = {:.4} {}".format(CI[1] - CI[0], "dB")) 
    quant.reset()
    
    return np.mean(sqnr)
    
def rateAllocation(dim_std, c_min_dist = 1e-3, c_max_dist = 5e-3, n_points = 100):
    """
    Perform optimal rate allocation using the waterfilling algorithm for 
    iid Gaussian sources, using a minimum threshold variance of c_min_dist and 
    max threshold variance of c_max_dist. All dimensions whose variance is less 
    than the set threshold variance are not allocated any bits while the other dimensions
    are pulled down to the threshold variance
    """
    min_dist = c_min_dist*np.min(dim_std)**2
    max_dist = c_max_dist*np.max(dim_std)**2
    dist = np.linspace(min_dist, max_dist, n_points) 
    min_dist_rates = {}
    for d_val in dist:
         r = 0.5*np.log2(dim_std**2/d_val)
         r[r < 0] = 0
         r = np.ceil(r)
         min_dist_rates[np.sum(r)] = r
    
    return min_dist_rates

def compress(x, rate, memory, lc_coeff, c_scale):
    """
    Construct a per dimension quantizer to compress the input sequence x
    """
    n, d = x.shape
    x_comp = np.zeros((n, d), dtype=np.int8)
    mse = []
    sqnr = []
    for i in range(d):
        x_col = x[:, i]
        mu, s = stats.norm.fit(x_col)
        quant = SR_LC_Int_Quantizer(memory[i], rate[i], lc_coeff, mu, s, c_scale[rate[i]][memory[i]])
        encoded = np.array(quant.encode(x_col))
        mse.append(quant.distortion)
        sqnr.append(quant.sqnr)
        x_comp[:, i] = encoded.T 
        
    return x_comp, sqnr

def decompress(x_comp, rate, memory, lc_coeff, mu, s, c_scale):
    """
    Construct a per dimension reconstructor to decompress the quantized input 
    x_comp
    """
    n, d = x_comp.shape
    x_rxn = np.zeros((n, d), dtype=np.float64)
    for i in range(d):
        x_col = x_comp[:, i]
        decoder = SR_LC_Int_Reconstructor(memory[i], rate[i], lc_coeff, mu[i], s[i], c_scale[rate[i]][memory[i]])
        rxn = np.array(decoder.decode(x_col))
        x_rxn[:, i] = rxn.T
    
    return x_rxn

def distCosSim(x, x_rxn, n1 = 100, n2 = 100):
    """
    Randomly sample vectors and calculate the distortion in all-pairs cosine 
    similarity after quantization and reconstruction
    """
    n = x.shape[0]
    true_dist = np.zeros((n1, n2))
    l = np.random.choice(n, n1, replace = False)
    m = np.random.choice(n, n2, replace = False)
    for i in range(n1):
        for j in range(n2):
            true_dist[i, j] = np.dot(x[l[i], :], x[m[j],:])/(np.linalg.norm(x[l[i], :])*np.linalg.norm(x[m[j], :]))
            
    rec_dist = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            rec_dist[i, j] = np.dot(x_rxn[l[i], :], x_rxn[m[j], :])/(np.linalg.norm(x_rxn[l[i], :])*np.linalg.norm(x_rxn[m[j], :]))
    
    rel_error = np.abs((true_dist - rec_dist)/true_dist)
    max_rel_error = np.max(rel_error)
    avg_rel_error = np.mean(rel_error)
    print("Avg rel error = {:.4}".format(avg_rel_error))
    print("Max rel error = {:.4}".format(max_rel_error))
    
def preProcess(vectors, sample_size = 0.01, center = True, unit_norm = False, pca = True, toAngle = False):
    """
    Preprocessing pipeline
    """
    n, d = vectors.shape
    subset = int(n*sample_size)
    vec_idx = np.random.choice(range(n), subset, replace = False)
    
    if center:
        mean_vector = np.mean(vectors, axis = 0)
        x = (vectors - mean_vector)[vec_idx]
    else:
        x = vectors[vec_idx]
        
    if unit_norm:
        norm = np.linalg.norm(x, axis = 1)
        x = x/norm[:, None]
        
    if pca:
        x_cov = np.cov(x, rowvar = False)
        ev, eig = np.linalg.eig(x_cov)
        idx = np.argsort(ev)[::-1]
        ev = np.sort(ev)[::-1]
        eig = eig[:, idx]
        x = np.matmul(x, eig)

    if toAngle:
        x = toSpherical(x)

    return x

def getLCCoeff():
    """
    Label generating LC coefficients for various rates and memory sizes
    """
    
    lc_coeff_r_1 = {2: [(5, 2), (5, 0)],
                    3: [(5, 7), (5, 3)],
                    4: [(5, 4), (5, 12)]}

    lc_coeff_r_2 = {2: [(5, i) for i in range(4)],
                    3: [(5, 2*i) for i in range(8)],
                    4: [(5, 4*i) for i in range(16)]}
    
    lc_coeff_r_3 = {3: [(5, i) for i in range(8)],
                    4: [(5, 2*i) for i in range(16)]}
    
    lc_coeff_r_4 = {4: [(5, i) for i in range(16)],
                    5: [(5, 2*i) for i in range(32)]}
    
    lc_coeff_r_5 = {5: [(5, i) for i in range(32)],
                    6: [(5, 2*i) for i in range(64)]}
    
    lc_coeff_r_6 = {6: [(5, i) for i in range(64)],
                    7: [(5, 2*i) for i in range(128)]}
    
    lc_coeff_r_7 = {7: [(5, i) for i in range(128)],
                    8: [(5, 2*i) for i in range(256)]}
    
    lc_coeff = {1: lc_coeff_r_1, 2: lc_coeff_r_2, 3: lc_coeff_r_3,
                4: lc_coeff_r_4, 5: lc_coeff_r_5, 6: lc_coeff_r_6,
                7: lc_coeff_r_7}
    
    return lc_coeff