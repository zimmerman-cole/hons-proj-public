"""
Kernel density estimation tools.

Uses uniform kernel.
"""
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle as pkl

def create_grid(sim, n_grid_pts=100, local=False):
    """ 
    Pass simulator 'sim', and number of grid pts per dimension 'n_grid_pts'. 
    If 'local' == True, construct grid(s) only around the global minimum (minima).
    
    Any grid returned will have (n_grid_pts)**input_dim points in it
        (unless 'local' == True and one or more minima is right next to a 
         boundary, in which case the corresponding grid will be 'cut off').
    """
    input_dim = sim.input_dim
    bounds = sim.bounds
    
    if local:
        grids = []
        minima_ = sim.argmin.reshape(-1, sim.input_dim)
        for min_ in minima_:
            dim_ranges = []
            for d, (b1, b2) in enumerate(bounds):
                m = min_[d]
                sp = (b2 - b1) * 0.05
                lb = max(m-sp, b1); ub = min(m+sp, b2)
                dim_ranges.append(np.linspace(lb, ub, n_grid_pts))
            meshes = np.meshgrid(*dim_ranges)
            grid = np.hstack([m.reshape(n_grid_pts**input_dim, 1) for m in meshes])
            grids.append(grid)
        if len(minima_) == 1:
            return grids[0]
        else:
            return grids
    else:
        dim_ranges = []
        for (b1, b2) in bounds:
            dim_ranges.append(np.linspace(b1, b2, n_grid_pts))
        meshes = np.meshgrid(*dim_ranges)
        grid = np.hstack([m.reshape(n_grid_pts**input_dim, 1) for m in meshes])
        return grid

def calculate_true_pdf(sim, h, thetas):
    """
    Returns true normalized probability density function, calculated with 
      uniform kernel with bandwidth h.

    Args:
              sim:   Simulator function (simulators.Simulator)
                h:   Kernel bandwidth.
           thetas:   Array of grid points to evaluate pdf at.
    """
    true_mus = sim.noiseless_f(thetas).reshape(len(thetas), )
    true_pdf = norm.cdf((h - true_mus) / sim.obs_noise)
    # normalize
    true_pdf /= np.sum(true_pdf)

    return true_pdf

def calculate_approx_pdf(model, h, thetas):
    """
    Same as calculate_true_pdf, but using a GP model.
    """
    m_mus, m_vars = model.mu_and_v(thetas)
    #m_mus = model.mu(thetas).reshape(len(thetas), )
    #m_vars = np.array([model.v(t) for t in thetas]).reshape(len(thetas), )
    
    # Note: GP model stores obs noise variance, not sd, so needs to be sqrted here
    normed = (h - m_mus) / np.sqrt(m_vars + model.obs_noise)

    m_pdf = norm.cdf(normed)
    m_pdf /= np.sum(m_pdf)

    return m_pdf

def calculate_data_probs(sim_name, h=None):
    if sim_name == 'BactInf':
        if os.getcwd()[-5] == 'mylib':
            raw_data = pkl.load(open('../BACTERIAL_RESULTS.p', 'rb'))
        else:
            raw_data = pkl.load(open('BACTERIAL_RESULTS.p', 'rb'))
        std = np.sqrt(0.35)
    else:
        raise NotImplementedError()
    
    thetas = np.array([d[0] for d in raw_data])
    discrs = np.array([d[1] for d in raw_data])
    if h is None:
        max_discr = discrs.max(); min_discr = discrs.min()
        print('Discrepancy range: %f' % (max_discr - min_discr))
        h = min_discr + (max_discr - min_discr) * 0.05
    print('Using h=%f' % h)
    
    probs = norm.cdf((h - discrs) / std)
    probs /= la.norm(probs)

    data_w_probs = zip(thetas, discrs, probs)
    
    data_w_probs = sorted(data_w_probs, key=lambda x: x[2], reverse=True)
    
    return data_w_probs, h

def calculate_bact_inf_pdf(sim, data, h):
    true_vals = data[:, -1]
    true_pdf = norm.cdf((h - true_vals) / sim.obs_noise)
    # normalize
    true_pdf /= np.sum(true_pdf)

    return true_pdf



