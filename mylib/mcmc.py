"""
Markov chain Monte Carlo methods.
"""
import numpy as np
from scipy.stats import norm
from simulators import Simulator
from models import GP
import os

if 'jupyter' in os.environ['_']:
    from ipywidgets import IntProgress
    from IPython.display import display


def sample_mcmc(
    model, h, x0=None, burnin=1000, n_samples=10000, sample_rate=10, g=None,
    noiseless_sample=False, progress_bar=False
):
    """
        Sample points (theta) from either a Gaussian process model or simulator using the
          Metropolis-Hastings algorithm.

        Default proposal density, g, is a Gaussian with diagonal covariance; covariances set to 
          a small value based on the range of possible parameter settings for each dimension.

        Args:
            (models.GP) OR (simulators.Simulator) model:
                GP model of the discrepancy, OR Simulator instance with callable f(), noiseless_f()

               (float)                h:   bandwidth for KDE.
          (np.ndarray)               x0:   initial starting point.
                 (int)           burnin:   number of burn-in samples.
                 (int)      sample_rate:   how many iterations sampling. 
            (callable)                g:   proposal density.
                (bool) noiseless_sample:   whether to call noiseless_f or f (when `model' is a Simulator).
                (bool)     progress_bar:   whether to show progress bar in Jupyter notebook.

        Returns: 
          (np.ndarray)          samples:   with shape (n_samples, input_dim).
    """
    input_dim = model.input_dim
    bounds = model.bounds
    
    # function proportional to predictive distribution
    if isinstance(model, GP):
        f = lambda x: norm.cdf((h - model.mu(x)) / np.sqrt(model.v(x) + model.obs_noise))
    elif isinstance(model, Simulator):
        # std. dev. of obs noise is stored in simulator, so no np.sqrt
        if noiseless_sample:
            f = lambda x: norm.cdf((h - model.noiseless_f(x) / model.obs_noise))
        else:
            f = lambda x: norm.cdf((h - model.f(x) / model.obs_noise))
    else:
        raise ValueError('pass simulator or GP model as first argument.')                    
    
    if x0 is None:
        x0 = np.array([np.random.uniform(b1, b2) for (b1, b2) in bounds]).reshape(1, input_dim)
     
    if g is None:
        cov = []
        for (b1, b2) in bounds:
            cov.append(0.025 * (b2 - b1))
        cov = np.diag(np.array(cov)).reshape(input_dim, input_dim)
        
        g = lambda xt: np.random.multivariate_normal(xt.squeeze(), cov).reshape(1, input_dim)
    
    progress_bar = progress_bar and 'jupyter' in os.environ['_']
    
    # ================================================
    # Burn-in period =================================
    if progress_bar:
        prog = IntProgress(value=0, max=burnin, description='Burn-in')
        display(prog)
        
    x = np.array(x0)
    for i in range(burnin):
        cand = g(x)               # candidate point
        if not model.within_bounds(cand):
            continue
        
        a = f(cand) / f(x)        # acceptance ratio
        if np.random.rand() < a:  # accept/reject
            x = np.copy(cand)
            
        if progress_bar:
            prog.value += 1

    # ================================================
    # Begin sampling =================================
    if progress_bar:
        prog.close()
        prog = IntProgress(value=0, max=n_samples, description='Sampling')
        display(prog)
    
    samples = []
    i = 0
    while len(samples) < n_samples:
        cand = g(x)               # candidate point
        if not model.within_bounds(cand):
            continue
        
        a = f(cand) / f(x)        # acceptance ratio
        if a < 0:
            continue
        
        if np.random.rand() < a:  # accept/reject
            x = np.copy(cand)
    
        if (i % sample_rate) == 0:
            samples.append(np.copy(x))
            if progress_bar:
                prog.value += 1

        i += 1
        
    if progress_bar:
        prog.close()
    
    return np.array(samples).reshape(n_samples, input_dim)