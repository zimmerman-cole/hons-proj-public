"""
For calculating TV distance/KL divergence between two probability distributions.
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle as pkl

class DistanceCalculator(object):  
    """
===================================================================================
    Approximates probability distributions using kernel density estimation
    (with a uniform kernel where c = 1.0 and h is provided).
    
    p(theta | y0) ~= cdf(h - mu(theta) / sqrt(obs_noise)),
        
    where cdf is a standard normal CDF and mu(theta) is the GP model's posterior 
      prediction for the discrepancy at point theta.
===================================================================================
    Calculates distances using Marko's Matlab code.
=================================================================================== 
    Args:
        input_dim:   Simulator input dimension.
           bounds:   Bounds within which distances are calculated in a grid.
        obs_noise:   Observation noise.
              sim:   Simulator function.
                h:   Kernel bandwith for kernel density estimation.
          session:   pymatlab Matlab session for calling Marko's code.
             acqs:   Acquisition.Acquistion instances to track distances for.
===================================================================================
    """
    
    def __init__(self, input_dim, bounds, obs_noise, sim, h, session, acqs):
        self.input_dim = int(input_dim)
        self.h = float(h)
        self.obs_noise = float(obs_noise)
        self.sd_noise = np.sqrt(self.obs_noise)
        self.bounds = bounds
        
        # matlab session
        self.session = session
        # ========================================================
        # For 1D and 2D simulators, calculate distance in a grid 
        # ========================================================
        if input_dim in [1, 2]:
            # Construct xgrid
            if input_dim == 1:
                low, high = bounds[0][0], bounds[0][1]
                self.thetas = np.arange(low, high, (high-low) / 100.0).reshape(100, )
            elif input_dim == 2:
                t1 = np.linspace(bounds[0][0], bounds[0][1], 100)
                t2 = np.linspace(bounds[1][0], bounds[1][1], 100)
                T1, T2 = np.meshgrid(t1, t2)
                self.thetas = np.hstack([T1.reshape(100*100, 1), \
                                         T2.reshape(100*100, 1)])

            # Warning: not all simulators guaranteed to have noiseless evaluations
            # (i.e. bacterial infections simulator). Will return noisy evaluations 
            # if so.
            true_mus = sim.noiseless_f(self.thetas).reshape(len(self.thetas), )

            self.true_pdf = norm.cdf((h - true_mus) / self.sd_noise)

            # normalize
            self.true_pdf = self.true_pdf / la.norm(self.true_pdf)
            
            # track distances for each acquisition at each iteration
            self.dists = {a.acq_name: [] for a in acqs}
            
            # put xgrid and true_pdf in Matlab
            self.session.putvalue('xgrid', self.thetas)
            self.session.putvalue('true_pdf', self.true_pdf)
            
            # Store model pdfs for each acquisition
            self.m_pdfs = {a.acq_name: [] for a in acqs}
            
        # ============================================================
        # For 3D+ simulators, get marginal distributions and calculate 
        # TV distance dimension-wise.
        # ============================================================
        else:
            # dim > 2
            raise NotImplementedError()
            
    def approximate_pdf(self, gp_model):
        """
        Approximate a GP model's posterior distribution using KDE.
        """
        if self.input_dim in [1, 2]:
            m_mus = gp_model.mu(self.thetas).reshape(len(self.thetas), )
            
            m_vars = np.array([gp_model.v(t) for t in self.thetas]).reshape(len(self.thetas), ) # TODO: make v() take array of vals
            
            # estimate distribution using KDE w/ uniform kernel
            normed = (self.h - m_mus) / np.sqrt(m_vars + self.obs_noise)

            m_pdf = np.array([norm.cdf(n_t) for n_t in normed])
            # normalize
            m_pdf /= la.norm(m_pdf)
            
            return m_pdf
        else:
            raise NotImplementedError('TODO: dist-calcs for 3D+')
        
    def tv_distance(self, gp_model, acq_name, call_matlab=True):
        # =========================================================================
        # For 1D and 2D simulators, calculate distance in a grid 
        # =========================================================================
        if self.input_dim in [1, 2]:
            model_pdf = self.approximate_pdf(gp_model)
            self.m_pdfs[acq_name].append(model_pdf)

            # =======================================================
            # To minimize pymatlab overhead, only call Marko's distance-calculation
            # code when call_matlab=True (e.g. could do call_matlab=True every 
            # 20 iterations). If call_matlab=False, store the model 
            # pdf and calculate a batch of distances later.
            if call_matlab:
                # calculate distances using Marko's Matlab code
                self.session.putvalue('est_pdf', model_pdf)
                self.session.run('tv = total_variation_distance(xgrid, true_pdf, est_pdf)')
                self.dists[acq_name].append(self.session.getvalue('tv'))
            else:
                # TODO
                raise NotImplementedError()
            
        # ========================================================================
        # For 3D+ simulators, get marginal distributions and calculate TV distance 
        # dimension-wise.
        # ========================================================================
        else:
            # dim > 2, TODO
            raise NotImplementedError()
    
    def show_last_posteriors(self, acq, theta=None):
        """
        Show most recent approximated posteriors vs true posterior 
          for given acquisition.
        
        theta: pass a point to show it on the plot (i.e. point that was 
                  just chosen by the acquisition).
        """
        if self.input_dim == 1:
            x = np.arange(self.bounds[0][0], self.bounds[0][1], 0.01)
            dist = self.dists[acq][-1]
                
            f, ax = plt.subplots(figsize=(10, 3))
            title = """
            %s approx, true posteriors (dist=%.3f)
            """ % (acq, dist)
                
            ax.set_title(title)
                
            app_pdf = self.m_pdfs[acq][-1]
            ax.plot(x, app_pdf, label='Approx.')
            ax.plot(x, self.true_pdf, label='True')
            if theta is not None:
                ax.scatter(theta, self.true_pdf[int(theta)], color='green', \
                           s=150, label='Chosen')
                
            plt.legend(); plt.show(); raw_input()
        elif self.input_dim == 2:
            raise NotImplementedError()
        else:
            raise NotImplementedError("Can't do for 3D+")
        
    def plot_distances(self, ax):
        """
        Plot TV distance vs. iteration for each acquisition function.
        """
        for i, (name, dists) in enumerate(self.dists.items()):
            n_iter = len(dists)
            x = range(n_iter)
            ax.plot(x, dists, label=name)
            
            print('Total TV dist for {} over {} iter: {}'.format(name, n_iter, \
                                                                 np.sum(dists)))
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('TV distance')
        return ax
    
    def reset(self):
        """
        Clear distances and pdfs.
        """
        keys = self.dists.keys()
        for k in keys:
            self.dists[k] = []
            self.pdfs[k] = []