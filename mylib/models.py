"""
GP models for the discrepancy.
"""
from copy import deepcopy
import warnings

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
import seaborn as sns
from kde import create_grid

#from GPy.models import GPRegression
#from GPy.kern import RBF

import torch

class GP(object):
    """
    Constant-mean, fixed-hyperparameters Gaussian process model for the discrepancy.
    
           thetas: Initial thetas:              (n_evidence x input_dim)
           discrs: Corresponding discrepancies: (n_evidence x 1)
                
           bounds: [(low, high)] for each input variable, e.g.
                    -[(0, 1), (0, None)] for 2 input dimensions
                
        obs_noise: Variance of additive Gaussian noise (sigma squared n)
                
                l: Length scale for each input dimension.
                
       signal_var: Signal variance (independent of theta). The signal variance is the
                    marginal variance of underlying function at a point theta if the
                    observation noise is zero. This model assumes the signal variance
                    is constant across all points.
                   
                
     Mean function:    m(theta) = 0, or mean(observed discrepancies).
    Covar function:    squared exponential covariance function.
    """
    
    def __init__(self, thetas, discrs, obs_noise, bounds=None, l=None, \
                signal_var=5.0, mean='mean', verbose=False, rng=None):
        self.verbose = verbose
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        
        self.discrs = np.array(discrs).reshape(-1, )
        self.t = len(self.discrs) # t: number of pieces of evidence gathered so far
        self.thetas = np.array(thetas).reshape(self.t, -1)
        self.t_init = self.t
        
        self.input_dim = thetas.shape[1]
        
        self.bounds = bounds
        if bounds is not None:
            for d, b in enumerate(bounds):
                assert all(thetas[:, d] >= b[0]) and all(thetas[:, d] <= b[1]), \
                "Input data is inconsistent with provided bounds."
            
        if l is None:
            print('No length scales provided.')
            print('Setting all length scales = 0.5')
                
            self.l = np.ones(self.input_dim) * 0.5
        else:
            self.l = np.array(l)
            
        self.signal_var = float(signal_var)
        
        assert mean in ['mean', 'zero'], "Choose 'mean' or 'zero' for prior mean."
        self._prior = mean
        
        self.K = np.zeros((self.t, self.t))
        for i in range(self.t):
            self.K[i] = self.k(self.thetas[i], self.thetas)
                
        # add observation noise to diagonal
        self.K += obs_noise * np.identity(self.t)
        self.obs_noise = float(obs_noise)
        
        self.K_inv = la.inv(self.K)
    
    def prior_mean(self, thetas):
        """
        Prior mean function. Equals either zero or mean of observed discrepancies.
        """
        thetas, _ = self._handle_array_shape(thetas)
        if self._prior == 'mean':
            return np.mean(self.discrs) * np.ones((len(thetas), 1))
        elif self._prior == 'zero':
            return np.zeros((len(thetas), 1))
        
    def info(self):
        info = """
        t: %d, input_dim: %d, bounds: %s,
        signal_variance: %.3f, obs_noise_var: %.3f,
        length_scales: %s
        """ % (self.t, self.input_dim, self.bounds, 
               self.signal_var, self.obs_noise, 
               self.l)
        return info

    def get_state_at_earlier_iter(self, t):
        """
        Get the state this GP was in at an earlier iteration, t.
        Assumes hyperparameters (observation noise variance, signal variance,
          length scales) haven't changed.
        """
        if t < 0: t = self.t + t + 1 # for python-style neg. # indexing
        if t == self.t: return deepcopy(self)

        out = deepcopy(self)
        out.thetas = out.thetas[:t]
        out.discrs = out.discrs[:t]
        out.K = out.K[:t, :t]
        out.K_inv = la.inv(out.K)
        out.t = t 

        return out
    
    def k(self, t1, t2):
        """
        Covariance function.
        
        Takes two SINGLE thetas t1 and t2.
        
        returns  s_f^2 * exp((1/ls^2)* -euclidean_dist(t1, t2))
        """
        t1, _ = self._handle_array_shape(t1)
        t2, _ = self._handle_array_shape(t2)
            
        exponent = np.sum((1.0 / self.l**2) * (t1 - t2)**2, axis=1)
        return self.signal_var * np.exp(-exponent)
    
    def k_t(self, theta):
        theta, n_thetas = self._handle_array_shape(theta)
        if n_thetas == 1:
            k_t = self.k(theta, self.thetas).reshape(1, self.t)
        else:
            k_t = np.zeros((n_thetas, self.t))
            for i in range(n_thetas):
                k_t[i] = self.k(theta[i], self.thetas)
                
        return k_t
    
    def mu(self, theta):
        """
        Return posterior mean(s) given the current evidence for the given theta(s).
        """
        theta, n_thetas = self._handle_array_shape(theta)
        
        k_t = self.k_t(theta).squeeze()
        
        ft_mt = self.discrs - self.prior_mean(self.thetas).squeeze()
        t2 = np.dot(self.K_inv, ft_mt) # (t, t) (t, )
        ret = np.dot(k_t, t2).squeeze() + self.prior_mean(theta).squeeze()
        return ret
    
    def v(self, theta):
        """
        Posterior variance (without the added sigma squared n) given the current 
        evidence.
        """
        theta, n_thetas = self._handle_array_shape(theta)
        k_t = self.k_t(theta).squeeze()
        lhs = np.dot(k_t, self.K_inv)
        if n_thetas == 1:
            out = np.sum(lhs * k_t)
        else:
            out = np.sum(lhs * k_t, axis=1)
        
        return self.signal_var - out
    
    def mu_and_v(self, theta):
        """ 
        Calculate both posterior mean and variance 
        (avoids duplicate calculations).
        """
        theta, n_thetas = self._handle_array_shape(theta)
        k_t = self.k_t(theta)
        T1 = np.dot(k_t, self.K_inv)
        
        if n_thetas == 1:
            vs = self.signal_var - np.sum(T1 * k_t)
        else:
            vs = self.signal_var - np.sum(T1 * k_t, axis=1)
        
        ft_mt = self.discrs - self.prior_mean(self.thetas).squeeze()
        mus = self.prior_mean(theta).squeeze() + np.dot(T1, ft_mt).squeeze()
        
        return mus, vs
    
    def update(self, theta, discr):
        """
        Update the GP model with new evidence.
        """
        theta, n_thetas = self._handle_array_shape(theta)
        
        self.thetas = np.append(self.thetas, theta, axis=0)
        if n_thetas == 1:
            self.discrs = np.append(self.discrs, discr)
        else:
            self.discrs = np.append(self.discrs, discr, axis=0)
        self.t = len(self.thetas)
        
        if n_thetas == 1:
            new_K = np.zeros((self.t, self.t))
            new_K[:-1, :-1] = self.K
            for i in range(self.t):
                # covar. function is commutative
                val = self.k(self.thetas[i], theta)

                new_K[-1, i] = val
                new_K[i, -1] = val

            # add observation noise to new covariance mat. element
            new_K[-1, -1] += self.obs_noise
            self.K = new_K
            self.K_inv = la.inv(self.K)
        else:
            self._recalculate_covariance_matrix()
        
    def _recalculate_covariance_matrix(self):
        """ For when e.g. hyperparameters are updated. """
        new_K = np.zeros((self.t, self.t))
        for i in range(self.t):
            new_K[i] = self.k(self.thetas[i], self.thetas)
            
        new_K += self.obs_noise * np.identity(self.t)
        self.K = new_K
        self.K_inv = la.inv(new_K)
    
    def plot(self, ax, show_uncertainty=True, point=None, sim=None, acq=None,
             show_evidence=True, show_minimizers=False, use_legend=True, cmap=None):
        """
        Plots the discrepancy model for grid thetas in parameter bounds.
        
        Pass a point (theta, discr) to highlight it.
        Pass your simulator to plot the true discrepancy function.
        """
        if cmap is None:
            cmap = cm.binary
        
        if self.input_dim == 1:
            if show_evidence:
                lbl = u'Evidence $\u03B5^{(t)}$' if use_legend else None
                ax.scatter(self.thetas, self.discrs, marker='o', color='red', label=lbl, zorder=3)
            
            low, high = self.bounds[0]

            # plot model predictions for unobserved (theta, delta_theta)
            thetas = np.arange(low, high, (high - low) / 100.0)
            predic = self.mu(thetas)
            uncertainties = np.sqrt(self.v(thetas))

            clr_gp_m, clr_gp_v = sns.color_palette('Blues')[3], sns.color_palette('Blues')[2]
            lbl = 'Model posterior' if use_legend else None
            ax.plot(thetas, predic, label=lbl, zorder=0, color=clr_gp_m)
            if show_uncertainty:
                ms_plus = predic + uncertainties
                ms_minus = predic - uncertainties
                ax.fill_between(thetas, ms_plus, predic, color=clr_gp_v, alpha=0.2)
                ax.fill_between(thetas, predic, ms_minus, color=clr_gp_v, alpha=0.2)

            # highlight a point if one is provided
            if point is not None:
                ax.scatter(point[0], point[1], s=250, marker='o', color='green', label='New point', zorder=4)

            # plot true (noiseless!) discrep. function if provided
            # (warning: not all simulators have noiseless evaluations available)
            if sim is not None:
                xs = np.arange(low, high, (high-low) / 100.0)
                lbl = u'True E[f(\u03B8)]' if use_legend else None
                ax.plot(xs, sim.noiseless_f(xs), label=lbl, zorder=1, color='orange')
                
            if show_minimizers:
                assert sim is not None, 'Simulator required to show minimizers'
                t_min = self.find_minimizer()
                d_min = self.mu(t_min)
                ax.scatter(t_min, d_min, label=u'Minimizer of GP post. mean \u03BC', color='green')
                
                for minim in sim.argmin:
                    lbl = 'True minimizer(s)' if i == 0 else None
                    clr = 'orange'
                    ax.scatter(minim, sim.noiseless_f(minim), label=lbl, color=clr)

            if acq is None:
                acq = 'Unknown'

            ax.set_title(u'Discrepancy vs. \u03B8 (' + acq + ')')
            ax.set_xlabel(u'\u03B8')
            ax.set_ylabel('Discrepancy')
            ax.set_xlim(self.bounds[0])

            return ax
        
        elif self.input_dim == 2:
            if type(ax) == np.ndarray:
                m_ax, t_ax = ax[0], ax[1]
                do_sim = True
            else:
                m_ax = ax
                do_sim = False
            
            # create grid ====================================================
            low1, high1 = self.bounds[0]
            low2, high2 = self.bounds[1]
            
            t1 = np.linspace(low1, high1, 100)
            t2 = np.linspace(low2, high2, 100)
            T1, T2 = np.meshgrid(t1, t2)
            thetas = np.hstack([T1.reshape(-1, 1), T2.reshape(-1, 1)])
            
            # plot true function values ======================================
            if sim is not None and do_sim:
                true_vals = sim.noiseless_f(thetas)
                vmin, vmax = true_vals.min(), true_vals.max()    
                t_plot = t_ax.contourf(T1, T2, true_vals.reshape(100, 100), 100,
                                      vmin=vmin, vmax=vmax, cmap=cmap)
                t_ax.set_title(u'True E[f(\u03B8)]')
                cbar_true = plt.colorbar(t_plot)
         
            # plot model predictions =========================================
            predics = self.mu(thetas)
            if not do_sim:
                vmin, vmax = predics.min(), predics.max()
            
            m_plot = m_ax.contourf(
                T1, T2, predics.reshape(100, 100), 100, vmin=vmin, vmax=vmax, cmap=cmap
            )
            m_ax.set_title('Posterior mean under GP model')

            # plot observed data =============================================
            if show_evidence:
                m_ax.plot(self.thetas[:, 0], self.thetas[:, 1], 'w.', markersize=10,
                          label='Evidence', color='red')
            
            # plot true function minimizers ==================================
            if sim is not None and show_minimizers:
                t_min = self.find_minimizer()
                m_ax.scatter(t_min[0], t_min[1], label=u'Minimizer of GP post. mean \u03BC', 
                           color='green')
                t_ax.scatter(t_min[0], t_min[1], label=u'Minimizer of GP post. mean \u03BC', 
                           color='green')
                
                for i, minim in enumerate(sim.argmin):
                    lbl = 'True minimizer(s)' if i == 0 else None
                    clr = 'orange'
                    t_ax.scatter(minim[0], minim[1], label=lbl, color=clr)
                    m_ax.scatter(minim[0], minim[1], label=lbl, color=clr)
            
        else:
            raise NotImplementedError('no visualization for above 2D')
    
    def sample_from_posterior(self, theta):
        """
        Sample a discrepancy from posterior at given point theta.
        """
        p_mean = self.mu(theta)
        p_std = np.sqrt(self.v(theta))
        
        return self.rng.normal(loc=p_mean, scale=p_std)
    
    def log_marginal_likelihood(self):
        """
        log p(y|X, theta) = -0.5 y.T Ky^-1 y - 0.5 log |Ky| - (n/2)log(2pi)
        """
        y = self.discrs; K = self.K; n = self.t
        t1 = -0.5 * np.dot(y.T, np.dot(self.K_inv, y))
        t2 = -0.5 * np.log(la.det(K))
        t3 = -(n/2.) * np.log(2. * np.pi)
        
        return t1 + t2 + t3
    
    def _handle_array_shape(self, theta):
        """ Helper function. """
        if self.input_dim == 1:
            if type(theta) != np.ndarray:
                theta = np.array([[theta]])
                n_thetas = 1
            else:
                theta = theta.reshape(len(theta), 1)
                n_thetas = len(theta)
        else:
            if theta.ndim == 1:
                theta = theta.reshape(1, len(theta))
            n_thetas = theta.shape[0]
            assert theta.shape[1] == self.input_dim
            
        return theta, n_thetas
    
    def find_minimizer(self, debug=False):
        """ Find the approximate minimizer of this GP's posterior mean. """
        sorted_idx = np.argsort(self.discrs[::-1])
        step = [0.03*(b[1] - b[0]) for b in self.bounds]
        thetas = np.array(self.thetas).reshape(-1, self.input_dim)
        
        def remove_similar(arr, elem, step):
            arr = arr.reshape(-1, self.input_dim)
            elem = elem.reshape(-1)
            bls = np.zeros(len(arr), dtype=bool)
            for i in range(arr.shape[1]):
                bls = np.logical_or(bls, (abs(arr[:, i] - elem[i]) > step[i]).squeeze()).reshape(-1, )
            return arr[bls, :]

        # use 8 'unique' starting points for minimization (if 8 'unique' exist)
        start_pts = [thetas[0]]
        for _ in range(8):
            thetas = remove_similar(thetas[1:, :], thetas[0], step)
            if len(thetas) == 0:
                break
            start_pts.append(thetas[0])
        
        if debug:
            print('start_pts', start_pts)
            
        min_val = np.inf
        min_theta = None
        
        for theta in start_pts:
            result = minimize(fun=self.mu, x0=theta, bounds=self.bounds)
            if result.fun < min_val:
                min_val = result.fun
                min_theta = result.x
                
        return min_theta

    @property
    def observed_data(self):
        return np.hstack([self.thetas, self.discrs.reshape(-1, 1)])
    
    def within_bounds(self, theta):
        theta = theta.reshape(self.input_dim, )
        
        for ti, (bimn, bimx) in zip(theta, self.bounds):
            if (ti < bimn) or (ti > bimx):
                return False
            
        return True
    
class AdaptiveGP(GP):
    """
    Gaussian Process model for the discrepancy. This one continuously adapts its 
      hyperparameters as it gathers data. It uses a constant mean function which 
      equals the mean observed discrepancy.
    
           thetas: Initial thetas:              (n_evidence x input_dim)
           discrs: Corresponding discrepancies: (n_evidence x 1)
                
           bounds: [(low, high)] for each input variable, e.g.
                    -[(0, 1), (0, None)] for 2 input dimensions
                    
    Sets hyperparameters by maximizing log leave-one-out predictive probability.
      (see Rasmussen and Williams 5.4.2)
    """
    
    def __init__(self, thetas, discrs, bounds=None, verbose=False, rng=None):
        # Initialize basic stuff
        self.verbose = verbose
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        
        if len(thetas.shape) == 1:
            thetas = thetas.reshape(len(thetas), 1)
        
        self.t = thetas.shape[0]
        self.input_dim = thetas.shape[1]
        self.bounds = bounds
        self.thetas = thetas
        self.discrs = discrs
        
        # Set hyperparameters to arbitrary initial values
        self.a = np.zeros(self.input_dim)
        self.b = np.zeros(self.input_dim)
        self.c = np.mean(self.discrs) 
        self.l = np.ones(self.input_dim) * .5
        self.obs_noise = np.mean((self.discrs - self.c)**2)
        self.signal_var = (self.discrs.max() - self.discrs.min())
        
        self.K = np.zeros((self.t, self.t))
        for i in range(self.t):
            self.K[i] = self.k(self.thetas[i], self.thetas)
                
        # add observation noise to diagonal
        self.K += self.obs_noise * np.identity(self.t)
        self.K_inv = la.inv(self.K)
        
        # FIT HYPERPARAMETERS ==================================================
        # Fit prior mean
        self.optimize_prior_mean()
        self.optimize_hyperparameters()
        
    def prior_mean(self, theta):
        return np.sum(self.a * theta + self.b * theta, axis=1) + self.c
    
    def optimize_prior_mean(self):
        # minimize an objective function f equal to sum of individual 
        # log square errors 
        def objective(params, X, y, input_dim, post_mean):
            #X, y, input_dim = args
            a = np.array(params[0:input_dim])
            b = np.array(params[input_dim:2*input_dim])
            c = params[-1]
            
            pm = np.sum((a * X**2) + (b * X) + c, axis=1)
            pm += post_mean(X)
            log_sq_errs = np.log(np.sum((y - pm)**2, axis=1))
            return np.sum(log_sq_errs) + la.norm(params[:-1])
        
        x0 = np.append(np.append(self.a, self.b), self.c)
        
        bds = [(0., None)]*self.input_dim
        bds.extend([(None, None)]*(self.input_dim+1))
        prms = minimize(objective, x0, args=(self.thetas, self.discrs, 
                                             self.input_dim, self.mu),
                        bounds=bds).x
        
        self.a = prms[0:self.input_dim]
        self.b = prms[self.input_dim:2*self.input_dim]
        self.c = prms[-1]
        
        #self.a = np.zeros_like(self.a)
        #self.b = np.zeros_like(self.b)
        #self.c = 0.
        
    def optimize_hyperparameters(self):
        """
        Use GPy to optimize Gaussian process hyperparameters.
        """
        #kern = RBF(self.input_dim, variance=self.signal_var, lengthscale=self.l)
        #gpr = GPRegression(self.thetas, self.discrs, kernel=kern, noise_var=self.obs_noise)
        #kern, lik = gpr.parameters
        #gpr.optimize()
        
        #s_var, ls = kern.parameters
        #obs_noise = lik.parameters[0]
        #print(s_var)
        #print(ls)
        #print(obs_noise)
        
        #self.signal_var = s_var.T[0]
        #self.obs_noise = obs_noise.T[0]
        #self.l = np.array(ls.T[0]).reshape(self.input_dim, )
        #self.l = np.array([.1])
        
        self.optimize_prior_mean()

    def get_state_at_earlier_iter(self, t):
        """
        Get the state this GP was in at an earlier iteration, t.
        Assumes hyperparameters (observation noise variance, signal variance,
          length scales) haven't changed.
        """
        assert t > 0
        if t == self.t: return deepcopy(self)

        out = deepcopy(self)
        out.thetas = out.thetas[:t]
        out.discrs = out.discrs[:t]
        out.K = out.K[:t, :t]
        if torch.cuda.is_available():
            cpy = torch.inverse(torch.from_numpy(out.K).cuda())
            out.K_inv = cpy.cpu().numpy()
        else:
            out.K_inv = la.inv(out.K)
        out.t = t
        out.optimize_hyperparameters()

        return out
    
    def mu(self, theta):
        """
        Return posterior mean(s) given the current evidence for the given theta(s).
        """
        theta, n_thetas = self._handle_array_shape(theta)
        mt = self.prior_mean(self.thetas).reshape(self.t, 1)
        ft = self.discrs.reshape(self.t, 1)
        
        if n_thetas == 1:
            k_t = self.k(theta, self.thetas)

            # Simplified version of (43) from Gutmann + Corander
            # kt(theta).T Kt^-1 (ft - mt)
            t2 = np.dot(self.K_inv, ft - mt)
            ret = np.dot(k_t.reshape(1, self.t), t2)[0][0]

            return ret + self.prior_mean(theta)[0]
        else:
            # TODO: get rid of this for loop
            k_t = np.zeros((n_thetas, self.t))
            for i in range(n_thetas):
                k_t[i] = self.k(theta[i], self.thetas)
                
            t2 = np.dot(self.K_inv, ft - mt)
            ret = np.dot(k_t, t2).reshape(self.t, )
                
            return ret + self.prior_mean(theta)
    
    def update(self, theta, discr):
        """
        Update the GP model with new piece(s) of evidence.
        """
        theta, _ = self._handle_array_shape(theta)
        
        self.thetas = np.append(self.thetas, theta, axis=0)
        self.discrs = np.append(self.discrs, discr, axis=0)
        self.t = len(self.thetas)
        
        new_K = np.zeros((self.t, self.t))
        new_K[:-1, :-1] = self.K
        
        for i in range(self.t):
            # covar. function is commutative
            val = self.k(self.thetas[i], theta)
            
            new_K[-1, i] = val
            new_K[i, -1] = val
            
        # add observation noise to new covariance mat. element
        new_K[-1, -1] += self.obs_noise
        
        self.K = new_K
        self.K_inv = la.inv(self.K)
        
        self.optimize_hyperparameters()
