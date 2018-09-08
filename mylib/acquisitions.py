"""
Acquisition functions contained here.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

import models

class Acquisition(object):
    
    """
    Base class for acquisition functions.
    
       model: GP model of the discrepancy.
    acq_name: Name of acquisition, e.g. 'EI' or 'LCB'.
     verbose: Verbosity.
         rng: np.RandomState for reproducable results.
    """
    
    def __init__(self, model, acq_name, rel_tol=0.05, verbose=False, rng=None):
        assert isinstance(model, models.GP)
        self.model = model
        self.bounds = model.bounds
        self.input_dim = model.input_dim
        self.acq_name = str(acq_name)
        self.verbose = verbose
        
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        
        # BELOW: for stochastic acquisition ==========================================================
        # for stochastic acquisition (determining relative tolerance for std. devs)
        self.min_acq = np.inf
        self.max_acq = -np.inf
        self.rel_tol = rel_tol    # relative tolerance: rel_tol * (max_obs_acq, min_obs_acq)
        
        # 'absolute' tolerance: make sure the stochastically chosen point isn't more than
        #   0.05 * (theta_upper_bound - theta_lower_bounds) away from the deterministic minimizer
        self.abs_tol = [0.05*(b[1] - b[0]) for b in self.bounds]
        
        self.init_std = np.diag(np.array(self.abs_tol) * 0.30)
    
    def acq(self, theta):
        """
        Must be implemented by subclass. This function is MINIMIZED via 
          scipy.optimize.minimize, so make sure that makes sense for your scenario.
        """
        raise NotImplementedError()
        
    def select_next_theta(self, x0s=None, deterministic=True):
        # default: multi-start optimization with 5 starting points chosen u.a.r. within simulator bounds
        if x0s is None:
            x0s = np.array(
                [np.random.uniform(b[0], b[1], size=(1, 5)) for b in self.bounds]
            ).reshape(-1, self.input_dim)
        elif type(x0s) not in [list, np.ndarray, np.array]:
            x0s = [x0s]
        
        # choose minimum from the 5/however many multistart runs
        min_ = np.inf
        min_x = None
        for x0 in x0s:
            minim = minimize(fun=self.acq, x0=x0, bounds=self.bounds)  # deterministic minimizer of acquisition
            #print(minim)
            if minim.fun < self.min_acq: 
                self.min_acq = minim.fun
            elif minim.fun > self.max_acq: 
                self.max_acq = minim.fun
            
            val = self.acq(minim.x)
            if val < min_:
                min_ = val
                min_x = minim.x
                
        # stochastic acquisition (optional) =======================     
        if not deterministic:
            if not (np.isinf(self.min_acq), np.isinf(self.max_acq)):
                # stochastically sample thetas until you find one that meets the conditions, i.e.
                #   its acquisition value isn't too much higher than the minimum acquisition value, and 
                #   it isn't 'too far away' from the determininistic minizer theta
                cov = self.init_std
                while 1:
                    theta = np.random.multivariate_normal(mean=min_x, cov=cov)
                    if self.acq(theta) <= (min_ + self.rel_tol):  # if acquisition not too high
                        if all([abs(theta[i] - min_x[i]) < self.abs_tol[i] for i in range(self.input_dim)]):
                            min_x = theta
                            break
                    cov *= 0.9
                
        return min_x.reshape(1, self.input_dim)
        
    def plot(self, ax):
        if (self.bounds is None) or (self.input_dim > 1):
            raise NotImplementedError()
        
        thetas = np.arange(self.bounds[0][0], self.bounds[0][1], 0.01).reshape(100, 1)
        acqs = np.zeros((100, 1))
        for i in range(len(thetas)):
            acqs[i] = self.acq(thetas[i])
        
        ax.plot(thetas, acqs)
        #ax.set_title('Acquisition Function (' + self.acq_name + ')')
        #ax.set_xlabel('Theta')
        #ax.set_ylabel('Acquisition (arbitrary units)')
        
        return ax

    
class Expintvar(Acquisition):
    """
    Integrated variance loss function.
    
    https://arxiv.org/pdf/1704.00520.pdf
    """
    
    def __init__(self, model, verbose=False, rng=None):
        super(Expintvar, self).__init__(model, 'expintvar', verbose, rng)
        
    def acq(self, theta):
        raise NotImplementedError()

        
class PostVar(Acquisition):
    
    def __init__(self, model, verbose=False, rng=None):
        super(PostVar, self).__init__(model, 'PostVar', verbose, rng)
    
    def acq(self, theta):
        return -self.model.v(theta)


class LCB(Acquisition):
    """
    Lower confidence bound acquisition function.
    
    See Gutmann and Corander, "Bayesian Optimization for Likelihood-Free 
        Inference of Simulator-Based Statistical Models":
    http://jmlr.org/papers/volume17/15-017/15-017.pdf
        Eq. (45), pg. 20
    
    Args:
        model:  GP model of the discrepancy.
          exp:  Exploration parameter.
    """
    
    def __init__(self, model, verbose=False, rng=None):
        super(LCB, self).__init__(model, 'LCB', verbose, rng)
    
    def acq(self, theta):
        t = self.model.thetas.shape[0]
        d = self.model.input_dim
        # using epsilon = 0.1 as in Gutmann + Corander
        eta = 2 * np.log((t**(.5*d + 2)*np.pi**2) / 0.3)
        
        mu, v = self.model.mu_and_v(theta)
        return mu - np.sqrt(eta * v) 

    
class MPI(Acquisition):
    """
    Maximum probability of improvement acquisition function.
    http://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
    Eq. (42) adapted for minimization.
    
    Args:
        model:  GP model of the discrepancy.
          tau:  Choice for what tau should be, i.e.
                  -lowest discrepancy observed so far ('best) (Default)
                  -highest discrepancy observed so far ('worst)
                  -... TODO TODO
    """
    def __init__(self, model, tau='best', verbose=False, rng=None):
        self.tau = str(tau)
        super(MPI, self).__init__(model, 'MPI', verbose, rng)
    
    def acq(self, theta):
        """
        Minimize probability f(theta) > tau.
        """
        
        if self.tau == 'best':
            tau = self.model.discrs.min()
        elif self.tau == 'min_posterior':
            tau = self.model.mu(self.model.thetas).min()
        else:
            raise NotImplementedError()
        
        mu, v = self.model.mu_and_v(theta)
        # -probability[mu(theta) smaller than incumbent]
        return -norm.cdf((tau - mu) / np.sqrt(v))

    
class EI(Acquisition):
    """
    Expected improvement acquisition function.
    http://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
    Eq. (44) adapted for minimization.
    
    Args:
        model:  GP model of the discrepancy.
          tau:  Choice for what tau should be, i.e.
                  -lowest discrepancy observed so far ('best) (Default)
                  -highest discrepancy observed so far ('worst)
                  -... TODO TODO
    """
    def __init__(self, model, tau='best', verbose=False, rng=None):
        self.tau = str(tau)
        super(EI, self).__init__(model, self.__repr__(), verbose, rng)
    
    def acq(self, theta):
        """
        doc TODO
        """
        mu_theta, v = self.model.mu_and_v(theta)
        sd_theta = np.sqrt(v)
        
        if self.tau == 'best':
            tau = self.model.discrs.min()
        elif self.tau == 'min_posterior':
            tau = self.model.mu(self.model.thetas).min()
        else:
            raise NotImplementedError()
        
        st_norm = (tau - mu_theta) / sd_theta
            
        # probability of improvement
        improve_cdf = norm.cdf(st_norm) 
        lhs = (tau - mu_theta) * improve_cdf
        
        improve_pdf = norm.pdf(st_norm)
        rhs = sd_theta * improve_pdf
            
        return -(lhs + rhs)
    
    def __repr__(self):
        return 'EI(tau={})'.format(self.tau)

    
class Random(Acquisition):
    """
    Picks randomly.
    """
    def __init__(self, model, verbose=False, rng=None):
        super(Random, self).__init__(model, 'Rand', verbose, rng)
        
    def acq(self, theta):
        raise RuntimeError("shouldn't call this")
        
    def select_next_theta(self, x0s=None):
        """
        Select theta randomly from uniform distribution.
        """
        out = np.array(
            [np.random.uniform(b[0], b[1]) for b in self.bounds]
        )
        return out

    
class Bad(Acquisition):
    """
    Purposefully picks bad theta.
    
    Gives theta that it expects to yield high discrepancy, 
      and also penalizes exploration by prioritizing theta with
      low posterior variance under the model:
      
    A(x) = sqrt(eta*var(x)) - mu(x),
    
      where eta is a hyperparameter, var(x) is the posterior variance at point x,
      mu(x) is the posterior mean at point x.
      
    Args:
        model: GP model of the discrepancy.
          exp: 'exploration' parameter.
    """
    def __init__(self, model, exp=1.0, verbose=False, rng=None):
        self.exp = float(exp)
        super(Bad, self).__init__(model, 'Bad', verbose, rng)
        
    def acq(self, theta):
        """
        Pick a bad point.
        """
        return np.sqrt(self.exp * self.model.v(theta)) - self.model.mu(theta)
        
    
        
