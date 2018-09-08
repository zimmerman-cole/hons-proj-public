"""
Portfolios of acquisition functions.

Hedge algorithm
Exp3 algorithm
'Explorer'
Baseline (picks randomly)

# NOT USING ==
NormalHedge
HedgeNorm
HedgeAnnealed
Entropy Search Portfolio (partially implemented; not using)
"""

from copy import deepcopy
from collections import OrderedDict

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar

import acquisitions
import models
from kde import create_grid
from mcmc import sample_mcmc

class Portfolio(object):
    
    def __init__(self, model, acq_name, acqs, verbose=False, rng=None):
        assert isinstance(model, models.GP)
        self.model = model
        self.bounds = model.bounds
        self.input_dim = model.input_dim
        self.acq_name = str(acq_name)
        self.verbose = verbose
        
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        
        assert all([isinstance(a, acquisitions.Acquisition) for a in acqs])
        self.acqs = acqs
        self.acq_names = [a.acq_name for a in acqs]
        self.K = len(acqs)
        self.choices = []
        
    def plot(self, *args, **kwargs):
        pass
    
    def __deepcopy__(self, memo):
        """
        Special __deepcopy__ override to make sure Portfolio's base acquisitions are
        using the same model as Portfolio.
        """
        model = deepcopy(self.model)
        base_acqs = [deepcopy(a) for a in self.acqs]
        # make sure base acquisitions using same model as copy
        for a in base_acqs:
            a.model = model
            
        return self.__class__(
            model=model, acqs=base_acqs, sim=self.sim, rng=self.rng, verbose=self.verbose
        )
    
    def new_model(self, model):
        """ Return a copy with a new model. """
        base_acqs = [deepcopy(a) for a in self.acqs]
        for a in base_acqs:
            a.model = model
        
        if self.__class__ is Explorer:
            return self.__class__(
                model, base_acqs, self.sim, h_mult=self.h_mult, rng=self.rng
            )
        
        return self.__class__(model, base_acqs, self.sim, rng=self.rng)
   
  
class GoldStandard(Portfolio):
    """
    This portfolio is meant to be the gold standard portfolio.
    It 'cheats' by evaluating every acquisition's nominee, then choosing the best point 
        (as measured by reduction in TV distance).
    """
    
    def __init__(self, model, acqs, sim, verbose=False, rng=None):
        super(GoldStandard, self).__init__(model, 'GoldStandard', acqs, verbose, rng)
        
        self.sim = sim
        if self.sim.input_dim > 2:
            raise NotImplementedError('not implemented for 3D+ sims')
        
        self.choices = []
        self.reducs = {a_n: [] for a_n in self.acq_names}
        
        self.thetas = create_grid(self.sim)
        discrs = self.sim.noiseless_f(self.thetas)
        self.h = discrs.min() + 0.035 * (discrs.max() - discrs.min())
        self.pdf_true = norm.cdf((h - discrs) / self.sim.obs_noise)
        self.pdf_true /= np.sum(self.pdf_true)
        
    def select_next_theta(x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
            
        # reductions in TV distance
        redux = []
        for acq_name, theta in candidates.items():
            ds_true = self.sim.noiseless_f(theta)
            
            m_cp = deepcopy(self.model)
            m_cp.update(theta, ds_true)
            
            mus, vs = m_cp.mu(self.thetas), m_cp.v(self.thetas)
            
            pdf_appr = norm.cdf((self.h - mus) / np.sqrt(vs + m_cp.obs_noise))
            pdf_appr /= np.sum(pdf_appr)
            
            tv = 0.5 * np.sum(np.abs(self.pdf_true - pdf_appr))
            
            self.reducs[acq_name].append(tv)
            redux.append(tv)
            
        max_reduc = np.argmax(redux)
        best_acq = candidates.keys()[max_reduc]
        self.choices.append(best_acq)
        next_theta = candidates[best_acq]
        
        next_discr = self.sim.f(next_theta)
        self.model.update(next_theta, next_discr)
        
        return next_theta
    

class LCBEI(Portfolio):
    
    def __init__(self, model, acqs, sim, n_samples=100, verbose=False, rng=None):
        super(LCBEI, self).__init__(model, 'LCBEI', acqs, verbose, rng)
        self.n_samples = int(n_samples)
        self.i_num = 0
        self.sim = sim
        
        self.LCB = None
        self.EI = None
        for acq in self.acqs:
            if 'EI' in acq.acq_name:
                self.EI = acq
            elif acq.acq_name == 'LCB':
                self.LCB = acq
         
        if self.LCB is None:
            raise ValueError('Please pass LCB acquisition to LCBEI portfolio.')
        if self.EI is None:
            raise ValueError('Please pass EI acquisition to LCBEI portfolio.')
            
        self.probs = {'EI': [], 'LCB': []}
        
    def select_next_theta(self, x0s=None):
        mx_obs = self.model.discrs.max()
        accept = lambda pm: max(min((mx_obs - pm) / mx_obs, 0.), 1.)
        
        samples = []
        while len(samples) < self.n_samples:
            theta = np.array(
                [np.random.uniform(b1, b2) for (b1, b2) in self.bounds]
            )
            
            mu_t = self.model.mu(theta)
            
            if np.random.rand() < accept(mu_t) or np.random.rand() < 0.25:
                samples.append(theta)
                
        samples = np.array(samples).reshape(self.n_samples, self.input_dim)
        tot_pv = np.sum(self.model.v(samples))
        
        p_LCB = tot_pv / float(self.n_samples * self.model.signal_var)
        self.probs['LCB'].append(p_LCB)
        self.probs['EI'].append(1. - p_LCB)
        
        if np.random.randn() < p_LCB:
            next_theta = self.LCB.select_next_theta(x0s)
            self.choices.append('LCB')
        else:
            next_theta = self.EI.select_next_theta(x0s)
            self.choices.append('EI')
        
        next_discr = self.sim.f(next_theta)
        self.model.update(next_theta, next_discr)
        
        return next_theta
        

class Baseline(Portfolio):
    """ Chooses an acquisition randomly. """
    
    def __init__(self, model, acqs, sim, verbose=False, rng=None):
        super(Baseline, self).__init__(model, 'Baseline', acqs, verbose, rng)
        self.sim = sim
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = 'not selected'
        
        idx = np.random.randint(self.K)
        chosen = candidates.items()[idx]
        candidates[chosen[0]] = self.acqs[idx].select_next_theta(x0s)
        chosen = (chosen[0], candidates[chosen[0]])
        self.choices.append(chosen[0])
        
        next_discr = self.sim.f(chosen[1])
        self.model.update(chosen[1], next_discr)
        
        return chosen[1]
    
    
class AdaptiveLearningRate(Portfolio):
    """ todo: doc """
    
    def __init__(self, portfolio_class, model, acqs, sim, batch_size=5, verbose=False, 
                 eta=1.0, rng=None):
        assert_msg = "Please provide two portfolios (with learning rates) of the same class."
        assert len(acqs) == 2, assert_msg
        assert type(acqs[0]) == type(acqs[1]), assert_msg
        assert hasattr(acqs[0], 'eta'), assert_msg
        
        name = 'Adap_' + self.acqs[0].acq_name
        super(AdaptiveLearningRate, self).__init__(model, name, acqs, verbose, rng)
        
        self.acqs[0].acq_name += '1'
        self.acqs[1].acq_name += '2'
        
        self.new_data = np.zeros((0, self.input_dim + 1))
        
        self.batch_size = int(batch_size)
        self.past_etas = [eta]
        self.eta = eta
        self.iter = 0
        self.acq_names = [a.acq_name for a in self.acqs]
        
        # typical portfolio stuff here
        self._past_probs = {a.acq_name: [] for a in self.acqs}
        self.gains = np.zeros(self.K)
        self.choices = []
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
            
        pass

    
class Explorer(Portfolio):
    """
    Based off Hedge and Exp3.
    """
    
    def __init__(self, model, acqs, sim, h_mult=0.05, verbose=False, eta=1.0, 
                 forget_factor=1.0, h_dcr=0., rng=None):
        super(Explorer, self).__init__(model, 'Explorer', acqs, verbose, rng)
        self.sim = sim
        self.eta = float(eta)
        self.probs = {a.acq_name: [] for a in self.acqs}
        self.gains = np.zeros(self.K)
        self.iter = 0
        self.h_mult = float(h_mult)
        self.forget_factor = float(forget_factor)
        self.h_dcr = float(h_dcr)
        
        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = 'not selected'
            
        while True:
            exp_g = np.exp(self.eta * self.gains)
            probs = exp_g / np.sum(exp_g)

            # Mix with uniform distribution; weight unif. less with more iterations
            m_iter = self.K * 5. # n_iters after which no more uniform
            unif = (1. / self.K) * (max(m_iter - self.iter, 0) / m_iter)
            probs = (probs + unif) / 2.
            probs = probs / np.sum(probs)

            for acq, p in zip(self.acqs, probs):
                self.probs[acq.acq_name].append(p)

            idx = np.random.choice(range(self.K), p=probs)
            chosen = candidates.items()[idx]
            
            if chosen is None:
                # in case of overflow
                self.gains *= 0.05     # <- this will change probs, but is necessary
            else:
                candidates[chosen[0]] = self.acqs[idx].select_next_theta(x0s)
                chosen = (chosen[0], candidates[chosen[0]])
                break
        
        self.iter += 1
     
        rewards = self._calc_rewards(chosen, candidates)
        self.gains = self.forget_factor * self.gains + rewards
        
        # TRACKING INFORMATION ===============================
        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(self.past_gains[-1] - self.past_gains[-2])
        self.choices.append(chosen[0])
        
        return chosen[1]
    
    def _calc_rewards(self, chosen, candidates):
        # Calculate old model predictions for chosen point.
        old_mu, old_var = self.model.mu(chosen[1]), self.model.v(chosen[1])
        
        # Evaluate objective at chosen candidate, then
        # update model with new (theta, discr).
        new_discr = self.sim.f(chosen[1])
        self.model.update(chosen[1], new_discr)
        
        # Calculate new model predictions for chosen point.
        new_mu, new_var = self.model.mu(chosen[1]), self.model.v(chosen[1])
        
        # Estimate suitable kernel bandwidth h.
        min_discr = self.model.discrs.min(); max_discr = self.model.discrs.max()
        rg = max_discr - min_discr
        #h1 = min_discr + 0.05 * rg
        #h2 = min_discr + 0.25 * rg
        #h3 = min_discr + 0.50 * rg
        h = max(0.05, self.h_mult - (float(self.iter) * self.h_dcr))
        h = min_discr + h * rg
        
        # For each potential bandwidth setting:
        # calculate likelihoods of chosen point before and after updating model.
        old_lik = norm.cdf((h - old_mu) / np.sqrt(old_var))
        new_lik = norm.cdf((h - new_mu) / np.sqrt(new_var))
        reward = abs(new_lik - old_lik) 
        
        rewards = np.zeros(self.K)
        for i, acq_name in enumerate(candidates.keys()):
            if acq_name == chosen[0]:
                rewards[i] = reward
            else:
                rewards[i] = 0.
                
        return rewards
    

class ExplorerFI(Portfolio):
    
    def __init__(self, model, acqs, sim, h_mult=0.05, verbose=False, eta=1.0, rng=None):
        super(ExplorerFI, self).__init__(model, 'ExplorerFI', acqs, verbose, rng)
        self.sim = sim
        self.eta = float(eta)
        self.gains = np.zeros(self.K)
        self.iter = 0
        self.h_mult = float(h_mult)
        
        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        
        if self.input_dim <= 2:
            self.use_grid = True
            self.thetas = create_grid(self.sim)
        else:
            self.use_grid = False
            self.thetas = []
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
        
        # calculate pre-update pdf ===================
        min_d, max_d = self.model.discrs.min(), self.model.discrs.max()
        h = min_d + self.h_mult * (max_d - min_d)
        if self.use_grid or (self.iter & (self.iter - 1)) != 0:
            thetas = self.thetas
        else: 
            self.thetas = np.zeros((0, self.input_dim))
            for _ in range(100):
                samples = sample_mcmc(self.model, h, burnin=20, n_samples=10)
                self.thetas = np.vstack([self.thetas, samples])
            thetas = self.thetas
        
        mus, vs = self.model.mu(thetas), self.model.v(thetas)
        
        pdf_bef = norm.cdf((h - mus) / np.sqrt(vs + self.model.obs_noise))
        pdf_bef /= np.sum(pdf_bef)
        
        # calculate pdfs after updating, rewards =====
        rewards = np.zeros(self.K)
        max_rwd, chosen = -1., None
        for k, acq_name in enumerate(self.acq_names):
            chosen_theta = candidates[acq_name]
            
            discr = self.model.sample_from_posterior(chosen_theta)
                
            m_cp = deepcopy(self.model)
            m_cp.update(chosen_theta, discr)
            
            # calculate pdf
            mus, vs = m_cp.mu(thetas), m_cp.v(thetas)
            pdf_i = norm.cdf((h - mus) / np.sqrt(vs + self.model.obs_noise))
            pdf_i /= np.sum(pdf_i)
            
            rwd = 0.5 * np.sum(np.abs(pdf_bef - pdf_i))
            rewards[k] = rwd
            
            if rwd >= max_rwd:
                max_rwd = rwd
                chosen = acq_name  
                
        # Select candidate, sample objective, update
        next_theta = candidates[chosen]
        next_discr = self.sim.f(next_theta)
        self.model.update(next_theta, next_discr)
        
        self.gains += rewards
        
        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(self.past_gains[-1] - self.past_gains[-2])
        self.choices.append(chosen)
        
        return next_theta

    
class ExplorerFIPR(Portfolio):
    
    def __init__(self, model, acqs, sim, h_mult=0.05, verbose=False, eta=1.0, rng=None):
        super(ExplorerFIPR, self).__init__(model, 'ExplorerFIPR', acqs, verbose, rng)
        self.sim = sim
        self.eta = float(eta)
        self.probs = {name: [] for name in self.acq_names}
        self.gains = np.zeros(self.K)
        self.iter = 0
        self.h_mult = float(h_mult)
        
        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        
        if self.input_dim <= 2:
            self.use_grid = True
            self.thetas = create_grid(self.sim)
        else:
            self.use_grid = False
            self.thetas = []
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
        
        chosen = None
        # choose candidate ===========================
        while chosen is None:
            exp_g = np.exp(self.eta * self.gains)
            probs = exp_g / np.sum(exp_g)

            for acq, p in zip(self.acqs, probs):
                self.probs[acq.acq_name].append(p)

            idx = np.random.choice(range(self.K), p=probs)
            chosen = candidates.items()[idx]
            
            if chosen is None:
                # in case of overflow
                self.gains *= 0.05
                    
        # calculate pre-update pdf ===================
        if 1:
            min_d, max_d = self.model.discrs.min(), self.model.discrs.max()
            h = min_d + self.h_mult * (max_d - min_d)
            if self.use_grid or (self.iter & (self.iter - 1)) != 0:
                thetas = self.thetas
            else: 
                self.thetas = np.zeros((0, self.input_dim))
                for _ in range(100):
                    samples = sample_mcmc(self.model, h, burnin=20, n_samples=10)
                    self.thetas = np.vstack([self.thetas, samples])
                thetas = self.thetas

            mus, vs = self.model.mu(thetas), self.model.v(thetas)

            pdf_bef = norm.cdf((h - mus) / np.sqrt(vs + self.model.obs_noise))
            pdf_bef /= np.sum(pdf_bef)
        
        # calculate pdfs after updating, rewards =====
        if 1:
            rewards = np.zeros(self.K)
            for k, acq_name in enumerate(self.acq_names):
                chosen_theta = candidates[acq_name]

                discr = self.model.sample_from_posterior(chosen_theta)

                m_cp = deepcopy(self)
                m_cp.model.update(chosen_theta, discr)

                # calculate pdf
                mus, vs = m_cp.model.mu(thetas), m_cp.model.v(thetas)
                pdf_i = norm.cdf((h - mus) / np.sqrt(vs + self.model.obs_noise))
                pdf_i /= np.sum(pdf_i)

                rewards[k] = 0.5 * np.sum(np.abs(pdf_bef - pdf_i))

        # Select candidate, sample objective, update
        next_theta = candidates.values()[idx]
        next_discr = self.sim.f(next_theta)
        self.model.update(next_theta, next_discr)
        
        self.gains += rewards
        
        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(self.past_gains[-1] - self.past_gains[-2])
        self.choices.append(chosen)
        
        return next_theta
    
    
class Exp3(Portfolio):
    """
    https://arxiv.org/pdf/0912.3995.pdf
    
    Exp3 portfolio.
    """
    
    def __init__(self, model, acqs, sim, verbose=False, eta=1.0, rng=None):
        super(Exp3, self).__init__(model, 'Exp3', acqs, verbose, rng)
        self.sim = sim
        self.eta = float(eta)
        
        self.probs = {a.acq_name: [] for a in self.acqs}
        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        
        self.gains = np.zeros(self.K)
        
        self.iter = 0
        
    def select_next_theta(self, x0s=None):
        """
        Select next point to evaluate objective at based on Exp3 algorithm.
        
        x0s: list of starting points for minimizing base acquisitions 
        """
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = 'not selected'
        
        while 1:
            # (to avoid precision loss w/ really small numbers)
            if 0 and any(self.gains < -400):
                self.gains *= .1
            
            # otherwise, calculate probability of selecting each candidate point
            exp_g = np.exp(self.eta * self.gains)
            probs = exp_g / np.sum(exp_g)

            for acq, p in zip(self.acqs, probs):
                self.probs[acq.acq_name].append(p)

            idx = np.random.choice(range(self.K), p=probs)
            chosen = candidates.items()[idx]   
            
            if chosen is None:
                self.gains *= 0.05
            else:
                candidates[chosen[0]] = self.acqs[idx].select_next_theta(x0s)
                chosen = (chosen[0], candidates[chosen[0]])
                break
                    
        self.iter += 1
        
        rewards = self._calc_rewards(chosen, candidates)
        self.gains += rewards
        
        # TRACKING INFORMATION ====================================================
        self.choices.append(chosen[0])
        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(self.past_gains[-1] - self.past_gains[-2])
        
        return chosen[1]
    
    def _calc_rewards(self, chosen, candidates):
        # evaluate objective at chosen candidate, then
        # update model with new (theta, discr)
        new_discr = self.sim.f(chosen[1])
        self.model.update(chosen[1], new_discr)
        
        # Only update reward for chosen point
        rwd = -self.model.mu(chosen[1])           # <- more of a regret, actually
        
        rewards = np.zeros(self.K)
        for i, acq_name in enumerate(candidates.keys()):
            if acq_name == chosen[0]:
                rewards[i] = rwd
            else:
                rewards[i] = 0.
                
        return rewards

    
class NormalHedge(Portfolio):
    """
    NormalHedge: from "A Parameter-free Hedging Algorithm",
                       https://arxiv.org/pdf/0903.2851.pdf, 
                       Chaudhuri et al.
    """
    def __init__(self, model, acqs, sim, verbose=False, rng=None):
        super(NormalHedge, self).__init__(model, 'NormalHedge', acqs, verbose, rng)
        
        self.regrets = np.zeros(self.K)                    # cumulative regrets
        self.p = np.ones(self.K) * (1. / self.K)           # probability of choosing each acq.
        self.sim = sim                                     # simulator
        
        self._past_probs = np.ones((1, self.K)) * (1. / self.K)  # past probabilities (for analysis)
        self.past_regrets = np.zeros((1, self.K))                # past regrets       (     ""     )
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
            
        while 1:
            idx = np.random.choice(range(self.K), p=self.p)
            chosen = candidates.items()[idx] 
                    
            # IN CASE OF UNDERFLOW (e^large negative number):  
            if chosen is None:
                self.gains *= 0.05 # this will change the probabilities, but is necessary
            else:
                break
                
        val = self.sim.f(chosen[1])
        self.model.update(chosen[1], val)
        
        # losses = posterior estimates of discrepancies under the updated model
        losses = np.array([self.model.mu(c) for n, c in candidates.items()])
        learner_regret = np.sum(losses * self.p)
        self.regrets += learner_regret - losses
        
        self.past_regrets = np.vstack([self.past_regrets, self.regrets])
        
        R_plus = np.where(self.regrets >= 0., self.regrets, 0.)
        e = np.exp(1)
        # step 4 in NormalHedge algorithm given in paper:
        # find c_t such that (1/N) exp(sum ...) = euler's constant
        def objective(ct):
            lhs = np.mean(np.exp(R_plus**2 / (2.*ct)))
            return abs(lhs - e)
        
        res = minimize_scalar(objective, bounds=(1e-10, None))
        assert res.fun < 1e-5, 'unable to calculate ct - fun=%f, ct=%.2f' % (res.fun, res.x)
        ct = res.x
        
        # update probabilities
        self.p = (R_plus / ct) * np.exp(R_plus**2 / (2.*ct))
        self.p /= np.sum(self.p)
        
        self._past_probs = np.vstack([self._past_probs, self.p])
        
    @property
    def probs(self):
        """ Returns past probabilities as a dictionary. """
        out = dict()
        for i, acq_name in enumerate(self.acq_names):
            out[acq_name] = self._past_probs[:, i]
        return out
    
        
class Hedge(Portfolio):
    """
    From 'Portfolio Allocation for Bayesian Optimization' 
                 (Hoffman, Brochu, Freitas)
    http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2011.pdf
    """
    
    def __init__(self, model, acqs, sim, verbose=False, eta=1.0, rng=None, 
                 forget_factor=1.0, reward='default', debug=False):
        
        super(Hedge, self).__init__(model, 'Hedge', acqs, verbose, rng)

        self.eta = float(eta)
        self.sim = sim
        self.forget_factor = float(forget_factor)
         
        # gains for each base acquisition
        self.gains = np.zeros(self.K)
        
        # reward strategy
        self.reward = reward
        self.debug = debug
        
        # for nicely formatted debug stuff
        mx_len = max([len(a.acq_name) for a in self.acqs])
        self.pds = [' ' * (mx_len - len(a.acq_name)) for a in self.acqs]

        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        self.probs = {a.acq_name: [] for a in self.acqs}
        
        # differences between rewards given and true expected discrepancies
        self.reward_ds = {a.acq_name: [] for a in self.acqs}
        
    def select_next_theta(self, x0s=None):
        """
        Select next point to evaluate objective at based on Hedge algorithm.
        
        x0s: list of starting points for minimizing base acquisitions 
        """
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
        
        while 1:
            # calculate probability of selecting each candidate point
            exp_g = np.exp(self.eta * self.gains)
            probs = exp_g / np.sum(exp_g)
            probs /= np.sum(probs)
            
            idx = np.random.choice(range(self.K), p=probs)
            chosen = candidates.items()[idx] 
                    
            # IN CASE OF UNDERFLOW (e^large negative number):  
            if chosen is None:
                self.gains *= 0.05 # this will change the probabilities
            else:
                break
        
        r = self._calc_rewards(chosen, candidates)
        
        # Track difference between given rewards and true expected discrepancies of nominee points
        for (a_name, cand), rwd in zip(candidates.items(), r):
            true_exp_dscr = self.sim.noiseless_f(cand)
            self.reward_ds[a_name].append(-rwd - true_exp_dscr)
            #print(rwd, true_exp_dscr)
            #print(a_name, (rwd - true_exp_dscr))
        
        # update corresponding gains
        self.gains = self.forget_factor * self.gains + r

        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(r)
        self.choices.append(chosen[0])
        for acq, p in zip(self.acqs, probs):
            self.probs[acq.acq_name].append(p)
 
        return chosen[1]

    def _calc_rewards(self, chosen, candidates):
        """ Update model and calculate rewards/regrets. """
        val = self.sim.f(chosen[1])
        self.model.update(chosen[1], val)
        
        rewards = -np.array([self.model.mu(c) for n, c in candidates.items()])
        return rewards

    
class HedgeNorm(Portfolio):
    """
    Hedge w/ probs calculated differently.
    """
    
    def __init__(self, model, acqs, sim, verbose=False, eta=1.0, rng=None):
        super(HedgeNorm, self).__init__(model, 'HedgeNorm', acqs, verbose, rng)
        self.sim = sim
        self.eta = eta
         
        # gains for each base acquisition
        self.gains = np.zeros(self.K)

        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        self.probs = {a.acq_name: [] for a in self.acqs}
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
        
        while 1:
            exp_g = np.exp(self.eta * self.gains)
            inverse_probs = exp_g / np.sum(exp_g)
            exp_g = 1. - inverse_probs
            probs = exp_g / np.sum(exp_g)

            for acq, p in zip(self.acqs, probs):
                self.probs[acq.acq_name].append(p)

            idx = np.random.choice(range(self.K), p=probs)
            chosen = candidates.items()[idx]
            
            # avoiding underflow error when 
            if chosen is None:
                self.gains *= 0.05
            else:
                break
        
        discr = self.sim.f(chosen[1])
        self.model.update(chosen[1], discr)
        
        rwds = np.array([self.model.mu(c) for n, c in candidates.items()])
        self.gains += rwds
        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(rwds)
        
        return chosen[1]
    
    def _calc_rewards(self, chosen, candidates):
        pass

    
class HedgeAnnealed(Portfolio):
    """ Hedge with an annealing/oscillating learning rate. """
    
    def __init__(self, model, acqs, sim, verbose=False, rng=None):
        super(HedgeAnnealed, self).__init__(model, 'HedgeAnnealed', acqs, verbose, rng)
        self.sim = sim
         
        # gains for each base acquisition
        self.gains = np.zeros(self.K)

        self.past_gains = [np.zeros(self.K)]
        self.past_rewards = []
        self.probs = {a.acq_name: [] for a in self.acqs}
        
        self.iter = 0
        self.eta_range = np.linspace(0.5, 2.0, 35)
        
        
    def select_next_theta(self, x0s=None):
        candidates = OrderedDict()
        for acq in self.acqs:
            candidates[acq.acq_name] = acq.select_next_theta(x0s)
        
        while 1:
            # calculate probability of selecting each candidate point
            eta = self.eta_range[self.iter % len(self.eta_range)]
            exp_g = np.exp(eta * self.gains)
            probs = exp_g / np.sum(exp_g)
            probs /= np.sum(probs)
            
            idx = np.random.choice(range(self.K), p=probs)
            chosen = candidates.items()[idx] 
                    
            # IN CASE OF UNDERFLOW (e^large negative number):  
            if chosen is None:
                self.gains *= 0.05 # this will change the probabilities
            else:
                break
        
        discr = self.sim.f(chosen[1])
        self.model.update(chosen[1], discr)
        
        rwds = -np.array([self.model.mu(c) for n, c in candidates.items()])
        self.gains += rwds
        self.past_gains.append(deepcopy(self.gains))
        self.past_rewards.append(rwds)
        for acq, p in zip(self.acqs, probs):
            self.probs[acq.acq_name].append(p)
        
        return chosen[1]


    
# ===================================================================   
# NOT USING =========================================================
# ===================================================================
class ESP(Portfolio):
    """
    model: GP model of the discrepancy.
     acqs: Sequence of base acquisition functions.
    close: If two base acquisitions provided candidates that are very
           close to each other, consider them as just one candidate.
           "close enough" <=> (la.norm(t1 - t2) < close)
    """
    
    def __init__(self, model, acqs, verbose=False, close=None, rng=None):
        super(ESP, self).__init__(model, 'ESP', acqs, verbose, rng)
        
        self.close = close
        
    def select_next_theta(self, ax=None, N=5, G=6, S=7, track_acqs=False):
        """
        Select next theta using entropy search portfolio.
        See this paper:
        https://arxiv.org/pdf/1406.4625.pdf (2015, Shahriari et al.) for details.
        
        TODO: make this track multip_acq_names properly
        """
        
        candidates_raw = [a.select_next_theta() for a in self.acqs]
        if track_acqs:
            cand_names_raw = [a.acq_name for a in self.acqs]
            # ^ names of base acquisitions providing each candidate theta
        
        # Combine candidates that are very close to each other 
        # (or don't, if 'close' not provided)
        if self.close is None:
            candidates = candidates_raw
            if track_acqs:
                cand_names = cand_names_raw
        else:
            candidates = [candidates_raw.pop(0)]
            if track_acqs:
                cand_names = [cand_names_raw.pop(0)]
                
            while len(candidates_raw) != 0:
                next_cand = candidates_raw.pop(0)
                if track_acqs:
                    next_cand_name = cand_names_raw.pop(0)
                    
                if self._isclose(next_cand, candidates):
                    continue
                else:
                    candidates.append(next_cand)
                    if track_acqs:
                        cand_names.append(next_cand_name)
                    
        if len(candidates) == 1:
            if track_acqs:
                print 'All base acquisitions returned sameish point.'
            return candidates[0]
        
        if ax is not None:
            ax.scatter(candidates, np.zeros(len(candidates)), color='green', \
                       marker='x', s=150)
        
        zs = self._generate_representer_points(G)
        
        if track_acqs:
            for label, x, y in zip(cand_names, candidates, np.zeros(len(candidates))):
                plt.annotate(label, xy=(x, y), xytext=(-20, 20), \
                    textcoords='offset points', ha='right', va='bottom', \
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), \
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
        # =================================================================
        # For each candidate, draw N hallucinations. For each hallucinated
        # point, update a copy of the model with the point, then sample 
        # the updated model at each representer point, z.
        # =================================================================
        
        cand_utils = np.zeros(self.K)
        k = 0
        for cand in candidates:
            # N hallucinated discrepancies
            h_discrs = [
                self.model.sample_from_posterior(cand) for _ in range(N)]
            
            # plot hallucinated points
            #ax.scatter([cand for _ in range(N)], h_discrs, \
            #           marker='x', color='purple')
            
            p_k = np.zeros((N, G))
            
            # For each hallucinated discrepancy, create copy of model, 
            # update it w/ new discr, then sample at representer points.
            for n in range(N):
                h_GP = deepcopy(self.model)
                h_GP.update(cand, h_discrs[n])
                
                # S times, sample this updated model at each of G 
                # representer points.
                f_kn = np.array([
                    [h_GP.sample_from_posterior(z) for z in zs] 
                    for _ in range(S)
                ])
                assert f_kn.shape == (S, G), (
                    str(f_kn.shape) + ' should be (%d, %d)' % (S, G))
            
                # Now calculate p_kn_i for each representer point 
                # (by averaging over each point's corresponding S samples). 
                # Use add-one smoothing to prevent log(0) errors.
                for i in range(G):
                    p_k[n][i] = np.array(
                        [(sum([i==np.argmin(f_kn_s) for f_kn_s in f_kn]) + 1) / float(S+1)]
                    )
            
            u_k = p_k * np.log(p_k)
            u_k = -np.sum(u_k.flatten()) / float(N)
            
            cand_utils[k] = u_k
            k += 1
        
        if track_acqs:
            print zip(candidates, cand_utils)
            
        # return candidate w/ highest 'utility'
        return candidates[np.argmax(cand_utils)]
        
    def _isclose(self, arr, candidates):
        """
        Is arr 'close' to any array in candidates.
        Probably can make more efficient.
        
        Returns [la.norm(arr - a) for a in candidates].any()
        """
        return any([la.norm(arr - a) < self.close for a in candidates])
    
    def _generate_representer_points(self, G):
        # TODO: implement this for real
        bds = self.bounds
        D = self.input_dim
        
        repr_pts = np.array([
            [self.rng.uniform(bds[d][0], bds[d][1]) for d in range(D)]
            for i in range(G)
        ])
        
        return repr_pts
