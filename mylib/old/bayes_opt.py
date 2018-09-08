"""
Class for high-level Bayesian optimization.
"""
import time, os
from copy import deepcopy
import pickle as pkl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import models
import acquisitions
from calculate_distance import DistanceCalculator
import util

class BayesOpt(object):
    
    """
    Runs Bayesian optimization on provided simulator using provided 
      acquisition functions. Requires an initial GP model, a bandwidth h
      for kernel density estimation, a pymatlab session.
    ==================================================================================
    Args:
              sim: Simulator.
             acqs: Acquisition functions. 
                     Note: if this does multiple runs of BOpt using 
                     these acquisitions,it resets them to this state 
                     after each run.
       init_model: Initial GP model (that all acquisitions start with).
                h: Bandwidth for KDE.
          session: pymatlab Matlab session.
          verbose: Whether to print information while running.
    ==================================================================================
    Keeps copies of the final state of each acquisition in dict final_acqs:
      bopt.final_acqs: One entry for each acquisition. Each entry has shape (n_runs, ).
    ==================================================================================
    If Hedge algorithm is one of the provided acquisitions, tracks information
      about Hedge. The information is stored in lists for convenience, and 
      if you convert the lists to numpy arrays you get the shapes:
      
        -hedge_gains   (n_runs, n_iter, n_acqs): Gains for each base acquisition, at 
                                                   each iteration, for all runs.
        -hedge_rewards (n_runs, n_iter, n_acqs): Rewards for each base acq, each iter, 
                                                   all runs.
                                                   
    The gains/rewards are ordered the same way you pass 'acqs' to the constructor, i.e.
      if you pass acqs=[lcb, ei, hedge] to the constructor, hedge_rewards[0][5][1] 
      contains EI's reward at run 1, iteration 6. 
    ==================================================================================
    Also tracks TV distance between the approximate posterior calculated with
      each acquisition's GP model. Stores them in dictionary; one entry for each 
      acquisition. Each entry is a list with shape (n_runs, n_iter). Dictionary
      is called 'dists'.
    """
    # TODO: spaghetti code
    
    def __init__(self, sim, acqs, init_model, h, session, verbose=False):
        self.sim = sim
        self.acq_inits = acqs
        self.input_dim = init_model.input_dim
        self.bounds = init_model.bounds
        self.obs_noise = init_model.obs_noise
        self.h = float(h)
        self.verbose = verbose
        self.session = session # pymatlab session
        
        assert isinstance(init_model, models.GP)
        
        # For tracking information ===============================================
        # dict storing list of TV/KL dists for each acquistion
        # each list's shape: (n_runs, n_iter)
        self.dists = {a.acq_name: [] for a in self.acq_inits}
        
        # track Hedge information if it's being used
        for acq in self.acq_inits:
            if acq.acq_name == 'Hedge':
                self.hedge_gains = []
                self.hedge_rewards = []
                self.hedge_probs = {a.acq_name: [] for a in acq.acqs}
            if acq.acq_name == 'Exp3':
                self.exp3_probs = {a.acq_name: [] for a in acq.acqs}
            
        
        # keep final acquisitions from each run (and their corresponding GP models)
        self.final_acqs = {a.acq_name: [] for a in self.acq_inits}
            
        # track approximate PDFs (only for 1D sims; too much memory otherwise)
        if self.input_dim == 1:
            self.pdfs = {a.acq_name: [] for a in self.acq_inits}
            
        # true pdf of the simulator function (estimated using KDE)
        self.true_pdf = None
        # =========================================================================
        # for plotting
        self.clr = {'EI': 'blue', 'Bad': 'orange', 'LCB': 'green', \
                    'Rand': 'red', 'MPI': 'purple', 'Exp3': 'black', \
                    'Hedge': 'cyan'}
            
    def run_optimization(self, n_iter=30, n_runs=1, show_discr=False, \
                         show_acq=False, show_posterior=False, show_hedge_probs=False):
        """
        Run Bayesian optimization for 'n_iter' iterations on given simulator
          function, using given acquisition functions (and corresponding GP models).
          
        If n_runs > 1, runs BO multiple times so averaged results can be shown.
        
            show_discr: Whether to plot posterior estimate of discrepancy for each
                          acquisition, at each iteration.
              show_acq: Whether to also show acquisition on discrepancy plot.
              
        show_posterior: whether to plot approximate posterior distribution for 
                          each model (for each acquisition), at each iteration.
        """
        for j in range(n_runs):
            if self.verbose:
                print('Run #{} ============='.format(j))
                begin_time = time.time()
            
            # create copies of acquisitions (so that they're not reusing
            # GP models from previous runs)
            acqs = [deepcopy(acq) for acq in self.acq_inits]
            
            # For tracking Hedge information
            if 'Hedge' in [a.acq_name for a in acqs]:
                self.hedge_gains.append([])
                self.hedge_rewards.append([])
            
            # create fresh distance calculator
            d_calc = DistanceCalculator(self.input_dim, self.bounds, self.obs_noise, \
                                        self.sim, self.h, self.session, acqs)
            self.true_pdf = d_calc.true_pdf
            
            for i in range(n_iter):
                if self.verbose:
                    iter_time = time.time()
                
                # Use 4 starting points to avoid local minima
                x0s = []
                for _ in range(4):
                    init_pt = np.array(
                        [np.random.uniform(self.bounds[d][0], self.bounds[d][1]) 
                                        for d in range(self.input_dim)])
                    x0s.append(init_pt)
                
                # Each acquisition now picks a point and updates accordingly.
                for acq in acqs:
                    # Hedge/Exp3 self-update (so they can calculate rewards), 
                    # so we only need to manually update other acquisitions
                    if acq.acq_name == 'Hedge':
                        next_theta, rewards = acq.select_next_theta(x0s=x0s)
                        next_discr = self.sim.f(next_theta)
                        
                        # track info on base acquisitions
                        self.hedge_gains[j].append(acq.gains)
                        self.hedge_rewards[j].append(rewards)
                    elif acq.acq_name == 'Exp3':
                        next_theta = acq.select_next_theta(x0s=x0s)
                        next_discr = self.sim.f(next_theta)
                    else:
                        next_theta = acq.select_next_theta(x0s=x0s)
                        next_discr = self.sim.f(next_theta)
                        acq.model.update(next_theta, next_discr)
                    
                    # Calculate and store distance
                    d_calc.tv_distance(acq.model, acq.acq_name)
                    
                    # PLOT IF REQUESTED =========================================
                    if show_discr:
                        if acq.acq_name in ['Hedge', 'Exp3']:
                            print('acq chose {}'.format(acq.acq_name, acq.choices[-1]))
                            
                        if self.input_dim == 1:
                            f, ax = plt.subplots(figsize=(10, 3))
                            ax = acq.model.plot(
                                ax, point=(next_theta, next_discr), sim=self.sim, \
                                acq=acq.acq_name
                            )
                            if show_acq:
                                acq.plot(ax)
                            plt.show(); raw_input()
                        elif self.input_dim == 2:
                            f, axarr = plt.subplots(1, 2, sharex=True, sharey=True, \
                                                    figsize=(10, 10))
                            
                            acq.model.plot(axarr, point=(next_theta, next_discr), \
                                           sim=self.sim, acq=acq.acq_name)
                            
                            plt.legend(); plt.show(); raw_input()
                        else:
                            # no visualization for 3D+
                            pass
                        
                    if show_posterior:
                        if acq.acq_name in ['Hedge', 'Exp3']:
                            print('{} chose {}'.format(acq.acq_name, acq.choices[-1]))
                        d_calc.show_last_posteriors(acq.acq_name, theta=next_theta)
                    # ==============================================================
                    # this acquisition has finished, on to next one
                    
                if self.verbose:
                    print('iter %d took %.3f sec' % (i, time.time() - iter_time))
            
            # finished the run; update results dict; update Hedge choices, probs
            for acq in acqs:
                self.dists[acq.acq_name].append(d_calc.dists[acq.acq_name])
                self.final_acqs[acq.acq_name].append(acq)
                if acq.acq_name == 'Hedge':
                    for name, probs in acq.probs.items():
                        self.hedge_probs[name].append(probs)
                elif acq.acq_name == 'Exp3':
                    for name, probs in acq.probs.items():
                        self.exp3_probs[name].append(probs)
                            
            # retrieve approximate pdfs
            if self.input_dim == 1:
                for acq_name in self.pdfs.keys():
                    self.pdfs[acq_name].append(d_calc.m_pdfs[acq_name])
                
            if self.verbose:
                iter_time = time.time() - begin_time
                print('Run %d took %.3f seconds' % (j, iter_time))
                
        return self.dists
                        
    def plot_distances(self, ax):
        """
        Plots TV distance versus iteration for all acquisition functions.
        If multiple runs were made, plots averaged results from those.
        """
        n_runs = len(self.dists.values()[0])
        for name, dists in self.dists.items():
            to_plt = np.sum(np.array(dists), axis=0) / len(dists)
            
            ax.plot(range(len(to_plt)), to_plt, label=name, color=self.clr[name])
        
        if n_runs == 1:
            ax.set_title('TV distance vs. iteration')
        else:
            ax.set_title(
                'TV distance vs. iteration; averaged over {} runs'.format(n_runs)
            )
            
        ax.set_xlabel('Iteration')
        ax.set_ylabel('TV distance')
            
        return ax
    
    def hedge_choices(self):
        """
        Return number of times Hedge chose each acquisition 
          over all runs.
        """
        counts = {acq.acq_name: 0 for acq in self.final_acqs['Hedge'][0].acqs}
        
        for hedge in self.final_acqs['Hedge']:
            for acq_name in counts.keys():
                counts[acq_name] += len([c for c in hedge.choices if c == acq_name])
                
        return counts
    
    def show_hedge_probs(self, ax):
        """
        Does same as Hedge.show_base_probs(), but for averaged probabilities over all
          runs.
        """
        # average results
        avgs = {k: None for k in self.hedge_probs.keys()}
        for acq in self.hedge_probs.keys():
            avgs[acq] = np.sum(np.array(self.hedge_probs[acq]), axis=0)
            avgs[acq] /= len(self.hedge_probs[acq])
        
        # plot probabilities on area chart
        n_iter = len(self.hedge_probs.values()[0])
        x = range(n_iter)
        last_plt = np.zeros(n_iter)
        for acq, probs in avgs.items():
            a_plt = np.array(probs) + last_plt
                
            ax.plot(x, a_plt, label='P({})'.format(acq), color=self.clr[acq])
            ax.fill_between(x, a_plt, last_plt, color=self.clr[acq])
            last_plt = a_plt
                
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Probability of choosing acquisition')
        
        return ax
        
    def show_hedge_rewards(self):
        """
        Show distribution of rewards (negative posterior estimates of
          the discrepancy at chosen point) for ALL runs.
        """
        rewards = np.array(self.hedge_rewards)
        rewards = rewards.reshape(rewards.shape[0]*rewards.shape[1], rewards.shape[2])
        
        f, axarr = plt.subplots(rewards.shape[1], 1, sharex=True, \
                                figsize=(10, rewards.shape[1]*3))
        to_plot = []
        mn, mx = np.inf, -np.inf
        
        for i, acq in enumerate(self.final_acqs['Hedge'][-1].acqs):
            rwd = rewards[:, i]
            if rwd.min() < mn:
                mn = rwd.min()
            if rwd.max() > mx:
                mx = rwd.max()
                
            to_plot.append((rwd, acq.acq_name))
            
        f.suptitle('Distributions of rewards for each base acquisition)')
        bins = np.arange(mn, mx, (mx-mn)/100.0)
        for i, (rwd, name) in enumerate(to_plot):
            axarr[i].hist(rwd, bins=bins)
            axarr[i].set_title(name)
            
        plt.tight_layout()
        plt.show()
        
        
    def show_hedge_gains(self, ax):
        """
        Show each base acquisition's gain at each iteration. Normalized each 
          iteration's gains so area chart can be used.
          
        If multiple runs were made, this averages the gains from each run 
          and shows the averaged results.
        """
        raise NotImplementedError
    
    def show_final_models_discrepancies(self):
        if input_dim == 1:
            b1, b2 = self.bounds[0]
            x = np.arange(b1, b2, (b2-b1)/100.)

            true_pdf = self.true_pdf
            true_discr = self.sim.noiseless_f(x).reshape(100, )

            for acq in self.pdfs.keys():
                f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

                pdfs = np.array(self.pdfs[acq])[:, -1, :]
                pdfs = np.sum(pdfs, axis=0) / len(pdfs)
                ax1.plot(x, true_pdf, label='True')
                ax1.plot(x, pdfs, label='Approximate')
                ax1.set_title('{}-approx vs true posterior pdf'.format(acq))

                final_acqs = self.final_acqs[acq]
                discrs = np.zeros((len(final_acqs), 100))
                varis = np.zeros_like(discrs)
                ev_theta = []
                ev_discr = []
                for i, model in enumerate([a.model for a in final_acqs]):
                    discrs[i] = model.mu(x).reshape(100, )
                    varis[i] = np.array([model.v(t) for t in x]).reshape(100, )
                    ev_theta.append(model.thetas)
                    ev_discr.append(model.discrs)

                discrs = np.sum(discrs, axis=0) / len(final_acqs)
                varis = np.sum(varis, axis=0) / len(final_acqs)

                n_runs, n_ev = len(ev_theta), len(ev_theta[0])
                ev_theta = np.array(ev_theta).reshape(n_runs*n_ev, )
                ev_discr = np.array(ev_discr).reshape(n_runs*n_ev, )

                ax2.plot(x, true_discr, label='True')
                ax2.errorbar(x, discrs, yerr=varis, label='Approximate')
                ax2.scatter(ev_theta, ev_discr, label='Evidence')
                ax2.set_title('{}-approx vs true discrepancy'.format(acq))

                plt.legend()
            plt.show()
        else:
            (b11, b12), (b21, b22) = self.bounds
        
        
        
        
        
        
        
        
        
        
        
