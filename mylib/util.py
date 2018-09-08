import numpy as np

def find_similar_points(arr, elem, tols, input_dim=None):
    if input_dim is None:
        input_dim = arr.shape[1]
    arr = arr[:, :input_dim]
    elem = elem.reshape(-1)
    bls = np.zeros(len(arr), dtype=bool)
    for i in range(arr.shape[1]):
        bls = np.logical_or(bls, (abs(arr[:, i] - elem[i]) > tols[i]).squeeze()).reshape(-1, )
    return np.invert(bls)

import pickle as pkl
import time
import seaborn as sns

from copy import deepcopy
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models import *
from acquisitions import *
from bayes_opt import *
from portfolios import *
import portfolios
from show_results import _C_MAP_

class SharedRNG(np.random.RandomState):
    
    def __init__(self, rs=None):
        super(SharedRNG, self).__init__(rs)
        
    def __deepcopy__(self, memo):
        return self

def set_up_portfolio(portfolio_class, init_model, sim, base_acquisitions=None, 
                     rng=None, **kwargs):
    if base_acquisitions is None:
        base_acquisitions = [LCB, EI, MPI, Random, PostVar]
        
    init_model = deepcopy(init_model)
    init_base_acqs = []
    for acq in base_acquisitions:
        init_base_acqs.append(acq(init_model, rng=rng))
    
    if portfolio_class == portfolios.Explorer and 'h_mult' not in kwargs:
        kwargs['h_mult'] = 0.05
        
    portfolio = portfolio_class(init_model, init_base_acqs, sim=sim, rng=rng, 
                                **kwargs)
    
    return portfolio


def set_up_acquisitions(acquisitions, init_model, sim, base_acqs=None, 
                        rng=None, **portfolio_args):
    out = []
    for acq_type in acquisitions:
        if issubclass(acq_type, Acquisition):
            model = deepcopy(init_model)
            out.append(acq_type(model, rng=rng))
        elif issubclass(acq_type, Portfolio):
            model = deepcopy(init_model)
            out.append(
                set_up_portfolio(acq_type, model, sim, base_acqs, rng,
                                 portfolio_args)
            )
        else:
            raise TypeError('Unknown portfolio/acquisition %s.' % acq_type)
            
    return out


def set_up_init_model(sim, n_init_pts=3, mean_f='mean', rng=None):
    assert mean_f in ['mean', 'zero']
    input_dim = sim.input_dim
    bounds = sim.bounds
    dims = []
    for (b1, b2) in bounds:
        dims.append(np.random.uniform(b1, b2, size=(n_init_pts, 1)))
    thetas = np.hstack(dims)
    
    if isinstance(sim, MultivariateGaussian):
        l = 1.6
        signal_var = sim.rec_sv
    elif type(sim) in [BacterialInfectionsSimulator, BacterialInfections2D]:
        l = np.array([3.6, 0.82, 0.46])
        signal_var = 18.
        obs_noise = 0.41
        if type(sim) is BacterialInfections2D:
            l = np.delete(l, sim.known_dim)
            signal_var = 14.
    else:
        l = 0.1
        signal_var = 10.
    
    discrs = sim.f(thetas)

    return GP(thetas, discrs, sim.obs_noise**2, sim.bounds, 
              np.array([l])*input_dim, signal_var=signal_var, 
              mean=mean_f, rng=rng)


def plot_base_probs(ax, probs, cmap):
    """
    Display base probabilities in an area chart.
    """
    ax.set_xlabel('Probability')
    ax.set_ylabel('Iteration')
    
    n_iters = len(probs.values()[0])
    iters = range(n_iters)
    xs = np.zeros(n_iters)
    
    for acq_name, probs in probs.items():
        ax.plot(iters, probs+xs, label=acq_name, color=cmap[acq_name])
        ax.fill_between(iters, xs, probs+xs, color=cmap[acq_name])
        xs += probs
        
    return ax


def calculate_gains_rewards(portfolios):
    assert isinstance(portfolios, list), "Pass the list of final portfolios, not i.e. a dictionary."
    n_iters = len(portfolios[0].past_rewards)
    n_runs = len(portfolios)
    K = len(portfolios[0].acqs)
    base_acq_names = [a.acq_name for a in portfolios[0].acqs]
    
    out_gains = {name: None for name in base_acq_names}
    out_rewards = {name: None for name in base_acq_names}
    
    for i, acq_name in enumerate(base_acq_names):
        gains = np.zeros((n_runs, n_iters))
        rwds = np.zeros((n_runs, n_iters))
        
        for j, portfolio in enumerate(portfolios):
            gains[j] = np.array(portfolio.past_gains)[1:, i]
            rwds[j] = np.array(portfolio.past_rewards)[:, i]
            
        avg_gains = np.mean(gains, axis=0)
        std_gains = np.std(gains, axis=0)
        avg_rwds = np.mean(rwds, axis=0)
        std_rwds = np.std(rwds, axis=0)
        
        out_gains[acq_name] = (avg_gains, std_gains)
        out_rewards[acq_name] = (avg_rwds, std_rwds)
        
    return out_gains, out_rewards


def show_gains(bopt, run_idx, save_figures=False):
    for name, portfolios in bopt.final_acquisitions.items():
        portfolio = portfolios[run_idx]
        if not isinstance(portfolio, Portfolio) or name == 'Baseline':
            continue
            
        f, ax = plt.subplots()
        ax.set_title('Gains per iteration (%s, run %d)' % (name, run_idx))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gain')
        gains = np.array(portfolio.past_gains)[1:]
        xs = range(len(gains))
        
        for i, acq_name in enumerate([a.acq_name for a in portfolio.acqs]):
            ax.plot(xs, gains[:, i], label=acq_name)
            
        plt.legend()
        if save_figures:
            fname = '../figures/gains_%s_%s.png' % (name, time.asctime())
            f.savefig(fname)
        plt.show()

        
def show_argmin_distances(bopt, run_idx, iter_idx, ax):
    assert bopt.input_dim <= 2, "Only for 1D/2D simulators"
    # PLOT ALL DISCREPANCIES ==============================
    if bopt.input_dim == 1:
        b1, b2 = bopt.bounds[0]
        TS = np.linspace(b1, b2, 200)
        YS = bopt.sim.noiseless_f(TS)
        ax.plot(TS, YS, label=u'True E[f(\u03B8)]')
    else:
        (b11, b12), (b21, b22) = bopt.bounds
        t1 = np.linspace(b11, b12, 200)
        t2 = np.linspace(b21, b22, 200)
        T1, T2 = np.meshgrid(t1, t2)
        TS = np.hstack([T1.reshape(200**2, 1), T2.reshape(200**2, 1)])
        YS = bopt.sim.noiseless_f(TS).reshape(200, 200)
        CS = ax.contourf(T1, T2, YS)
        cbar = plt.colorbar(CS)
    # =====================================================
    argmin = bopt.sim.argmin.reshape(-1, bopt.input_dim)
    m_lbl = 'Minimum' if len(argmin) == 1 else 'Minima'
    print(argmin, bopt.bounds)
    
    ys = argmin[:, 1] if bopt.input_dim == 2 else 0
    ax.scatter(argmin[:, 0], ys, label=m_lbl)
    
    for i, (acq_name, final_acqs) in enumerate(
        bopt.final_acquisitions.items()
    ):
        model = final_acqs[run_idx].model.get_state_at_earlier_iter(iter_idx)
        x0 = np.array([.5]*bopt.input_dim)
        argmin_ = minimize(
            fun=model.mu, x0=x0, bounds=bopt.bounds
        ).x.reshape(1, -1)
        dist = min([np.sum((amin - argmin_)**2) for amin in argmin])
        dist = np.sqrt(dist)
        
        lbl = '%s (%.2f)' % (acq_name, dist)
        x = argmin_[0, 0]
        y = 0 if bopt.input_dim == 1 else argmin_[0, 1]
        ax.scatter(x, y, label=lbl)

    return ax


def show_data_heatmap(bopt, acq_name, sim, ax):
    obs_data = bopt.get_observed_data_for_acq(acq_name)
    if sim.input_dim == 1:
        [(bmin, bmax)] = sim.bounds
        thetas = np.linspace(bmin, bmax, 100)
        discrs = sim.noiseless_f(thetas)
        
        ax.scatter(obs_data[:, 0], obs_data[:, 1], color='red', alpha=0.3)
        ax.plot(thetas, discrs, color='orange')
    else:
        (b1min, b1max), (b2min, b2max) = sim.bounds
        d2 = make_axes_locatable(ax)
        cax2 = d2.append_axes('bottom', '10%', pad=0.25)
        
        bins1 = np.linspace(b1min, b1max, 20)
        bins2 = np.linspace(b2min, b2max, 20)
        H, xbins, ybins = np.histogram2d(
            obs_data[:, 0], obs_data[:, 1], bins=[bins1, bins2]
        )
        H = H.T[::-1]
        sns.heatmap(
            H, xticklabels=[], yticklabels=[], ax=ax,
            cbar_ax=cax2, cbar_kws={'orientation': 'horizontal'}
        )
        
def show_hedge_diffs(bopt, ax, show_stds=False):
    f_hedges = bopt.final_acquisitions['Hedge']
    ds = {a_name: [] for a_name in f_hedges[0].acq_names}
    
    for f_hedge in f_hedges:
        rwd_ds = f_hedge.reward_ds
        for a_name, r_ds in rwd_ds.items():
            r_ds = np.array(r_ds).squeeze().tolist()
            ds[a_name].append(r_ds)

    for a_name, r_ds in ds.items():
        r_ms = np.mean(np.array(r_ds), axis=0).squeeze()
        r_ss = np.std(np.array(r_ds), axis=0).squeeze()
        
        r_ms = np.cumsum(r_ms)
        
        lbl = 'EI' if a_name == 'EI(tau=best)' else a_name
        lbl = 'PI' if a_name == 'MPI' else lbl
        xs = range(len(r_ms))
        ax.plot(xs, r_ms, label=lbl, color=_C_MAP_[a_name])
        
        if show_stds:
            print(r_ss[:5], r_ms[:5])
            rds_plus = r_ms + r_ss
            rds_mins = r_ms - r_ss
            ax.fill_between(xs, rds_plus, r_ms, color=_C_MAP_[a_name], alpha=0.2)
            ax.fill_between(xs, r_ms, rds_mins, color=_C_MAP_[a_name], alpha=0.2)
    
from simulators import *
