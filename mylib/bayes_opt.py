import time, os
from copy import deepcopy
import pickle as pkl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy, norm
from scipy.optimize import minimize
from ipywidgets import *
from IPython.display import display, clear_output
import warnings

import models
from acquisitions import *
import portfolios
import kde
import util
from show_results import _PORTFOLIOS_
from simulators import *
from mcmc import sample_mcmc

class BayesOptimizer(object):
    
    def __init__(self, sim, acqs, init_model, refresh_model=True, h_mult=0.20, verbose=False):
        self.sim = sim
        self.initial_acquisitions = acqs
        self.h_mult = h_mult
        err_msg = 'duplicate acquisitions'
        assert len([a.__class__ for a in acqs]) == len(set([a.__class__ for a in acqs])), err_msg
        
        self.acq_names = [a.acq_name for a in self.initial_acquisitions]
        self.has_eta = dict()
        for acq in self.initial_acquisitions:
            if hasattr(acq, 'eta'): 
                self.has_eta[acq.acq_name] = True
            else:
                self.has_eta[acq.acq_name] = False

        self.input_dim = init_model.input_dim
        self.bounds = init_model.bounds
        self.num_init_pts = len(init_model.discrs.squeeze())
        
        self.refresh_model = refresh_model
        self.verbose = verbose

        # keep copies of final acquisitions (GP models, etc.)
        self.final_acquisitions = {a_name: [] for a_name in self.acq_names}
        
        self.memo = str(time.asctime())
        self.results_cache = {'TV': [], 'KL': [], 'probs': [], 'local': [], 
                              'argmin': [], 'n_runs': None, 'n_iter': None,
                              'refresh_model': self.refresh_model,
                              'choices': []}
        self.results_cache['init_data'] = np.hstack(
            [init_model.thetas.reshape(-1, self.input_dim), init_model.discrs.reshape(-1, 1)]
        )
        self.results_cache['h_mult'] = h_mult
        if 'Explorer' in self.acq_names:
            exp = [a for a in self.initial_acquisitions if a.acq_name == 'Explorer'][0]
            self.results_cache['EXPLORER_H_MULT'] = exp.h_mult
            self.results_cache['EXPLORER_H_DECR'] = exp.h_dcr
            
        for acq in self.initial_acquisitions:
            if hasattr(acq, 'forget_factor'):
                self.results_cache[acq.acq_name.lower() + '_forget_factor'] = acq.forget_factor
                
        self.results_cache['acq_names'] = self.acq_names
        
        # track portfolio hyperparameter information
        for acq in self.initial_acquisitions:
            if hasattr(acq, 'eta'):
                self.results_cache[acq.acq_name.lower()+'_eta'] = acq.eta
            
        self.ipy = os.environ['_'][-7:] == 'jupyter'
        
    def save(self, temp=False):
        """ Save the entire BayesOptimizer object (can be up to a gig in size). """
        s = self.sim.name
        t = time.asctime()
        d = t[4:7]+t[8:10]+'_'+t[11:16]+'.p'
        
        if temp:
            fname = 'tmp_files/TEMP_'+s+'.p'
        else:
            fname = 'experiment_results/'+s+'_'+d
        
        # pickle fails when saving BayesOptimizer with a 
        # BacterialInfectionsSimulator, so in that case set sim=None
        if self.sim.name == 'BactInf':
            to_save = self._copy_for_saving()
            try:
                pkl.dump(to_save, open(fname, 'wb'))
                return
            except Exception as e:
                print(e)
                print('Results dump failed.')
        else:
            try:
                pkl.dump(self, open(fname, 'wb'))
                return
            except Exception as e:
                print(e)
                print('Full bopt instance dump failed.')
                
        # just save what you can if it fails
        to_save = [self.sim.name, time.asctime(), self.results_cache, 
                   self.final_acquisitions, self.memo]
        try:
            pkl.dump(to_save, open(fname[:-2]+'_FAILED.p', 'wb'))
        except Exception as e:
            print(e)
            print('PSEUDO-DUMP FAILED!')
    
    def _save_bac_data(self):
        obs_data = self.observed_data
        if isinstance(self.sim, BacterialInfections2D):
            obs_data = np.insert(obs_data, self.sim.known_dim, self.sim.known_param, axis=1)

        pkl.dump(obs_data, open('TEMP_BAC_RES_OBS_DATA.p', 'wb'))
        bac_res = pkl.load(open('BAC_DATA/BACTERIAL_RESULTS.p', 'rb'))
        og_length = bac_res.shape[0]
        bac_res = np.vstack([bac_res, obs_data])

        if len(bac_res) <= og_length:
            print('Something went wrong cleaning up+saving BAC_RES results. Aborting save.')
        else:
            try:
                pkl.dump(bac_res, open('BAC_DATA/BACTERIAL_RESULTS.p', 'wb'))
            except IOError as e:
                print('== ERROR: dumping bacterial data ==================')
                print(e)
                print('===================================================')
    
    def save_results(self, title_ext=''):
        """ Save just the results (to save space). """
        sim_name = self.sim.name
        t = time.asctime()[4:-5].replace(' ', '_')
        # ^ unique filename so old results aren't overwritten
        n_r = self.results_cache['n_runs']
        n_i = self.results_cache['n_iter']
        
        full_metrics = ['tv', 'kl', 'local', 'argmin']

        if isinstance(self.sim, BacterialInfectionsSimulator):
            self._save_bac_data()
        
        # =====================================================================================
        # Split up and process self.results_cache for saving
        for acq_name in self.acq_names:
            to_save = dict() # save one results dict for each acquisition
            eta = ''
            if self.has_eta[acq_name]:
                eta = 'eta'+str(self.results_cache[acq_name.lower()+'_eta'])+'_'
                
            if acq_name in _PORTFOLIOS_+['Baseline']:
                init_ptf = [p for p in self.initial_acquisitions if p.acq_name == acq_name][0]
                to_save['with_postvar'] = 'PostVar' in init_ptf.acq_names
            
            for key, all_results in self.results_cache.items():
                if key == acq_name.lower()+'_eta':
                    to_save[key] = all_results
                    continue
                elif key[-3:] == 'eta':
                    continue
                elif key in ['probs']:
                    to_save['probs'] = []
                    for (p_name, res) in all_results:
                        if p_name == acq_name: 
                            to_save['probs'].append(res) 
                    continue
                elif key == 'choices':
                    to_save['choices'] = []
                    for (p_name, res) in all_results:
                        if p_name == acq_name: 
                            to_save['choices'].append(res)
                    continue
                elif (type(all_results) in [float, int, str, bool]) or (key in ['init_data', 'acq_names']):
                    to_save[key] = self.results_cache[key]
                    continue

                to_save[key] = []
                
                # check iterable
                try:
                    for _ in all_results: pass
                except TypeError:
                    to_save[key] = all_results
                    continue
                    
                # results_dict: acq_name -> stats
                for results_dict in all_results: 
                    try:
                        to_save[key].append(results_dict[acq_name])
                    except Exception as e:
                        print('== ERROR saving metric ====================')
                        print('error_msg', e)
                        print('metric_type', key)
                        print('all_results', all_results)
                        print('===========================================')
           
            # ====================================================================
            # Pickle and dump the data ===========================================
            if n_i is not None and n_r is not None:
                title = '%s_%dr_%di__%s' % (title_ext, n_r, n_i, t)
                try:
                    if self.has_eta[acq_name]:
                        title = ('%s_' % eta) + title
                        pkl.dump(to_save, open(
                            'experiment_results/%s/%s/%s.p' % (acq_name, sim_name, title), 'wb'
                        ))
                    else:
                        pkl.dump(to_save, open(
                            'experiment_results/%s/%s/%s.p' % (acq_name, sim_name, title), 'wb'
                        ))
                except Exception as e:
                    print('== ERROR: result dump failed... ==============')
                    print('error_msg', e)
                    print(acq_name, sim_name, eta, n_r, n_i, t)
                    print('==============================================')
            else:
                print('no results to save...')
            # ====================================================================
            
    def run_optimization(self, n_iter=30, n_runs=1, gp_plots=False, acq_plot=None, show_path=False):
        if self.ipy:
            sim_name = Text(value=self.sim.name + ' BO runs.')
            run_progress = IntProgress(min=1, max=n_runs, value=1, description='Run [1/%d]' % n_runs)
            iter_progress = IntProgress(min=1, max=n_iter, value=1, description='Iter [1/%d]' % n_iter)
            run_timebox = FloatText(value='0.00', description='last run time:')
            iter_timebox = FloatText(value='0.00', description='last iter time:')
            
            def clr_redraw():
                clear_output()
                display(sim_name)
                display(run_progress)
                display(iter_progress)
                display(run_timebox)
                display(iter_timebox)
                
            clr_redraw()
        
        for run in range(n_runs):
            if self.verbose and not self.ipy:
                print('Run {} ========================='.format(run))
            begin_time = time.time()
            
            # Create a new randomly initialized starting GP model
            if self.refresh_model:
                acquisitions = []
                new_model = util.set_up_init_model(self.sim, self.num_init_pts)
                for acq in self.initial_acquisitions:
                    if isinstance(acq, Acquisition):
                        cpy = deepcopy(acq)
                        cpy.model = new_model
                        acquisitions.append(cpy)
                    else:
                        acquisitions.append(acq.new_model(deepcopy(new_model)))
            else:
                acquisitions = [deepcopy(a) for a in self.initial_acquisitions]
            
            for i in range(n_iter):
                if self.verbose and not self.ipy:
                    it_str = 'Iter [%d/%d]' % (i+1, n_iter)
                    rn_str = 'Run [%d/%d] - ' % (run+1, n_runs)
                    print(rn_str + it_str)
                iter_time = time.time()

                # 8 points for multistart optimization
                x0s = np.hstack(
                    [np.random.uniform(d[0], d[1], size=(8,1)) for d in self.bounds]
                )

                for acquisition in acquisitions:
                    if self.verbose and isinstance(self.sim, BacterialInfectionsSimulator):
                        print('Acq: %s ==' % acquisition.acq_name)

                    try:
                        next_theta = acquisition.select_next_theta(x0s=x0s)
                    except:
                        print(acquisition.acq_name + ' messed up')
                        raise
                      
                    # Portfolios self-update (so they can calculate rewards)
                    if not isinstance(acquisition, portfolios.Portfolio):
                        next_discr = self.sim.f(next_theta)
                        acquisition.model.update(next_theta, next_discr)
                        
                    if i != 0 and (gp_plots == True or (gp_plots != 0 and (i % gp_plots) == 0)):
                        n_d = acquisition.model.discrs[-1]
                        
                        if self.input_dim == 1:
                            f, ax = plt.subplots()
                            acquisition.model.plot(
                                ax, point=(next_theta, n_d), sim=self.sim
                            )
                            ax.set_title(acquisition.acq_name)
                            plt.legend()
                            
                        elif self.input_dim == 2:
                            if acq_plot is None:
                                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                            else:
                                assert acq_plot in self.acq_names, 'please pass one string acq. name to plot'
                                f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
                            
                            a_name, a_model = acquisition.acq_name, acquisition.model
                            argmins = self.sim.argmin
                            
                            ax1.set_title(r"$\mu_{i}(\theta)$ under %s's model" % a_name)
                            ax2.set_title(r"$\upsilon_{i}(\theta)$ under %s's model" % a_name)
                            (b11, b12), (b21, b22) = self.bounds
                            t1 = np.linspace(b11, b12, 100)
                            t2 = np.linspace(b21, b22, 100)
                            T1, T2 = np.meshgrid(t1, t2)
                            grid_thetas = np.hstack([T1.reshape(100**2, 1), T2.reshape(100**2, 1)])
                            mus, vs = a_model.mu_and_v(grid_thetas)
                            
                            im1 = ax1.contourf(
                                t1, t2, mus.reshape(100, 100), cmap=cm.binary, vmin=mus.min(), 
                                vmax=mus.max()
                            )
                            d1 = make_axes_locatable(ax1)
                            cax1 = d1.append_axes('bottom', size='10%', pad=0.25) 
                            f.colorbar(im1, cax=cax1, orientation='horizontal')
                            
                            im2 = ax2.contourf(
                                t1, t2, vs.reshape(100, 100), cmap=cm.binary, vmin=vs.min(),
                                vmax=vs.max()
                            )
                            d2 = make_axes_locatable(ax2)
                            cax2 = d2.append_axes('bottom', size='10%', pad=0.25)
                            f.colorbar(im2, cax=cax2, orientation='horizontal')
                            ax2.scatter([next_theta[0, 0]], [next_theta[0, 1]], color='red', label='Newest evidence')
                            
                            acq = [a for a in acquisitions if a.acq_name == acq_plot]
                            if len(acq) != 0:
                                acq_vals = acq.acq(grid_thetas)
                                ax3.set_title(u'$\u03B1_{%s}$' % acq_plot)
                                cbar_acq = ax3.contourf(acq_vals.reshape(100, 100), cmap=cm.binary)
                                ax3.scatter(
                                    [next_theta[0]], [next_theta[1]], color='red', 
                                    label='Newest evidence'
                                )
                                plt.colorbar(mappable=cbar_acq, cax=ax4)
                            
                            # Plot past evidence
                            n_init = self.num_init_pts
                            ax1.scatter(
                                a_model.thetas[n_init:, 0], a_model.thetas[n_init:, 1], 
                                c=np.linspace(0, i+1, i+1), cmap=cm.Reds, alpha=0.6
                            )
                            ax1.scatter(
                                [next_theta[0, 0]], [next_theta[0, 1]], color='red', 
                                label='Newest evidence'
                            )
                            # Plot 'path' of data acquisition
                            if show_path:
                                ts = a_model.thetas[n_init:]
                                if len(ts) >= 2:
                                    alps = np.linspace(0.2, 1.0, 15)[::-1]
                                    for a, i in zip(alps, range(len(ts)-1)):
                                        p_stt, p_end = ts[-i-2], ts[-i-1]
                                        pth = np.vstack([p_stt.reshape(-1, 2), p_end.reshape(-1, 2)])
                                        ax1.plot(pth[:, 0], pth[:, 1], color='red', alpha=a)
                            
                            # Plot minimizer in blue
                            for amin in argmins:
                                ax1.scatter(
                                    amin[0], amin[1], color='blue'
                                )
                                ax1.scatter(
                                    amin[0], amin[1], color='blue'
                                )
                        else:
                            print('GP model plots not available for 3D+ sims')
                                
                        
                        plt.show()
                        pause = raw_input()
                        if pause == 'save':
                            f.savefig('figures/acq%s_iter%d_run%d_model.pdf' % (a_name, i, run))
                        clr_redraw()

                if self.verbose and not self.ipy:
                    print('iter %d took %.3f seconds' % (i, time.time() - iter_time))
                elif self.ipy:
                    iter_timebox.value = time.time() - iter_time
                    iter_progress.value += 1
                    iter_progress.description = 'Iter [%d/%d]' % (iter_progress.value, n_iter)
            
            # run finished; store final acquisitions
            for acquisition in acquisitions:
                self.final_acquisitions[acquisition.acq_name].append(acquisition)

            self.save(temp=True) # save in case of error later

            if self.verbose and not self.ipy:
                runtime = time.time() - begin_time
                print('Run %d took %.3f seconds' % (run+1, runtime))
            elif self.ipy:
                run_progress.value += 1
                run_progress.description = 'Run [%d/%d]' % (run_progress.value, n_runs)
                run_timebox.value = time.time() - begin_time
                iter_progress.value = 1
                iter_progress.description = 'Iter [%d/%d]' % (iter_progress.value, n_iter)
        
        if self.ipy:
            run_timebox.close()
            iter_timebox.close()
            run_progress.close()
            iter_progress.close()
            sim_name.close()
    
    @property
    def observed_data(self):
        obs_data = set() # set for removing duplicate data points
        for acq_type, acqs in self.final_acquisitions.items():
            for acq in acqs:
                for theta, discr in zip(acq.model.thetas, acq.model.discrs):
                    data = np.zeros(self.input_dim + 1)
                    data[:-1] = theta
                    data[-1] = discr
                    obs_data.add(tuple(data))
                    
        return np.array(list(obs_data))
    
    def get_observed_data_for_acq(self, acq_name):
        f_acqs = self.final_acquisitions[acq_name]
        obs_data = set() 
        for acq in f_acqs:
            for theta, discr in zip(acq.model.thetas, acq.model.discrs):
                datum = np.zeros(self.input_dim + 1)
                datum[:-1] = theta
                datum[-1] = discr
                obs_data.add(tuple(datum))
                
        return np.array(list(obs_data))

    def _get_ground_truth(self, h, n_grid_pts, n_local_pts, c_spr):
        """
        Loads/calculates ground truth, depending on the simulator.
        """
        out = dict()
        
        # Bacterial infections simulator (true, not interpolated)
        if isinstance(self.sim, BacterialInfectionsSimulator):
            bac_res = pkl.load(open('BAC_DATA/BACTERIAL_RESULTS.p', 'rb')).reshape(-1, 4)
            if isinstance(self.sim, BacterialInfections2D):
                k_dim = self.sim.known_dim
                k_prm = self.sim.known_param
                full_data = bac_res[np.where(abs(bac_res[:, k_dim] - k_prm) < 0.0001), :].squeeze()
                full_data = np.delete(full_data, obj=k_dim, axis=1).reshape(-1, 3)
            else:
                full_data = bac_res
            
            if len(full_data) < n_grid_pts**2:
                warnings.warn('Only %d BactInf data points available...' % len(full_data))
            
            grid_thetas = full_data[:n_grid_pts**2, :-1]
            grid_pdf_true = kde.calculate_bact_inf_pdf(self.sim, full_data[:n_grid_pts**2, :], h) 
            local_thetas = full_data[:n_local_pts**2, :-1]
            local_pdf_true = kde.calculate_bact_inf_pdf(self.sim, full_data[:n_local_pts**2, :], h)
            multiple_minima = False
            
            out['grid_thetas'] = grid_thetas
            out['grid_pdf_true'] = grid_pdf_true
            out['local_thetas'] = local_thetas
            out['local_pdf_true'] = local_pdf_true
            out['multiple_minima'] = multiple_minima
            
            return out
        # =======================================================
        
        # 1 or 2 dimensions => create grid
        if self.input_dim in [1,2]:
            grid_thetas = kde.create_grid(self.sim, n_grid_pts)
            grid_pdf_true = kde.calculate_true_pdf(self.sim, h, grid_thetas)
            
        # Gaussian => sample from it
        elif self.sim.name[:20] == 'MultivariateGaussian':
            n_grid_pts = max(n_grid_pts, 10000)
            print('Sampling %d points for MC estimate of TV/KL' % n_grid_pts)
            grid_thetas = self.sim.sample(n_samples=n_grid_pts, cov_spread=c_spr)
            grid_pdf_true = self.sim.pdf(grid_thetas)
            
        # Other 3D+ => use MCMC to sample points for ground truth
        else:
            n_samps = int(n_grid_pts / 50)
            grid_thetas = np.zeros((0, self.input_dim))
            for i in range(50):
                print('MCMC iter %d' % i)
                samples = sample_mcmc(self.sim, h, burnin=200, n_samples=n_samps, progress_bar=True)
                grid_thetas = np.vstack([grid_thetas, samples])
                
                if self.ipy:
                    clear_output()
                    
            grid_pdf_true = kde.calculate_true_pdf(self.sim, h, grid_thetas)

        # Create local_thetas as grid around minima
        local_thetas = kde.create_grid(self.sim, n_local_pts, local=True)
        if type(local_thetas) == list: # <- if simulator has multiple global minima
            multiple_minima = True
            local_pdf_true = []
            for local_t in local_thetas:
                local_pdf_true__ = kde.calculate_true_pdf(self.sim, h, local_t)
                local_pdf_true.append(local_pdf_true__)
        else:
            multiple_minima = False
            local_pdf_true = kde.calculate_true_pdf(self.sim, h, local_thetas)
            
        out['grid_thetas'] = grid_thetas
        out['grid_pdf_true'] = grid_pdf_true
        out['local_thetas'] = local_thetas
        out['local_pdf_true'] = local_pdf_true
        out['multiple_minima'] = multiple_minima
        
        return out
    
    def calculate_distances(self, metrics='all', n_grid_pts=None, n_local_pts=20, c_spr=2.5):
        """
        Returns:
            -TV distances,
            -KL divergences (true | approx),
            -"Local" TV distances,
            -Euclid. dist. between true minimizer t and estimated minimizer t' 
                    under GP model.

           metrics:  The metrics to calculate. Pass 'all' to calculate all, else
                     pass a list containing anything in ['tv', 'kl', 'local', 'argmin'].
        n_grid_pts:  Number of grid (or sampled) points to calculate full TV/KL at.
                     If simulator is MVGaussian w/ dim. > 2, it is number of sampled 
                        points for evaluating full TV/KL.
        n_local_pts: Number of grid points for 'local' TV (only around minima).
              c_spr: Factor to multiply Gaussian covariances by for Monte Carlo TV/KL calculations.
        """
        if metrics != 'all':
            chs = ['kl', 'tv', 'local', 'argmin']
            assert isinstance(metrics, list), "pass metrics='all' or as a list of strings."
            assert all([m in chs for m in metrics]), "choose from: %s" % chs
            
        if n_grid_pts is None:
            if self.input_dim <= 2:
                n_grid_pts = 100
            else:
                n_grid_pts = 10000
                
        use_metric = lambda m: (metrics == 'all') or (m in metrics)
            
        # Find suitable value for the bandwidth, h ===============================
        observed_data = self.observed_data
        mx = observed_data[:, -1].max()
        mn = observed_data[:, -1].min()
        h = mn + (mx - mn) * self.h_mult
        print('Using h=%.4f' % h)
        
        # Calculate ground truth =================================================
        ground_truth = self._get_ground_truth(h, n_grid_pts, n_local_pts, c_spr)
        grid_thetas, local_thetas = ground_truth['grid_thetas'], ground_truth['local_thetas']
        grid_pdf_true = ground_truth['grid_pdf_true']
        local_pdf_true = ground_truth['local_pdf_true']
        multiple_minima = ground_truth['multiple_minima']

        n_runs = len(self.final_acquisitions.values()[0])
        n_iter = len(self.final_acquisitions.values()[0][0].model.discrs) - self.num_init_pts
        n_acqs = len(self.final_acquisitions.values())

        # Progress bars in Jupyter ===============================================
        if self.ipy:
            sim_name = Text(value=self.sim.name+' metric calculation.')
            run_progress = IntProgress(min=1, max=n_runs, value=1, description='Run [1/%d]' % n_runs)
            run_timebox = FloatText(value='0.00', description='last run time:')
            iter_progress = IntProgress(min=1, max=n_iter, value=1, description='Iter [1/%d]' % n_iter)
            display(sim_name)
            display(run_progress)
            display(run_timebox)
            display(iter_progress)
            
            if len(self.final_acquisitions.keys()) > 1:
                acq_timebox = FloatText(value='0.00', description='last acq. time:')
                acq_progress = IntProgress(min=1, max=n_acqs, value=1, description='Acq. [1/%d]' % n_acqs)
                display(acq_timebox)
                display(acq_progress)

        # acq_name: [avg distances, standard deviations] (over n_runs)
        tv_dists = {a: None for a in self.final_acquisitions.keys()}
        kl_divs = {a: None for a in self.final_acquisitions.keys()}
        local_tvs = {a: None for a in self.final_acquisitions.keys()}
        min_dists = {a: None for a in self.final_acquisitions.keys()}
        
        # Do the actual metric calculations ======================================
        for acq_num, (name, acquisitions) in enumerate(self.final_acquisitions.items()):
            start_time = time.time()
            if self.verbose and not self.ipy:
                print('Calculating metrics for {}'.format(name))
                
            T = n_iter

            all_dists = np.zeros((n_runs, n_iter))
            all_kls = np.zeros((n_runs, n_iter))
            all_locals = np.zeros((n_runs, n_iter))
            all_argmins = np.zeros((n_runs, n_iter))
            
            err_msg = 'n_runs: %d; n_acqs: %d; acq: %s' % (n_runs, len(acquisitions), acquisitions[0].acq_name)
            assert n_runs == len(acquisitions), err_msg
              
            
            for run, acquisition in enumerate(acquisitions):
                run_time = time.time()
                for i in range(n_iter):
                    iter_time = time.time()
                    model = acquisition.model.get_state_at_earlier_iter(
                        T-i+self.num_init_pts
                    )
                    
                    # GRID METRICS (KL/FULL TV) ==================================
                    if use_metric('tv') or use_metric('kl'):
                        grid_pdf_m = kde.calculate_approx_pdf(
                            model, h, thetas=grid_thetas
                        )
                        if use_metric('tv'):
                            # TV distance, KL divergence
                            all_dists[run][n_iter-i-1] = 0.5 * np.sum(
                                np.abs(grid_pdf_true - grid_pdf_m)
                            )
                        if use_metric('kl'):
                            all_kls[run][n_iter-i-1] = entropy(
                                grid_pdf_true, grid_pdf_m
                            )
                    # ============================================================
                    # LOCAL TV ===================================================
                    if use_metric('local'):
                        if multiple_minima:   # <- multiple global minima
                            tv = 0.
                            for j in range(len(local_thetas)):
                                local_grid = local_thetas[j]
                                lcl_pdf_true = local_pdf_true[j]
                                lcl_pdf_m = kde.calculate_approx_pdf(
                                    model, h, local_grid
                                )
                                tv += 0.5 * np.sum(
                                    np.abs(lcl_pdf_true - lcl_pdf_m)
                                )
                                
                            tv /= len(local_thetas)
                        else:                 # <- only one global minimum
                            local_pdf_m = kde.calculate_approx_pdf(
                                model, h, local_thetas
                            )
                            tv = 0.5 * np.sum(
                                np.abs(local_pdf_true - local_pdf_m)
                            )
                        all_locals[run][n_iter-i-1] = tv
                    # ============================================================
                    # ARGMIN_DISTANCES ===========================================
                    if use_metric('argmin'):
                        argmin_m = model.find_minimizer()
                            
                        all_argmins[run][n_iter-i-1] = min(
                            [np.sqrt(np.sum((argmin_m - argmin)**2)) for argmin in self.sim.argmin.reshape(-1, self.input_dim)]
                        )
                        
                    if self.verbose and not self.ipy:
                        t_time = time.time() - iter_time
                        acq_msg = 'Acq: %s [%d/%d] - ' % (name, acq_num+1, n_acqs)
                        run_msg = 'Run [%d/%d] - ' % (run+1, n_runs)
                        ite_msg = 'Iter [%d/%d] - ' % (i+1, n_iter)
                        msg = '%s%s%sTime: %.3f' % (acq_msg, run_msg, ite_msg, t_time)
                        print(msg)
                    elif self.ipy:
                        iter_progress.value += 1
                        iter_progress.description = 'Iter [%d/%d]' % (iter_progress.value, n_iter)
                    
                if self.ipy:
                    run_timebox.value = time.time() - run_time
                    run_progress.value += 1
                    run_progress.description = 'Run [%d/%d]' % (run_progress.value, n_runs)
                    iter_progress.value = 1
                    iter_progress.description = 'Iter [1/%d]' % n_iter
                
            # Now calculate means and std. devs across the runs
            avg_dists = np.mean(all_dists, axis=0)   # <- full TV dists
            std_dists = np.std(all_dists, axis=0)
            tv_dists[name] = (avg_dists, std_dists) 
            avg_kls = np.mean(all_kls, axis=0)       # <- full KL divs.
            std_kls = np.std(all_kls, axis=0)
            kl_divs[name] = (avg_kls, std_kls)
            avg_locals = np.mean(all_locals, axis=0) # <- local TV dists
            std_locals = np.std(all_locals, axis=0)
            local_tvs[name] = (avg_locals, std_locals)
            avg_argmins = np.mean(all_argmins, axis=0)
            std_argmins = np.std(all_argmins, axis=0)
            min_dists[name] = (avg_argmins, std_argmins)
            
            elapsed = time.time() - start_time
            if self.verbose and not self.ipy:
                print('Acq. took %.3f seconds' % elapsed)
            elif self.ipy:
                if len(self.final_acquisitions.keys()) > 1:
                    acq_timebox.value = time.time() - start_time
                    acq_progress.value += 1
                    acq_progress.description = 'Acq. [%d/%d]' % (acq_progress.value, n_acqs)
                run_progress.value = 1
                run_progress.description = 'Run [%d/%d]' % (run_progress.value, n_runs)
        
        # Cache results ==========================================================
        cache_results = True
        if cache_results:
            if use_metric('tv'):
                self.results_cache['TV'].append(tv_dists)
            if use_metric('kl'):
                self.results_cache['KL'].append(kl_divs)
            if use_metric('local'):
                self.results_cache['local'].append(local_tvs)
            if use_metric('argmin'):
                self.results_cache['argmin'].append(min_dists)
            self.results_cache['n_runs'] = n_runs
            self.results_cache['n_iter'] = n_iter
        
        if self.ipy:
            sim_name.close()
            run_timebox.close()
            run_progress.close()
            iter_progress.close()
            
            if len(self.final_acquisitions.keys()) > 1:
                acq_progress.close()
                acq_timebox.close()
        
        return tv_dists, kl_divs, local_tvs, min_dists

    def calculate_base_probs(self, portfolio_name):
        """
        Takes a name of portfolio (string, e.g. 'Hedge' or 'Exp3').
        
        Returns a dictionary w/ base acquisition names (strings) as keys, 
            and their corresponding average (across runs) probabilities 
            of being chosen at each iteration, along with standard 
            deviations. 
        """
        final_portfolios = self.final_acquisitions[portfolio_name]
        all_probs = {a.acq_name: [] for a in final_portfolios[0].acqs}
        for portfolio in final_portfolios:
            for base_acq, run_probs in portfolio.probs.items():
                if base_acq == 'EI':
                    base_acq = 'EI(tau=best)'

                all_probs[base_acq].append(run_probs)
        
        stats = dict()
        for base_acq, probs in all_probs.items():
            probs = np.array(probs)
            means = np.mean(probs, axis=0)
            stds = np.std(probs, axis=0)
        
            stats[base_acq] = (means, stds)
        
        self.results_cache['probs'].append((portfolio_name, stats))
        return stats
    
    def _copy_for_saving(self):
        """For saving BayesOptimizer instances with BactInf sims"""
        init_acqs = []
        init_model = deepcopy(self.initial_acquisitions[0].model)
        for acq in self.initial_acquisitions:
            new_acq = deepcopy(acq)
            if isinstance(acq, portfolios.Portfolio):
                new_acq.sim = 'DELETED'
            init_acqs.append(new_acq)
        out = BayesOptimizer('DELETED', init_acqs, init_model, self.verbose)
        out.memo = deepcopy(self.memo)
        out.results_cache = deepcopy(self.results_cache)
        
        final_acqs_copy = {}
        for k, _acqs in self.final_acquisitions.items():
            if _acqs == [] or not isinstance(_acqs[0], portfolios.Portfolio): 
                final_acqs_copy[k] = deepcopy(_acqs) 
            else:
                final_acqs_copy[k] = []
                for _acq in _acqs:
                    _acq_copy = deepcopy(_acq)
                    _acq_copy.sim = 'DELETED'
                    final_acqs_copy[k].append(_acq_copy)
        
        out.final_acquisition = final_acqs_copy
        return out

    def get_choices_hist(self, portfolio_name):
        final_ps = self.final_acquisitions[portfolio_name]
        out = {a.acq_name: 0 for a in final_ps[0].acqs}
        for p in final_ps:
            for a in p.acqs:
                a_name = a.acq_name
                out[a_name] += len([a_n for a_n in p.choices if a_n == a_name])
        
        self.results_cache['choices'].append((portfolio_name, out))
        
        return out
        
    def save_all_results(self, metrics='all', title_ext='GOOD'):
        """ Calculate and save all results. """
        _ = self.calculate_distances(metrics=metrics)
        
        for p_name in _PORTFOLIOS_:
            try:
                _ = self.calculate_base_probs(p_name)
            except Exception as e:
                if p_name in self.acq_names:
                    print('='*60)
                    print('Error calculating base probabilities for %s' % p_name)
                    print(e)
                continue
            try:
                _ = self.get_choices_hist(p_name)
            except Exception as e:
                if p_name in self.acq_names:
                    print('='*60)
                    print('Error calculating choice histogram for %s' % p_name)
                    print(e)
                continue
        
        if 'Baseline' in self.acq_names:
            _ = self.get_choices_hist('Baseline')
            
        self.save_results(title_ext=title_ext)


        
        
        
        
        
pass
