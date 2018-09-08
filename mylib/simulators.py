"""
Simulator functions contained here. 
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from abc import abstractmethod
import pickle as pkl
from kde import create_grid
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Simulator(object):
    """
    Base class for simulators.
    """
    
    def __init__(self, sd=1.0, rng=None, input_dim=None):
        # standard deviation of observation noise
        self.obs_noise = float(sd)
        self.input_dim = int(input_dim)
        
        if rng is None:
            rng = np.random.RandomState() 
        self.rng = rng
        
    def f(self, theta):
        theta = np.array(theta).reshape(-1, self.input_dim)
        out = self._simulate(theta, noiseless=False)
        out = np.where(out > 1e-5, out, 1e-5)
        return out
    
    def noiseless_f(self, theta):
        theta = np.array(theta).reshape(-1, self.input_dim)
        out = self._simulate(theta, noiseless=True)
        out = np.where(out > 1e-5, out, 1e-5)
        return out

    @abstractmethod
    def _simulate(self, theta, *args, **kwargs):
        raise NotImplementedError()
        
    def plot(self, ax, cmap=None, cbar=True):
        if self.input_dim == 1:
            [b1, b2] = self.bounds
            thetas = create_grid(self)
            discrs = self.noiseless_f(thetas)
            
            ax.plot(thetas, discrs)
        elif self.input_dim == 2:
            if cmap is None:
                cmap = cm.binary
                
            thetas = create_grid(self)
            t1 = np.unique(thetas[:, 0])
            t2 = np.unique(thetas[:, 1])
            
            discrs = self.noiseless_f(thetas)
            
            cbar = ax.contourf(t1, t2, discrs.reshape(100, 100), cmap=cmap)
            
            if cbar:
                dloc = make_axes_locatable(ax)
                cax = dloc.append_axes('bottom', '10%', pad=0.25)
                plt.colorbar(cbar, cax=cax, orientation='horizontal')
            
        else:
            print('No plotting for 3D+ sims')
            
    def within_bounds(self, theta):
        theta = theta.reshape(self.input_dim, )
        
        for ti, (bimn, bimx) in zip(theta, self.bounds):
            if (ti < bimn) or (ti > bimx):
                return False
            
        return True
        
                
class Forrester(Simulator):
    """
    Forrester function. 1 minimum.
    """

    def __init__(self, sd=1.0, rng=None):
        self.bounds = [(0, 1)]
        super(Forrester, self).__init__(sd=sd, rng=rng, input_dim=1)
        self.name = 'Forrester'
        self.argmin = np.array([0.78]).reshape(1, 1)
            
    def _simulate(self, theta, noiseless=False):
        N = len(theta)
        fval = ((6*theta - 2)**2) * np.sin(12*theta - 4)
        
        if not noiseless:
            noise = self.rng.normal(0, self.obs_noise, N).reshape(N, 1)
            fval += noise
        
        return fval.reshape(N, 1) + 6.0

    
class SumOfSinoids(Simulator):
    """
    Sum of sinoids (1D). 2 minima.
    """
    def __init__(self, sd=1.0, rng=None):
        self.bounds = [(0, 1)]
        super(SumOfSinoids, self).__init__(sd=sd, rng=rng, input_dim=1)
        self.name = 'SumSinoids'
        
        self.argmin = np.array([0.17881194, 0.80713047]).reshape(2, 1)
            
    def _simulate(self, theta, noiseless=False):
        N = len(theta)
        t1 = np.sin(30*theta)
        t2 = -np.cos(20*(theta+30))
        t3 = -np.sin(40*theta)
        if not noiseless:
            out = t1 + t2 + t3
            out += self.rng.normal(0, self.obs_noise, N).reshape(N, 1)
            #print(out.shape, theta.shape)
            return 5 + out.reshape(N, 1)
        else:
            return 5 + (t1+t2+t3).reshape(N, 1)

        
class Complex1D(Simulator):
    
    def __init__(self, sd=1.0, rng=None):
        super(Complex1D, self).__init__(sd=sd, rng=rng, input_dim=1)
        self.bounds = [(0, 20)]
        self.name = 'Complex1D'
        self.argmin = None
        
    def _simulate(self, theta, noiseless=False):
        pass

    
class Simulator2D(Simulator):
    """
    Six hump camel function. From GPyOpt objective_examples.
    """

    def __init__(self, sd=1.0, rng=None):
        self.bounds = [(-2, 2), (-1, 1)]
        super(Simulator2D, self).__init__(sd=sd, rng=rng, input_dim=2)
        self.name = 'SixHump'
        self.argmin = np.array([[0.0898, -0.7126], [-0.0898, 0.7126]])
            
    def _simulate(self, theta, noiseless=False):
        N = len(theta)
        t1 = theta[:, 0]
        t2 = theta[:, 1]
        
        f1 = (4 - 2.1 * t1**2 + ((t1**4) / 3)) * t1**2
        f2 = t1 * t2
        f3 = (-4 + 4 * t2**2) * t2**2
        
        fval = f1 + f2 + f3
        if not noiseless:
            noise = self.rng.normal(0., self.obs_noise, N).reshape(N, 1)
        else:
            noise = np.zeros((N, 1))
        
        return fval.reshape(N, 1) + noise + 5

    
class BacterialInfectionsSimulator(Simulator):
    """
    Wrapper for Bacterial Infections simulator (run through Marko's 
    Matlab code).
    
    session: pymatlab.session_factory() session.
    """
    
    def __init__(self, session, sd=1.0, memoize=False, rng=None):
        raise NotImplementedError('Bacterial infections simulator is NOT available.')
        super(BacterialInfectionsSimulator, self).__init__(sd=sd, rng=rng, input_dim=3)
        self.bounds = [(0.001, 10.999), (0.001, 1.999), (0.001, 0.999)]
        self.session = session
        self.name = 'BactInf'
        self.obs_noise = np.sqrt(0.41)    # observation noise is actually heteroschedastic,
                                          # but we approximate it homoschedastically
            
        # theta w/ lowest discrepancy; as found in Marko's code
        self.argmin = np.array([3.589, 0.593, 0.097]).reshape(1, 3)
        
        self.memoize = memoize
        if memoize:
            if os.getcwd()[-9:] != 'hons-proj':
                print('memoization disabled: current directory must be hons-proj/')
                self.memoize = False
            else:
                self.data = pkl.load(open('BAC_DATA/BACTERIAL_RESULTS.p', 'rb'))
                self.tols = [0.04, 0.01, 0.005]
    
    def _simulate(self, theta, noiseless):
        if noiseless:
            print('Warning: noiseless simulations not available for BacterialInfections.')
            
        discrs = np.zeros(theta.shape[0])
        if self.memoize:
            new_theta = np.zeros((0, theta.shape[1]))
            new_idx = []
            
            for i, t in enumerate(theta):
                similar_idx = find_similar_points(self.data, t, self.tols, input_dim=3)
                if np.sum(similar_idx) > 0:
                    avg_discr = np.mean(self.data[similar_idx, -1])
                    discr = self.obs_noise * np.random.randn() + avg_discr
                    discr = np.clip(discr, self.data[:, -1].min(), self.data[:, -1].max())
                    discrs[i] = discr
                    print('memoized for theta=%s' % t)
                else:
                    new_theta = np.vstack([new_theta, t])
                    new_idx.append(i)

            if new_theta.shape[0] > 0:                    
                print('No memoization for thetas, idx:', new_theta, new_idx)
        else:
            new_theta = np.array(theta)
            new_idx = range(theta.shape[0])
            
        if new_theta.shape[0] > 0:
            print('Simulating bacterial sim. at theta=%s' % new_theta)
            try:
                self.session.putvalue('theta', new_theta)
                self.session.run('[dist, summaries] = run_bacterial_infections_simulator(theta, 1)')
                sim_discrs = self.session.getvalue('dist')
                sim_discrs = np.array(sim_discrs).reshape(-1, )
            
                for i, d in zip(new_idx, sim_discrs):
                    discrs[i] = d
                    
                return np.array(discrs)
            except RuntimeError:
                print('SIMULATION FAILED FOR THETA=%s' % theta)
                raise
        else:
            return np.array(discrs)
            

class BacterialInfections2D(BacterialInfectionsSimulator):
    """
    Bacterial infections, but one parameter is known, so only 2 parameters are optimized.
    
      session: pymatlab session.
    known_dim: index of known dimension.
    """
    
    def __init__(self, session, known_dim, sd=1.0, memoize=False, rng=None):
        assert known_dim in [0, 1, 2]
        super(BacterialInfections2D, self).__init__(session, sd=sd, memoize=memoize, rng=rng)
        
        self.known_dim = known_dim
        self.full_bounds = list(self.bounds)
        self.bounds = self.bounds[:known_dim]+self.bounds[known_dim+1:]
        self.name += '2D'
        self.input_dim = 2
        
        self.argmin = self.argmin.squeeze()
        self.known_param = self.argmin[known_dim]
        self.argmin = np.append(self.argmin[:known_dim], self.argmin[known_dim+1:]).reshape(1, 2)
        
    def _simulate(self, theta, noiseless):
        theta = np.insert(theta, self.known_dim, self.known_param, axis=1).reshape(-1, 3)
        return super(BacterialInfections2D, self)._simulate(theta, noiseless)
    
    def plot(self, ax):
        raise NotImplementedError()
        bac_data = pkl.load(open('BAC_DATA/BACTERIAL_RESULTS.p', 'rb'))
        assert bac_data.shape[1] == 4, 'what'
        
        bac_data = np.delete(bac_data, obj=0, axis=1)

        
class BactInf2D(Simulator):
    
    def __init__(self, sd=0.5, rng=None):
        super(BactInf2D, self).__init__(
            sd=sd, rng=rng, input_dim=2
        )
        self.bounds = [(0., 2.), (0., 1.)]
        self.name = 'I_BactInf2D'
        
        self.argmin = np.array([0.593, 0.097]).reshape(1, 2)
        
        # warning: atrocious coding practices ========
        if 1:
            data = pkl.load(open('BAC_DATA/bac_data_interp.p', 'rb'))
            self.data_dict = dict()
            for datum in data:
                k1, k2 = np.round(datum[0], 2), np.round(datum[1], 2)
                k = np.round(100. * np.array([k1, k2])).astype(int)
                assert k[0] % 2 == 0
                self.data_dict[tuple(k)] = datum[-1]
    
    # warning: atrocious coding practices ========
    def _simulate_single(self, theta):
        th_r1 = np.around(theta, decimals=2)
        th_r1_int = int(np.round(th_r1[0]*100))
        if th_r1_int % 2 != 0:
            if int(np.round(th_r1[0]))*100 % 2 != 0:
                th_r1 -= 0.01
            else:
                th_r1 += 0.01
                
        if np.sum((theta - th_r1)**2) <= 1e-6:
            k = np.round(100. * np.round(th_r1, 2)).astype(int)         
            return self.data_dict[tuple(k)]
      
        t1_abv = th_r1[0] > theta[0]        
        t2_abv = th_r1[1] > theta[1]
        th_r2 = th_r1 - [0.02, 0.] if t1_abv else th_r1 + [0.02, 0.]
        th_r3 = th_r1 - [0., 0.01] if t2_abv else th_r1 + [0., 0.01]
        th_r4 = np.array([th_r2[0], th_r3[1]])

        ths = np.array([th_r1, th_r2, th_r3, th_r4])
        ths[:, 0] = np.where(ths[:, 0] >= 0., ths[:, 0], 0.)
        ths[:, 0] = np.where(ths[:, 0] <= 2., ths[:, 0], 2.)
        ths[:, 1] = np.where(ths[:, 1] >= 0., ths[:, 1], 0.)
        ths[:, 1] = np.where(ths[:, 1] <= 1., ths[:, 1], 1.)
        if np.sum(np.diff(ths)) <= 1e-6:
            k = np.round(100. * np.round(ths[0, :], 2)).astype(int)
            return self.data_dict[tuple(k)]

        th_dists = np.array([np.sum((theta - t)**2) for t in ths])
        sum_d = np.sum(np.exp(-th_dists))
        discr = 0.
        for d, th in zip(th_dists, ths):
            k = np.round(100. * np.round(th, 2)).astype(int)
            discr += np.exp(-d) * self.data_dict[tuple(k)]
        
        discr /= sum_d
        
#         if discr >= 40.:
#             print('norms', norms)
#             print('sum(norms)', np.sum(norms))
#             print('sum_d', sum_d)
#             print('discr', discr)
#             print('ths', ths)
#             print('th_dists', th_dists)
#             for th in ths:
#                 k = np.round(100. * np.round(th, 2)).astype(int)
#             raw_input()
#         else:
#             print('sum(norms)', np.sum(norms))
#             raw_input()
        
        return discr
        
    def _simulate(self, theta, noiseless=False):
        theta = theta.reshape(-1, 2)
        n_thetas = len(theta)
        
        out = np.zeros(n_thetas)
        for i, t in enumerate(theta):
            out[i] = self._simulate_single(t)

        if not noiseless:
            out += self.rng.normal(0., self.obs_noise, size=out.shape)
            
        return out
    

class MultivariateGaussian(Simulator):
    """
    A multivariate Gaussian with randomly chosen mean and covariance matrix.
    By default, the covariance matrix is diagonal (i.e. parameter dimensions
        are not correlated).
    
    Returns standardized Euclidean distance as a discrepancy.
    """
    
    def __init__(self, sd=1.0, input_dim=3, rng=None):
        super(MultivariateGaussian, self).__init__(
            sd=sd, rng=rng, input_dim=input_dim
        )
        self.bounds = [(0, 20)]*input_dim
        self.name = 'MultivariateGaussian%d' % self.input_dim
        
        self.mean = np.array([10.]*self.input_dim).reshape(-1, self.input_dim)
        self.cov = np.diag(np.random.uniform(1., 6., size=(self.input_dim, )))
                    
        self.cov_inv = la.inv(self.cov)
        # denominator for pdf
        self.denom = np.sqrt(la.det(self.cov) * (2*np.pi)**self.input_dim)
        
        self.argmin = self.mean
        
        self.rec_sv = self.noiseless_f(np.array([20.]*self.input_dim)) / 1.5
    
    def _simulate(self, theta, noiseless):
        diag_2 = np.diag(self.cov)**2
        
        # Mahalanobis distance
        out = np.zeros(len(theta), )
        for i, t in enumerate(theta):
            out[i] = np.sum(((t - self.mean)**2).flatten() / diag_2)
        
        if not noiseless:
            out += self.rng.normal(0., self.obs_noise, size=out.shape)
            
        return out.reshape(-1, 1)
    
    def sample(self, n_samples=1, cov_spread=None):
        if cov_spread is None:
            cov = self.cov
        else:
            cov = self.cov * float(cov_spread)
        
        out = np.zeros((n_samples, self.input_dim))
        for j in range(n_samples):
            # make sure sample is within bounds
            while 1:
                smp = self.rng.multivariate_normal(self.mean.reshape(-1, ), cov)
                if all([smp[i] >= b[0] and smp[i] <= b[1] for i, b in enumerate(self.bounds)]):
                    out[j] = smp
                    break
        
        return out.reshape(n_samples, self.input_dim)
    
    def pdf(self, theta):
        theta = theta.reshape(-1, self.input_dim)
        out = np.zeros(len(theta), )
        for i, t in enumerate(theta):
            x_u = t - self.mean
            pr = -0.5 * np.dot(x_u, np.dot(self.cov_inv, x_u.T))
            out[i] = np.exp(pr) / self.denom
            
        return out.reshape(-1, )
        
        
class Alpine1(Simulator):
    """
    https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/objective_examples/experimentsNd.py
    """
    
    def __init__(self, sd=1.0, input_dim=3, rng=None):
        super(Alpine1, self).__init__(sd=sd, rng=rng, input_dim=input_dim)
        
        self.name = 'Alpine1'
        self.argmin = np.array([0.]*self.input_dim).reshape(1, self.input_dim)
        self.bounds = [(-10, 10)]*self.input_dim
        
        
    def _simulate(self, theta, noiseless):
        theta = theta.reshape(-1, self.input_dim)
        
        fval = np.sum(theta * np.sin(theta) + 0.1*theta, axis=1)
        
        if noiseless:
            return fval.reshape(-1, 1)
        else:
            noise = self.rng.normal(0, self.obs_noise, len(theta))
        
            return (fval + noise).reshape(-1, 1)
        
        
class Alpine2(Simulator):
    """
    https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/objective_examples/experimentsNd.py
    
    Not a great one, maybe shouldn't use this
    """
    
    def __init__(self, sd=1.0, input_dim=3, rng=None):
        super(Alpine2, self).__init__(sd=sd, rng=rng, input_dim=input_dim)
        
        self.name = 'Alpine2'
        self.argmin = np.array([7.917]*self.input_dim).reshape(1, self.input_dim)
        self.bounds = [(1, 10)]*self.input_dim
        
        
    def _simulate(self, theta, noiseless):
        theta = theta.reshape(-1, self.input_dim)
        
        f1 = np.cumprod(np.sqrt(theta), axis=1)
        f2 = np.cumprod(np.sin(theta), axis=1)
        fval = f1[:, self.input_dim-1] * f2[:, self.input_dim-1]
        
        if noiseless:
            return fval.reshape(-1, 1)
        else:
            noise = self.rng.normal(0, self.obs_noise, len(theta))
        
            return (fval + noise).reshape(-1, 1)
        

        
class Ackley(Simulator):
    """
    https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/objective_examples/experimentsNd.py
    
    
    """
    
    def __init__(self, sd=1.0, input_dim=3, rng=None):
        super(Ackley, self).__init__(sd=sd, rng=rng, input_dim=input_dim)
        
        self.name = 'Ackley'
        self.argmin = np.array([0.]*self.input_dim).reshape(1, self.input_dim)
        self.bounds = [(-32.768, 32.768)]*self.input_dim
        
        
    def _simulate(self, theta, noiseless):
        theta = theta.reshape(-1, self.input_dim)
        
        t1 = np.exp(-0.2 * np.sqrt((theta**2).sum(1) / self.input_dim))
        t2 = np.exp(np.cos(2*np.pi*theta).sum(1) / self.input_dim)
        fval = 20 + np.exp(1) - 20 * t1 - t2
        
        if noiseless:
            return fval.reshape(-1, 1)
        else:
            noise = self.rng.normal(0, self.obs_noise, len(theta))
        
            return (fval + noise).reshape(-1, 1)
        
        
        
        
        
        
        
        
        
        
from mylib.util import find_similar_points
