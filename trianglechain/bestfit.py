import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from tqdm import trange

from ekit import logger as logger_utils
LOGGER = logger_utils.init_logger(__name__)

def get_bestfit(samples, lnprobs, bestfit_method):
    if bestfit_method == 'mode':
        lim = [np.min(samples), np.max(samples)]
        return mode(samples, lim)
    elif bestfit_method == 'mean':
        return np.mean(samples)
    elif bestfit_method == 'median':
        return np.median(samples)
    elif bestfit_method == 'best_sample':
        if lnprobs is not None:
            return best_sample(samples, lnprobs)
        else:
            LOGGER.error('best_sample cannot be computed without logprobability of the samples.')
            LOGGER.info('mode is used instead')
            lim = [np.min(samples), np.max(samples)]
            return mode(samples, lim)
    else:
        LOGGER.error(f'{bestfit_method} is not known')
        LOGGER.info('mode is used instead')
        lim = [np.min(samples), np.max(samples)]
        return mode(samples, lim)

def best_sample(samples, lnprobs):
    return samples[np.argmax(lnprobs)]

def get_means_and_medians(samples):
    names = samples.dtype.names
    means = np.empty(1,dtype=samples.dtype)
    medians = np.empty(1,dtype=samples.dtype)
    for n in names:
        means[n] = np.mean(samples[n]).item()
        medians[n] = np.median(samples[n]).item()
    return means, medians

def get_best_likelihood(params_chain, lnprobs, emu, cl_fid, inv_C, n_bin, ells, lims, prior_ind = [], gauss_mean = [], gauss_sigma = [], use_best_n=1):
    if use_best_n == 1:
        bl = params_chain[np.argmax(lnprobs)]
    else:
        sorted_indices = np.unique(-lnprobs, return_index=True)[1][:use_best_n]
        bl = params_chain[sorted_indices]

    def chi2(cosmo_p,cl_fid,inv_C,ells,lims):
        for i,p in enumerate(cosmo_p):
            if ((p<lims[i,0]) or (p>lims[i,1])):
                return np.inf
        cl = emu(cosmo_p.reshape(1,-1), ell=ells).flatten()
        def log_gauss(x,mu,sigma):
            return -0.5*(x-mu)**2/sigma**2 # + np.log(1.0/(np.sqrt(2*np.pi)*sigma))
        gaussian_add = 0
        for i,p in enumerate(prior_ind):
            gaussian_add += log_gauss(cosmo_p[p],gauss_mean[i],gauss_sigma[i])
        diff = cl - cl_fid
        return 0.5 * (diff.dot(inv_C)*diff).sum() - gaussian_add
    if use_best_n == 1:
        x0 = np.zeros(len(bl))
        for i,p in enumerate(bl):
            x0[i]=p
        best_l = np.empty(1,dtype=params_chain.dtype)
        res = minimize(chi2, x0, args = (cl_fid, inv_C, ells, lims), method= 'Nelder-Mead',options={'maxiter':5000})
        best_l = np.empty(1,dtype=params_chain.dtype)
        if res.success:
            for i,p in enumerate(params_chain.dtype.names):
                best_l[p]=res.x[i]
            return True, best_l, -res.fun
        else:
            best_l[0] =  params_chain[np.argmax(lnprobs)]
            return False, best_l, np.max(lnprobs)
    else:
        best_l = np.empty(use_best_n,dtype=params_chain.dtype)
        for ii in trange(use_best_n):
            x0 = np.zeros(len(bl[ii]))
            for i,p in enumerate(bl[ii]):
                x0[i]=p
            res = minimize(chi2, x0, args = (cl_fid, inv_C, ells, lims),
                           method= 'Nelder-Mead',)#options={'maxiter':5000})
            if res.success:
                for i,p in enumerate(params_chain.dtype.names):
                    best_l[ii][p]=res.x[i]
            else:
                best_l[ii] =  params_chain[np.argmax(lnprobs)]
        return True, get_mode(best_l, lims), np.max(lnprobs)

def get_best_likelihood_from_MCMC(params_chain, lnprobs):
    bl = np.empty(1,dtype=params_chain.dtype)
    bl[0] =  params_chain[np.argmax(lnprobs)]
    return bl, np.max(lnprobs)

def mode(sample, lim):
    x = np.linspace(lim[0], lim[1], 100)
    kde = gaussian_kde(sample)
    func = lambda x: -kde(x)
    res = minimize(func,x[np.argmax(kde(x))])
    if -res.fun > np.max(kde(x)):
        return res.x[0]
    else:
        return x[np.argmax(kde(x))]


def get_mode(samples,lims):
    names = samples.dtype.names
    modes = np.empty(1,dtype=samples.dtype)
    for i,n in enumerate(names):
        modes[n] = mode(samples[n], lims[i,:])
    return modes

def get_mean_median_best_likelihood_from_MCMC(params_chain, lnprobs):
    sorted_indices = np.unique(-lnprobs, return_index=True)[1][:10]
    samples = params_chain[sorted_indices]
    return get_means_and_medians(samples)

def get_all_bl_estimates(params_chain, lnprobs, emu, cl_fid, inv_C, n_bin, ells, lims, prior_ind = [], gauss_mean = [], gauss_sigma = [], just_names=False, use_best_n=1, both_minimization=False, flat_chi2minimization=False):
    names = ['means', 'medians', 'blMCMC', 'blmeanMCMC', 'blmedianMCMC', 'mode', 'chi2minimization']
    if both_minimization:
        names.append('improved chi2minimization')
    if flat_chi2minimization:
        names.append('flat_chi2minimization')
    if just_names:
        return names
    bl = []

    mean, median = get_means_and_medians(params_chain)
    bl.append(mean)
    bl.append(median)

    bl.append(get_best_likelihood_from_MCMC(params_chain, lnprobs)[0])

    mean, median = get_mean_median_best_likelihood_from_MCMC(params_chain, lnprobs)
    bl.append(mean)
    bl.append(median)

    bl.append(get_mode(params_chain, lims))

    if both_minimization:
        bl.append(get_best_likelihood(params_chain, lnprobs, emu,
                                      cl_fid, inv_C, n_bin, ells,
                                      lims, prior_ind, gauss_mean,
                                      gauss_sigma, use_best_n=1)[1])
    bl.append(get_best_likelihood(params_chain, lnprobs, emu,
                                  cl_fid, inv_C, n_bin, ells,
                                  lims, prior_ind, gauss_mean,
                                  gauss_sigma,use_best_n=use_best_n)[1])
    if flat_chi2minimization:
        bl.append(get_best_likelihood(params_chain, lnprobs, emu,
                                      cl_fid, inv_C, n_bin, ells,
                                      lims, prior_ind=[])[1])
    return bl, names

def get_all_bl_estimates_except_mode(params_chain, lnprobs, emu, cl_fid, inv_C, n_bin, ells, lims, prior_ind = [], gauss_mean = [], gauss_sigma = [], just_names=False, use_best_n=1, both_minimization=False, flat_chi2minimization=False):
    names = ['means', 'medians', 'blMCMC', 'blmeanMCMC', 'blmedianMCMC', 'chi2minimization']
    if both_minimization:
        names.append('improved chi2minimization')
    if flat_chi2minimization:
        names.append('flat_chi2minimization')
    if just_names:
        return names
    bl = []

    mean, median = get_means_and_medians(params_chain)
    bl.append(mean)
    bl.append(median)

    bl.append(get_best_likelihood_from_MCMC(params_chain, lnprobs)[0])

    mean, median = get_mean_median_best_likelihood_from_MCMC(params_chain, lnprobs)
    bl.append(mean)
    bl.append(median)

    if both_minimization:
        bl.append(get_best_likelihood(params_chain, lnprobs, emu,
                                      cl_fid, inv_C, n_bin, ells,
                                      lims, prior_ind, gauss_mean,
                                      gauss_sigma, use_best_n=1)[1])
    bl.append(get_best_likelihood(params_chain, lnprobs, emu,
                                  cl_fid, inv_C, n_bin, ells,
                                  lims, prior_ind, gauss_mean,
                                  gauss_sigma,use_best_n=use_best_n)[1])
    if flat_chi2minimization:
        bl.append(get_best_likelihood(params_chain, lnprobs, emu,
                                      cl_fid, inv_C, n_bin, ells,
                                      lims, prior_ind=[])[1])
    return bl, names
