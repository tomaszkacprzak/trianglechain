import numpy as np, arviz
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import root_scalar
from  trianglechain import utils_pj_hpd

from ekit import logger as logger_utils
LOGGER = logger_utils.init_logger(__name__)

def get_levels(samples, lnprob = None, levels_method = 'hdi', credible_interval = 0.68):
    if levels_method == 'hdi':
        return hdi(samples, credible_interval)
    elif levels_method == 'percentile':
        return percentile(samples, credible_interval)
    elif levels_method == 'PJ_HPD':
        if lnprob is not None:
            return PJ_HPD(samples, lnprob, credible_interval)
        else:
            LOGGER.error('PJ_HPD cannot be computed without probability of the samples.')
            LOGGER.info('hdi is used instead')
            return hdi(samples, credible_interval)
    else:
        LOGGER.error(f'{levels_method} is not known')
        LOGGER.info('hdi is used instead')
        return hdi(samples, credible_interval)

def percentile(samples, credible_interval = 0.68):
    s = 100 * (1 - credible_interval) / 2
    lower = np.percentile(samples, s)
    upper = np.percentile(samples, 100-s)
    return lower, upper

def hdi(samples, credible_interval = 0.68):
    lower, upper = arviz.hdi(samples, hdi_prob = credible_interval)
    return lower, upper

def PJ_HPD(samples, lnprobs, credible_interval = 0.68, interpolator = utils_pj_hpd.interpolate, **interp_kwargs):
    #from KiDS-1000
    post_1D = interpolator(samples, **interp_kwargs)
    sorted_chain = utils_pj_hpd.sort_according_to_lnprobs(samples, lnprobs)
    p_min = sorted_chain[0]
    p_max = sorted_chain[0]
    for par in sorted_chain:
        p_min_new, p_max_new, changed =  utils_pj_hpd.update_interval(par, p_min, p_max)
        if changed and  utils_pj_hpd.check_threshold(post_1D, p_min_new, p_max_new, credible_interval):
            p_min, p_max =  utils_pj_hpd.find_threshold_param(post_1D, p_min_new, p_max_new, changed, credible_interval)
            break
        p_min = p_min_new
        p_max = p_max_new
    return p_min, p_max

def get_uncertainty_band(lower, upper):
    return (upper-lower)/2

def uncertainty(samples, model, **model_kwargs):
    lower, upper = model(samples, **model_kwargs)
    return get_uncertainty_band(lower, upper)
