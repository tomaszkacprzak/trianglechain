import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


def interpolate(sample, bins=1000):
    hist, bin_edges = np.histogram(sample, bins=bins, density=True)
    central_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    return interp1d(central_values, hist, fill_value=0, bounds_error=False)


def sort_according_to_lnprobs(samples, lnprobs):
    return samples[np.argsort(-lnprobs)]


def update_interval(p, p_min, p_max):
    changed = False
    if p < p_min:
        p_min = p
        changed = 1
    elif p > p_max:
        p_max = p
        changed = 2
    return p_min, p_max, changed


def check_threshold(func, p_min, p_max, threshold):
    # return quad(func, p_min, p_max)[0] > threshold
    p = np.linspace(p_min, p_max, 1000)
    return np.trapz(func(p), p) > threshold


def find_threshold_param(
    func, p_min, p_max, which_changed, credible_interval=0.68
):
    if which_changed == 1:

        def f(p):
            x = np.linspace(p, p_max, 1000)
            return np.trapz(func(x), x) - credible_interval

        res = root_scalar(f, x0=p_min, x1=p_max)
        return res.root, p_max
    elif which_changed == 2:

        def f(p):
            x = np.linspace(p_min, p, 1000)
            return np.trapz(func(x), x) - credible_interval

        res = root_scalar(f, x0=p_min, x1=p_max)
        return p_min, res.root
