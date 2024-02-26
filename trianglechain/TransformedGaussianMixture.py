from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm

# import chaospy


def scale_fwd(x, param_bounds, param_bounds_trans):
    xs = x - param_bounds[0]
    xs = (
        xs
        / (param_bounds[1] - param_bounds[0])
        * (param_bounds_trans[1] - param_bounds_trans[0])
    )
    xs = xs + param_bounds_trans[0]
    return xs


def scale_inv(x, param_bounds, param_bounds_trans):
    xs = x - param_bounds_trans[0]
    xs = (
        xs
        * (param_bounds[1] - param_bounds[0])
        / (param_bounds_trans[1] - param_bounds_trans[0])
    )
    xs = xs + param_bounds[0]
    return xs


def trans_fwd(x, param_bounds, param_bounds_trans):
    xs = scale_fwd(x, param_bounds, param_bounds_trans)
    ppfx = norm.ppf(xs)
    if np.any(~np.isfinite(ppfx)):
        import ipdb

        ipdb.set_trace()
    return ppfx


def trans_inv(x, param_bounds, param_bounds_trans):
    xi = norm.cdf(x)
    xi = scale_inv(xi, param_bounds, param_bounds_trans)
    return xi


class TransformedGaussianMixture:
    def __init__(self, param_bounds=None, *args, **kwargs):

        self.eps = 1e-8
        self.bounds_trans = [self.eps, 1 - self.eps]
        self.param_bounds = param_bounds

        self.gm = GaussianMixture(*args, **kwargs)

    def set_bounds(self, X):

        if self.param_bounds is None:
            self.param_bounds = np.array(
                [
                    np.min(X, axis=0) - 10 * self.eps,
                    np.max(X, axis=0) + 10 * self.eps,
                ]
            ).T

    def fit(self, X, y=None):

        self.set_bounds(X)
        X_trans = self._transform_params_forward(X)
        return self.gm.fit(X_trans, y)

    def fit_predict(self, X, y=None):

        self.set_bounds(X)
        X_trans = self._transform_params_forward(X)
        return self.gm.fit_predict(X_trans, y)

    def predict_proba(self, X):

        X_trans = self._transform_params_forward(X)
        return self.gm.predict_proba(X_trans)

    def sample(self, n_samples=1):

        # print('ol')
        # import chaospy
        # means = self.gm.means_[:, :2]
        # covariances = self.gm.covariances_[:, :2, :2]
        # self.chaos_gm = chaospy.GaussianMixture(means, covariances)
        # X_trans, y = self.chaos_gm.sample(n_samples, rule="halton")
        X_trans, y = self.gm.sample(n_samples)
        X = self._transform_params_inverse(X_trans)
        return X, y

    def score(self, X, y=None):

        X_trans = self._transform_params_forward(X)
        return self.gm.score(X_trans)

    def score_samples(self, X):

        X_trans = self._transform_params_forward(X)
        return self.gm.score_samples(X_trans)

    def set_params(self, **params):

        return self.gm.set_params(**params)

    def get_params(self, deep=True):

        return self.gm.get_params(deep)

    def bic(self, X):

        X_trans = self._transform_params_forward(X)
        return self.gm.bic(X_trans)

    def aic(self, X):

        X_trans = self._transform_params_forward(X)
        return self.gm.aic(X_trans)

    def _transform_params_forward(self, X):

        X_trans = X.copy()
        for i in range(X.shape[1]):
            X_trans[:, i] = trans_fwd(
                x=np.array(X[:, i]),
                param_bounds=self.param_bounds[i],
                param_bounds_trans=self.bounds_trans,
            )

        return X_trans

    def _transform_params_inverse(self, X_trans):

        X = X_trans.copy()
        for i in range(X.shape[1]):
            X[:, i] = trans_inv(
                x=np.array(X_trans[:, i]),
                param_bounds=self.param_bounds[i],
                param_bounds_trans=self.bounds_trans,
            )

        return X
