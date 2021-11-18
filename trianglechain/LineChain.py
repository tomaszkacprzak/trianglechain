import pylab as plt, numpy as np, scipy, warnings, math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from functools import partial
from scipy.stats import median_absolute_deviation
from sklearn.preprocessing import MinMaxScaler
from trianglechain.utils_plots import *
from trianglechain.BaseChain import BaseChain

class LineChain(BaseChain):

    def __init__(self, fig=None, size=4, **kwargs):
        
        super().__init__(fig=fig, size=size, **kwargs)

        self.add_plotting_functions(self.add_plot)

    def add_plot(self, data, plottype, prob=None, color='b', cmap=plt.cm.plasma, plot_histograms_1D=True, marker='o'):
        
        self.fig = plot_line_maringals(fig=self.fig,
                                       size=self.size,
                                       func=plottype,
                                       cmap=cmap,
                                       data=data,
                                       prob=prob,
                                       color=color, 
                                       plot_histograms_1D=plot_histograms_1D, 
                                       marker=marker,
                                       **self.kwargs)
        
        return self.fig


def get_param_pairs(n_output):
    
    pairs = []
    for i in range(n_output):
        for j in range(i+1, n_output):
            pairs+=[[i,j]]
    return pairs
    


def plot_line_maringals(data, prob=None, func='contour_cl', orientation='horizontal',   
                        color='b', cmap=plt.cm.plasma, marker='o',
                        ranges={}, ticks={}, n_bins=20, fig=None, size=4,
                        fill=True, grid=False, labels=None, plot_histograms_1D=True,
                        density_estimation_method='smoothing', n_ticks=3,
                        alpha_for_low_density=False, alpha_threshold=0,
                        subplots_kwargs={}, de_kwargs={}, hist_kwargs={}, axes_kwargs={},
                        labels_kwargs={}, grid_kwargs={}, scatter_kwargs={}, grouping_kwargs={}):

    data = ensure_rec(data)
    columns = data.dtype.names
    n_dim = len(columns)
    n_samples = len(data)

    if prob is not None:
        prob = prob/np.sum(prob)

    if orientation[0]=='h':
        n_rows = 1
        n_cols = (n_dim**2-n_dim)//2
    elif orientation[0]=='v':
        n_cols = 1
        n_rows = (n_dim**2-n_dim)//2

    # Create figure if necessary and get axes
    if fig is None:
        fig, _ = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*size, n_rows*size*0.7), **subplots_kwargs)
        fig.subplots_adjust(wspace=0.2)
        ax = np.array(fig.get_axes()).ravel()
    else:
        ax = np.array(fig.get_axes()).ravel()

    eps = 1e-6
    for c in columns:
        if c not in ranges:
            ranges[c] = (np.nan, np.nan) if c == 'EMPTY' else (np.amin(data[c])-eps, np.amax(data[c])+eps)
        if c not in ticks:
            ticks[c] = np.zeros(n_ticks) if c == 'EMPTY' else find_optimal_ticks((ranges[c][0], ranges[c][1]), n_ticks)
    
    # data
    pairs = get_param_pairs(n_dim)
    for k, (i,j) in enumerate(pairs):
        if columns[i]!='EMPTY' and columns[j]!='EMPTY':
            
            axc = ax[k]

            if func=='contour_cl':
                contour_cl(axc, data=data, ranges=ranges, columns=columns, i=i, j=j, fill=fill, color=color, de_kwargs=de_kwargs, prob=prob, density_estimation_method=density_estimation_method)
            
            if func=='density_image':
                density_image(axc, data=data, ranges=ranges, columns=columns, i=i, j=j, fill=fill, color=color, cmap=cmap, de_kwargs=de_kwargs, prob=prob,
                              density_estimation_method=density_estimation_method, alpha_for_low_density=alpha_for_low_density, alpha_threshold=alpha_threshold)
            
            elif func=='scatter':
                axc.scatter(data[columns[j]], data[columns[i]], c=color, cmap=cmap, **scatter_kwargs)
            
            elif func=='scatter_prob':
                sorting = np.argsort(prob)
                axc.scatter(data[columns[j]][sorting], data[columns[i]][sorting], c=prob[sorting], **scatter_kwargs)
            
            elif func=='scatter_density':
                scatter_density(axc, points1=data[columns[j]], points2=data[columns[i]], n_bins=n_bins, lim1=ranges[columns[j]], lim2=ranges[columns[i]], norm_cols=False, n_points_scatter=-1, cmap=cmap)

            axc.set(xlim=ranges[columns[j]], 
                    ylim=ranges[columns[i]],
                    xlabel=columns[j],
                    ylabel=columns[i])
            # axc.get_yaxis().set_major_formatter(FormatStrFormatter('%.3e'))
            # axc.get_xaxis().set_major_formatter(FormatStrFormatter('%.3e'))

    for a in ax:
        a.grid(grid)

    return fig

