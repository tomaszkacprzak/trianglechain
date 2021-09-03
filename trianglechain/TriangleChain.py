import pylab as plt, numpy as np, scipy, warnings
from matplotlib.ticker import FormatStrFormatter
from functools import partial
from scipy.stats import median_absolute_deviation
from sklearn.preprocessing import MinMaxScaler

class TriangleChain():

    def __init__(self, fig=None, size=4, labels=None, ranges={}, ticks={}, **kwargs):
        """
        :param fig: matplotlib figure to use
        :param size: figures size for a new figure, for a single panel. All panels are square 
        :param labels: labels for the parameters, default taken from columns
        :param ranges: dictionary with ranges for parameters, keys correspond to column names
        :param ticks: values for axis ticks, defaults taken from range with 3 equally spaced values
        :param n_bins: number of bins for 1d histograms, default: 100
        :param fill: if to fill contours
        :param density_estimation_method: method for density estimation, options:
                                          smoothing: first create a histogram of samples, and then smooth it with a Gaussian kernel corresponding to the variance of the 10% of the smallest eigenvalue of the 2D distribution
                                          gaussian_mixture: use Gaussian mixture to fit the 2D samples
                                          median_filter: use median filter on the 2D histogram
                                          kde: use TreeKDE, may be slow
                                          hist: simple 2D histogram
        :param de_kwargs: density estimation kwargs, dictionary with keys: 
                          n_points: number of bins for 2d histograms used to create contours, etc, default: n_bins
                          levels: density levels for contours, the contours will enclose this level of probability, default: [0.68, 0.95]
                          n_levels_check: number of levels to check when looking for density levels, default: 1000. More levels is more accurate, but slower
        :param grid_kwargs: kwargs for the plot grid, with keys:
                            fontsize_ticklabels: font size for tick labels, default 14
                            tickformat: numerical format for tick numbers, default {: 0.2e}
        :param hist_kwargs: kwargs for histograms, for plt.hist function

        Basic usage:
        samples: numpy recarray containing the samples, with named columns
        prob: probability corresponding to samples
        color: color for the contour_cl or scatter
        cmap: colormap for density_image or scatter_density

        tri = TriangleChain()
        tri.contour_cl(samples) # plot contours at given confidence levels
        tri.density_image(samples) # plot PDF density image
        tri.scatter(samples) # simple scatter plot
        tri.scatter_prob(samples) # scatter plot, with probability for each sample provided
        tri.scatter_density(samples) # scatter, color corresponds to probability



        """

        kwargs.setdefault('ticks', ticks)
        kwargs.setdefault('ranges', ranges)
        kwargs.setdefault('labels', None)
        kwargs.setdefault('n_bins', 100)
        kwargs.setdefault('de_kwargs', {})
        kwargs.setdefault('grid_kwargs', {})
        kwargs.setdefault('hist_kwargs', {})
        kwargs.setdefault('density_estimation_method', 'smoothing')
        kwargs.setdefault('fill', False)
        kwargs.setdefault('scatter_kwargs', {})
        kwargs['de_kwargs'].setdefault('n_points', kwargs['n_bins'])
        kwargs['de_kwargs'].setdefault('levels', [0.68, 0.95])
        kwargs['de_kwargs'].setdefault('n_levels_check', 1000)
        kwargs['de_kwargs']['levels'].sort()
        kwargs['grid_kwargs'].setdefault('fontsize_ticklabels', 14)
        kwargs['grid_kwargs'].setdefault('tickformat', '{: 0.2e}')
        kwargs['hist_kwargs'].setdefault('lw', 4)

        self.fig = fig
        self.size = size 
        self.kwargs = kwargs
        self.funcs = ['contour_cl', 'density_image', 'scatter', 'scatter_prob', 'scatter_density']

        for fname in self.funcs:

            f = partial(self.add_plot, plottype=fname)
            setattr(self, fname, f)
  
    def add_plot(self, data, plottype, prob=None, color='b', cmap=plt.cm.plasma, tri='lower', plot_histograms_1D=True):

        self.fig = plot_triangle_maringals(fig=self.fig, size=self.size, func=plottype, cmap=cmap, data=data, prob=prob, tri=tri, color=color, plot_histograms_1D=plot_histograms_1D, **self.kwargs)
        return self.fig

def histogram_2D(data_panel, prob, bins_x, bins_y):

    if prob is None:

        prob2d = np.histogram2d(*data_panel, bins=(bins_x, bins_y))[0].T.astype(np.float32)

    else:

        assert prob.shape[0] == data_panel[0].shape[0]

        hist2d_counts = np.histogram2d(*data_panel, bins=(bins_x, bins_y))[0].T.astype(np.float32)
        hist2d_prob = np.histogram2d(*data_panel, weights=prob, bins=(bins_x, bins_y))[0].T.astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob2d = hist2d_prob/hist2d_counts.astype(float)
        prob2d[hist2d_counts==0]=0

    prob2d = prob2d / np.sum(prob2d)
    return prob2d


def histogram_1D(data, prob, binedges):

    if prob is None:

        prob1D, _ = np.histogram(data, bins=binedges)

    else:

        assert prob.shape[0] == data.shape[0]

        hist_counts, _ = np.histogram(data, bins=binedges)
        hist_prob, _ = np.histogram(data, bins=binedges, weights=prob)
        prob1D = hist_prob/hist_counts.astype(np.float)
        prob1D[hist_counts==0]=0
    
    prob1D = prob1D/np.sum(prob1D)
    return prob1D


def get_density_grid_1D(data, binedges, bincenters, lims, prob=None, method='smoothing', de_kwargs={}):

    if method=='gaussian_mixture':

        if prob is not None:
            ind = np.random.choice(a=len(prob), size=10000, p=prob/np.sum(prob), replace=True)
            data_ = data[ind][:,np.newaxis]
        else:
            data_ = data[:,np.newaxis]

        from SoboLike.utils.TransformedGaussianMixture import TransformedGaussianMixture
        from sklearn.mixture import GaussianMixture
        clf = TransformedGaussianMixture(param_bounds=[lims], n_components=20, covariance_type='full')
        clf.fit(data_)
        logL = clf.score_samples(bincenters[:,np.newaxis])

        de = np.exp(logL - np.max(logL))
        de = de/np.sum(de)

    elif method=='smoothing':
        from scipy import signal
        from scipy.ndimage import convolve
        from scipy import ndimage
        n_pix = 10
        prob1D = histogram_1D(data=data, prob=prob, binedges=binedges)
        data_pixel = pixel_coords(data, [[binedges[0], binedges[-1]]], n_pix_img=len(bincenters))
        if prob is not None:
            ind = np.random.choice(prob1D.shape[0], p=prob1D, size=1000)
            data_pixel = data_pixel[0,ind]

        sig_pix = get_smoothing_sigma(data_pixel, prob1D.shape[0])
        kernel = signal.gaussian(n_pix, sig_pix)
        de = scipy.ndimage.convolve(prob1D, kernel, mode='reflect')
        de = de/np.sum(de)

        # de = np.convolve(a=prob1D, v=np.ones(5), mode='same')
        # de = signal.convolve2d(prob2d, kernel, mode='same', boundary='wrap')

    elif method=='median_filter':

        from scipy import ndimage
        prob1D = histogram_1D(data=data, prob=prob, binedges=binedges)
        de = ndimage.median_filter(prob1D, int(prob1D.shape[0]/10))


    elif method=='kde':

        from KDEpy import TreeKDE
        de = TreeKDE(kernel='gaussian', bw='ISJ').fit(data).evaluate(bincenters)

    elif method=='hist':

        de = histogram_1D(data=data, prob=prob, binedges=binedges)

    else:

        raise Exception('unknown density estimation method {}'.format(method))

    return de


def pixel_coords(x, ranges, n_pix_img):
    xt = np.atleast_2d(x.copy())
    for i in range(xt.shape[0]):
        try:
            xt[i] -= ranges[i][0]
            xt[i] /= (ranges[i][1]-ranges[i][0])
        except:
            import ipdb; ipdb.set_trace()
    return xt * n_pix_img

def get_smoothing_sigma(x, max_points=5000):

    x = np.atleast_2d(x)
    from sklearn.decomposition import PCA
    
    if x.shape[0]==2:
        pca = PCA()
        pca.fit(x.T)
        sig_pix = np.sqrt(pca.explained_variance_[-1])*0.1
    elif x.shape[0]==1:
        mad = median_absolute_deviation(x, axis=1)
        sig_pix = np.min(mad) * 0.1

    return sig_pix

def get_density_grid_2D(data, ranges, columns, i, j, prob=None, method='smoothing', de_kwargs={}):

    data_panel = np.vstack((data[columns[j]], data[columns[i]]))
    n_samples = len(data_panel)
    x_ls = np.linspace(*ranges[columns[j]], num=de_kwargs['n_points'])
    y_ls = np.linspace(*ranges[columns[i]], num=de_kwargs['n_points'])
    x_grid, y_grid = np.meshgrid(x_ls, y_ls)
    gridpoints = np.vstack([x_grid.ravel(), y_grid.ravel()])
    bins_x = np.linspace(x_ls[0], x_ls[-1], num=de_kwargs['n_points'] + 1)
    bins_y = np.linspace(y_ls[0], y_ls[-1], num=de_kwargs['n_points'] + 1)
    bins_x_centers = (bins_x[1:]+bins_x[:-1])/2.
    bins_y_centers = (bins_y[1:]+bins_y[:-1])/2.
    x_grid_centers, y_grid_centers = np.meshgrid(bins_x_centers, bins_y_centers)
    
    if method=='gaussian_mixture':

        if prob is not None:
            ind = np.random.choice(a=len(prob), size=10000, p=prob/np.sum(prob), replace=True)
            data_panel = data_panel[:,ind]

        bounds = [ranges[columns[j]], ranges[columns[i]]]

        from SoboLike.utils.TransformedGaussianMixture import TransformedGaussianMixture
        from sklearn.mixture import GaussianMixture
        clf = TransformedGaussianMixture(param_bounds=bounds, n_components=10, covariance_type='full')
        clf.fit(data_panel.T)
        logL = clf.score_samples(gridpoints.T)

        de = np.exp(logL - np.max(logL)).reshape(x_grid.shape)
        de = de/np.sum(de)

    elif method=='smoothing':

        prob2d = histogram_2D(data_panel, prob, bins_x, bins_y)
        prob2d = prob2d/np.sum(prob2d)
        from scipy import signal
        data_panel_pixel = pixel_coords(data_panel, [ranges[columns[j]], ranges[columns[i]]], n_pix_img=de_kwargs['n_points'])
        if prob is not None:
            ids = np.random.choice(a=len(prob), p=prob, size=1000, replace=True)
            data_panel_pixel = data_panel_pixel[:,ids]

        sig_pix = get_smoothing_sigma(data_panel_pixel)
        n_pix= int(np.ceil(sig_pix*5))

        kernel = np.outer(signal.gaussian(n_pix, sig_pix), signal.gaussian(n_pix, sig_pix))
        de = signal.convolve2d(prob2d, kernel, mode='same', boundary='wrap')


    elif method=='median_filter':

        from scipy import ndimage
        prob2d = histogram_2D(data_panel, prob, bins_x, bins_y)
        prob2d = prob2d/np.sum(prob2d)
        n_pix = int(prob2d.shape[0]/5)
        de = ndimage.median_filter(prob2d, n_pix)

    elif method=='kde':

        from KDEpy import TreeKDE
        de = TreeKDE(kernel='gaussian').fit(data_panel.T).evaluate(gridpoints.T).reshape(x_grid.shape)
      
    elif method=='hist':

        de = histogram_2D(data_panel, prob, bins_x, bins_y)

    else:

        raise Exception('unknown density estimation method {}'.format(method))

    de[~np.isfinite(de)] = 0
    de = de/np.sum(de)
    return de, x_grid_centers, y_grid_centers

def density_image(axc, data, ranges, columns, i, j, fill, color, cmap, de_kwargs, prob=None, density_estimation_method='smoothing'):
    """
    axc - axis of the plot
    data - numpy struct array with column data
    ranges - dict of ranges for each column in data
    columns - list of columns 
    i, j - pair of columns to plot
    fill - use filled contour
    color - color for the contour
    de_kwargs - dict with kde settings, has to have n_points, n_levels_check, levels, defaults below
    prob - if not None, then probability attached to the samples, in that case samples are treated as grid not a chain
    """
    kde, x_grid, y_grid = get_density_grid_2D(data=data, ranges=ranges, columns=columns, i=i, j=j, de_kwargs=de_kwargs, prob=prob, method=density_estimation_method)

    axc.pcolormesh(x_grid, y_grid, kde, cmap=cmap, shading='auto', vmin=0)

def get_confidence_levels(de, levels, n_levels_check):

    lvl_max = 0.9
    levels_check = np.linspace(0, np.amax(de)*lvl_max, n_levels_check)
    frac_levels = np.zeros_like(levels_check)

    for il, vl in enumerate(levels_check):
        pixels_above_level = de > vl
        frac_levels[il] = np.sum(pixels_above_level * de)

    levels_contour = [levels_check[np.argmin(np.fabs(frac_levels - level))] for level in levels][::-1]
    # print('levels_contour', levels_contour/np.amax(de)/lvl_max)
    # if np.any(levels_contour==levels_check[-1]):
        # print('contour hitting the boundary level {}'.format(str(levels_contour/np.amax(de)/lvl_max)))
    return levels_contour


def contour_cl(axc, data, ranges, columns, i, j, fill, color, de_kwargs, prob=None,  density_estimation_method='smoothing'):
    """
    axc - axis of the plot
    data - numpy struct array with column data
    ranges - dict of ranges for each column in data
    columns - list of columns 
    i, j - pair of columns to plot
    fill - use filled contour
    color - color for the contour
    de_kwargs - dict with kde settings, has to have n_points, n_levels_check, levels, defaults below
    prob - if not None, then probability attached to the samples, in that case samples are treated as grid not a chain
    """
                                                 
    de, x_grid, y_grid = get_density_grid_2D(i=i, j=j,
                                             data=data,
                                             prob=prob,
                                             ranges=ranges,
                                             columns=columns,
                                             method=density_estimation_method, 
                                             de_kwargs=de_kwargs)

    levels_contour = get_confidence_levels(de=de, levels=de_kwargs['levels'], n_levels_check=de_kwargs['n_levels_check'])

    with warnings.catch_warnings():
        # this will suppress all warnings in this block
        warnings.simplefilter("ignore")

        for lvl in levels_contour:
            if fill:
                axc.contourf(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=0.1)
                axc.contour(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=1, linewidths=2)
            else:
                axc.contour(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=1, linewidths=4)

def scatter_density(axc, points1, points2, n_bins=50, lim1=None, lim2=None, norm_cols=False, n_points_scatter=-1, colorbar=False, **kwargs):

    import numpy as np
    if lim1 is None:
        min1 = np.min(points1)
        max1 = np.max(points1)
    else:
        min1 = lim1[0]
        max1 = lim1[1]
    if lim2 is None:
        min2 = np.min(points2)
        max2 = np.max(points2)
    else:
        min2 = lim2[0]
        max2 = lim2[1]

    bins_edges1=np.linspace(min1, max1, n_bins)
    bins_edges2=np.linspace(min2, max2, n_bins)

    hv,bv,_ = np.histogram2d(points1,points2,bins=[bins_edges1, bins_edges2])

    if norm_cols==True:
        hv = hv/np.sum(hv, axis=0)[:,np.newaxis]

    bins_centers1 = (bins_edges1 - (bins_edges1[1]-bins_edges1[0])/2)[1:]
    bins_centers2 = (bins_edges2 - (bins_edges2[1]-bins_edges2[0])/2)[1:]

    from scipy.interpolate import griddata

    select_box = (points1<max1) & (points1>min1) & (points2<max2) & (points2>min2)
    points1_box, points2_box = points1[select_box], points2[select_box]

    x1,x2 = np.meshgrid(bins_centers1, bins_centers2)
    points = np.concatenate([x1.flatten()[:,np.newaxis], x2.flatten()[:,np.newaxis]], axis=1)
    xi = np.concatenate([points1_box[:,np.newaxis], points2_box[:,np.newaxis]],axis=1)


    if lim1 is not None:
        axc.set_xlim(lim1);
    if lim2 is not None:
        axc.set_ylim(lim2)
   

    if n_points_scatter>0:
        select = np.random.choice(len(points1_box), n_points_scatter)
        c = griddata(points, hv.T.flatten(), xi[select,:], method='linear', rescale=True, fill_value=np.min(hv) )
        sc = axc.scatter(points1_box[select], points2_box[select], c=c, **kwargs)
    else:
        c = griddata(points, hv.T.flatten(), xi, method='linear', rescale=True, fill_value=np.min(hv) )
        sorting = np.argsort(c)
        sc = axc.scatter(points1_box[sorting], points2_box[sorting], c=c[sorting],  **kwargs)

    if colorbar:
        plt.gcf().colorbar(sc, ax=axc)


def add_markers(fig, data_markers, tri='lower',scatter_kwargs={}):

    columns = data.dtype.names
    n_dim = len(columns)
    n_box = n_dim+1
    ax = np.array(fig.get_axes()).reshape(n_box, n_box)

    if tri[0]=='l':
        tri_indices = np.tril_indices(n_dim, k=-1)
    elif tri[0]=='u':
        tri_indices = np.triu_indices(n_dim, k=1)
    else:
        raise Exception('tri={} should be either l or u'.format(tri))

    for i, j in zip(*tri_indices):

        axc = get_current_ax(ax, tri, i, j)



def ensure_rec(data, column_prefix=''):

    if data.dtype.names is not None:
        return data
    
    else:

        n_rows, n_cols = data.shape
        dtype = np.dtype(dict(formats=[data.dtype]*n_cols, names=[f'{column_prefix}{i}' for i in range(n_cols)]))
        rec = np.empty(n_rows, dtype=dtype)
        for i in range(n_cols):
            rec[f'{column_prefix}{i}'] = data[:,i]
        return rec


def plot_triangle_maringals(data, prob=None, func='contour_cl', tri='lower', single_tri=True, color='b', cmap=plt.cm.plasma, ranges={}, ticks={}, n_bins=20, fig=None, size=4, fill=True, labels=None, plot_histograms_1D=True, density_estimation_method='smoothing', subplots_kwargs={}, de_kwargs={}, hist_kwargs={}, axes_kwargs={}, labels_kwargs={}, grid_kwargs={}, scatter_kwargs={}):

    data = ensure_rec(data)

    columns = data.dtype.names
    n_dim = len(columns)
    if single_tri:
        n_box = n_dim
    else:
        n_box = n_dim+1

    n_samples = len(data)
    
    if prob is not None:
        prob = prob/np.sum(prob)

    if tri[0]=='l':
        tri_indices = np.tril_indices(n_dim, k=-1)
    elif tri[0]=='u':
        tri_indices = np.triu_indices(n_dim, k=1)
    else:
        raise Exception('tri={} should be either l or u'.format(tri))

    # Create figure if necessary and get axes
    if fig is None:
        fig, _ = plt.subplots(nrows=n_box, ncols=n_box, figsize=(n_box*size, n_box*size), **subplots_kwargs)
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)
        for axc in ax.ravel():
            axc.axis('off')
    else:
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)

    for c in columns:
        if c not in ranges:
            print(c)
            ranges[c] = (np.amin(data[c])-1e-6, np.amax(data[c])+1e-6)
        if c not in ticks:
            ticks[c] = np.linspace(ranges[c][0], ranges[c][1], 5)[1:-1] 

    # Bins for histograms
    hist_binedges = {c: np.linspace(*ranges[c], num=n_bins + 1) for c in columns}
    hist_bincenters = {c: (hist_binedges[c][1:]+hist_binedges[c][:-1])/2 for c in columns}

    if len(color)==len(data):
        color_hist = 'k'
    else:
        color_hist = color

    def get_current_ax(ax, tri, i, j):

        if tri[0]=='u':
            if single_tri:
                axc = ax[i, j]
            else:
                axc = ax[i, j+1]
        elif tri[0]=='l':
            if single_tri:
                axc = ax[i, j]
            else:
                axc = ax[i+1, j]
        axc.axis('on')
        return axc

    # Plot histograms
    if plot_histograms_1D:
        for i in range(n_dim):

            prob1D = get_density_grid_1D(data=data[columns[i]],
                                         prob=prob,
                                         lims=ranges[columns[i]],
                                         binedges=hist_binedges[columns[i]],
                                         bincenters=hist_bincenters[columns[i]],
                                         method=density_estimation_method, 
                                         de_kwargs=de_kwargs)
            # prob1D = histogram_1D(data=data[columns[i]], prob=prob, binedges=hist_binedges[columns[i]], bincenters=hist_bincenters[columns[i]])

            axc = get_current_ax(ax, tri, i, i)
            axc.plot(hist_bincenters[columns[i]], prob1D, '-', color=color_hist, **hist_kwargs)
            if fill:
                axc.fill_between(hist_bincenters[columns[i]], np.zeros_like(prob1D), prob1D, alpha=0.1, color=color_hist)
            axc.set_xlim(ranges[columns[i]])


    # data
    for i, j in zip(*tri_indices):

        axc = get_current_ax(ax, tri, i, j)

        if func=='contour_cl':
            contour_cl(axc, data=data, ranges=ranges, columns=columns, i=i, j=j, fill=fill, color=color, de_kwargs=de_kwargs, prob=prob, density_estimation_method=density_estimation_method)
        if func=='density_image':
            density_image(axc, data=data, ranges=ranges, columns=columns, i=i, j=j, fill=fill, color=color, cmap=cmap, de_kwargs=de_kwargs, prob=prob, density_estimation_method=density_estimation_method)
        elif func=='scatter':
            axc.scatter(data[columns[j]], data[columns[i]], c=color, cmap=cmap, **scatter_kwargs)
        elif func=='scatter_prob':
            sorting = np.argsort(prob)
            axc.scatter(data[columns[j]][sorting], data[columns[i]][sorting], c=prob[sorting], **scatter_kwargs)
        elif func=='scatter_density':
            scatter_density(axc, points1=data[columns[j]], points2=data[columns[i]], n_bins=n_bins, lim1=ranges[columns[j]], lim2=ranges[columns[i]], norm_cols=False, n_points_scatter=-1, cmap=cmap)
            
        axc.set_xlim(ranges[columns[j]])
        axc.set_ylim(ranges[columns[i]])
        axc.get_yaxis().set_major_formatter(FormatStrFormatter('%.3e'))
        axc.get_xaxis().set_major_formatter(FormatStrFormatter('%.3e'))      


    # ticks
    n = n_dim-1
    # ticks = lambda i: np.linspace(ranges[columns[i]][0], ranges[columns[i]][1], 5)[1:-1]
    get_ticks = lambda i: ticks[columns[i]]

    # delete all ticks
    for axc in ax.ravel():
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axc.grid(False)


    # ticks
    if tri[0]=='l':
        for i in range(1, n_dim): # rows
            for j in range(0, i): # columns
                axc = get_current_ax(ax, tri, i, j)
                axc.yaxis.tick_left()
                axc.set_yticks(get_ticks(i))
        for i in range(1, n_dim): # rows
            for j in range(0,i+1): # columns
                axc = get_current_ax(ax, tri, i, j)
                axc.xaxis.tick_bottom()
                axc.set_xticks(get_ticks(j))
    elif tri[0]=='u':
        for i in range(0, n_dim-1): # rows
            for j in range(i+1, n_dim): # columns
                axc = get_current_ax(ax, tri, i, j)
                axc.yaxis.tick_right()
                axc.set_yticks(get_ticks(i))
        for i in range(0, n_dim-1): # rows
            for j in range(0, n_dim): # columns
                axc = get_current_ax(ax, tri, i, j)
                axc.xaxis.tick_top()
                axc.set_xticks(get_ticks(j))       


    def fmt_e(x):
        return grid_kwargs['tickformat'].format(x).replace('e+0', 'e+').replace('e-0', 'e-')

    # ticklabels
    if tri[0]=='l':
        # y tick labels 
        for i in range(1, n_dim):  
            axc = get_current_ax(ax, tri, i, 0)
            ticklabels = [fmt_e(t) for t in get_ticks(i)]
            axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')
        # x tick labels
        for i in range(0, n_dim): 
            axc = get_current_ax(ax, tri, n, i)
            ticklabels = [fmt_e(t) for t in get_ticks(i)]
            axc.set_xticklabels(ticklabels, rotation=90, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')    
    elif tri[0]=='u':
        # y tick labels 
        for i in range(0, n_dim-1):  
            axc = get_current_ax(ax, tri, i, n)
            ticklabels = [fmt_e(t) for t in get_ticks(i)]
            axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')
        # x tick labels
        for i in range(0, n_dim): 
            axc = get_current_ax(ax, tri, 0, i)
            ticklabels = [fmt_e(t) for t in get_ticks(i)]
            axc.set_xticklabels(ticklabels, rotation=90, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')    


    # grid
    if tri[0]=='l': 
        for i in range(1,n_dim):
            for j in range(i):  
                axc = get_current_ax(ax, tri, i, j)
                axc.grid(True)
    elif tri[0]=='u':
        for i in range(0,n_dim-1):
            for j in range(i+1,n_dim):  
                axc = get_current_ax(ax, tri, i, j)
                axc.grid(True) 

    # Axes labels
    if labels is None:
        labels = columns


    if tri[0]=='l':
        labelpad = 10
        for i in range(n_dim):
            axc = get_current_ax(ax, tri, i, 0)
            axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
            axc.yaxis.set_label_position("left")
            axc = get_current_ax(ax, tri, n, i)
            axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
            axc.xaxis.set_label_position("bottom")
    elif tri[0]=='u':
        labelpad = 20
        for i in range(n_dim):
            axc = get_current_ax(ax, tri, i, n)
            axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
            axc.yaxis.set_label_position("right")
            axc = get_current_ax(ax, tri, 0, i)
            axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
            axc.xaxis.set_label_position("top")



    plt.subplots_adjust(hspace=0, wspace=0)


    return fig
