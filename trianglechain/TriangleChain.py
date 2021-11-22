import pylab as plt, numpy as np, scipy, warnings, math
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from functools import partial
from scipy.stats import median_absolute_deviation
from sklearn.preprocessing import MinMaxScaler

class TriangleChain():

    def __init__(self, fig=None, size=4, **kwargs):
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

        kwargs.setdefault('ticks', {})
        kwargs.setdefault('ranges', {})
        kwargs.setdefault('labels', None)
        kwargs.setdefault('n_bins', 100)
        kwargs.setdefault('de_kwargs', {})
        kwargs.setdefault('grid_kwargs', {})
        kwargs.setdefault('hist_kwargs', {})
        kwargs.setdefault('labels_kwargs', {})
        kwargs.setdefault('line_kwargs', {})
        kwargs.setdefault('axvline_kwargs', {})
        kwargs.setdefault('density_estimation_method', 'smoothing')
        kwargs.setdefault('alpha_for_low_density', False)
        kwargs.setdefault('alpha_threshold', 0)
        kwargs.setdefault('n_ticks', 3)
        kwargs.setdefault('fill', False)
        kwargs.setdefault('grid', False)
        kwargs.setdefault('scatter_kwargs', {})
        kwargs.setdefault('grouping_kwargs', {})
        kwargs.setdefault('add_empty_plots_like', None)
        kwargs.setdefault('label_fontsize', 24)
        kwargs.setdefault('params', 'all')
        kwargs['de_kwargs'].setdefault('n_points', kwargs['n_bins'])
        kwargs['de_kwargs'].setdefault('levels', [0.68, 0.95])
        kwargs['de_kwargs'].setdefault('n_levels_check', 1000)
        kwargs['de_kwargs'].setdefault('smoothing_parameter1D', 0.1)
        kwargs['de_kwargs'].setdefault('smoothing_parameter2D', 0.1)
        kwargs['de_kwargs']['levels'].sort()
        if kwargs['fill']:
            kwargs['line_kwargs'].setdefault('linewidths', 2)
        else:
            kwargs['line_kwargs'].setdefault('linewidths', 4)
        kwargs['grid_kwargs'].setdefault('fontsize_ticklabels', 14)
        kwargs['grid_kwargs'].setdefault('tickformat', '{: 0.2e}')
        kwargs['grid_kwargs'].setdefault('font_family', 'sans-serif')
        kwargs['hist_kwargs'].setdefault('lw', 4)
        kwargs['labels_kwargs'].setdefault('fontsize', 24)
        kwargs['labels_kwargs'].setdefault('family', 'sans-serif')
        kwargs['grouping_kwargs'].setdefault('n_per_group', None)
        kwargs['grouping_kwargs'].setdefault('empty_ratio', 0.2)

        self.fig = fig
        self.size = size
        self.kwargs = kwargs
        self.funcs = ['contour_cl', 'density_image', 'scatter', 'scatter_prob', 'scatter_density']
        for fname in self.funcs:

            f = partial(self.add_plot, plottype=fname)
            setattr(self, fname, f)

    def add_plot(self, data, plottype, prob=None, color='b', cmap=plt.cm.plasma, tri='lower', plot_histograms_1D=True, scatter_vline_1D=False, label=None, show_legend=False):
        self.fig = plot_triangle_maringals(fig=self.fig, size=self.size, func=plottype, cmap=cmap, data=data, prob=prob, tri=tri, color=color, plot_histograms_1D=plot_histograms_1D, scatter_vline_1D=scatter_vline_1D, label=label, show_legend=show_legend, **self.kwargs)
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

        from trianglechain.TransformedGaussianMixture import TransformedGaussianMixture
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

        sig_pix = get_smoothing_sigma(data_pixel, prob1D.shape[0])*de_kwargs['smoothing_parameter1D']
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
        sig_pix = np.sqrt(pca.explained_variance_[-1])
    elif x.shape[0]==1:
        mad = median_absolute_deviation(x, axis=1)
        sig_pix = np.min(mad)

    return sig_pix

def extend_ranges(ranges):
    # to avoid edge effects in convolution
    ext_ranges = {}
    for param in ranges.keys():
        diff = (ranges[param][1]-ranges[param][0])*0.25
        ext_ranges[param] = [ranges[param][0]-diff, ranges[param][1]+diff]
    return ext_ranges

def get_density_grid_2D(data, ranges, columns, i, j, prob=None, method='smoothing', de_kwargs={}):

    data_panel = np.vstack((data[columns[j]], data[columns[i]]))
    n_samples = len(data_panel)
    #if method == 'smoothing':
    #    ranges = extend_ranges(ranges)
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

        from trianglechain.TransformedGaussianMixture import TransformedGaussianMixture
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

        sig_pix = get_smoothing_sigma(data_panel_pixel)*de_kwargs['smoothing_parameter2D']
        n_pix= int(np.ceil(sig_pix*5))

        kernel = np.outer(signal.gaussian(n_pix, sig_pix), signal.gaussian(n_pix, sig_pix))
        de = signal.convolve2d(prob2d, kernel, mode='same', boundary='symm')


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

def density_image(axc, data, ranges, columns, i, j, fill, color, cmap, de_kwargs, prob=None, density_estimation_method='smoothing', label=None, alpha_for_low_density=False, alpha_threshold = 0):
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
    if alpha_for_low_density:
        cmap_plt = plt.get_cmap(cmap)
        my_cmap = cmap_plt(np.arange(cmap_plt.N))
        cmap_threshold = int(cmap_plt.N * alpha_threshold)
        my_cmap[:cmap_threshold,-1] = np.linspace(0, 1, cmap_threshold)
        cmap = ListedColormap(my_cmap)
    axc.pcolormesh(x_grid, y_grid, kde, cmap=cmap, vmin=0, label = label, shading='auto')

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


def contour_cl(axc, data, ranges, columns, i, j, fill, color, de_kwargs, line_kwargs, prob=None,  density_estimation_method='smoothing', label=None, alpha=1):
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
                axc.contourf(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=0.1*alpha)
                axc.contour(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=1*alpha, label=label, **line_kwargs)
            else:
                axc.contour(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=1*alpha, label=label, **line_kwargs)

def scatter_density(axc, points1, points2, n_bins=50, lim1=None, lim2=None, norm_cols=False, n_points_scatter=-1, colorbar=False, label = None, **kwargs):

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
        sc = axc.scatter(points1_box[select], points2_box[select], c=c, label=label, **kwargs)
    else:
        c = griddata(points, hv.T.flatten(), xi, method='linear', rescale=True, fill_value=np.min(hv) )
        sorting = np.argsort(c)
        sc = axc.scatter(points1_box[sorting], points2_box[sorting], c=c[sorting], label=label, **kwargs)

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


def plot_triangle_maringals(data, prob=None, params='all',
                            func='contour_cl', tri='lower',
                            single_tri=True, color='b', cmap=plt.cm.plasma,
                            ranges={}, ticks={}, n_bins=20, fig=None, size=4,
                            fill=True, grid=False, labels=None, plot_histograms_1D=True,
                            scatter_vline_1D=False,
                            label=None, density_estimation_method='smoothing', n_ticks=3,
                            alpha_for_low_density=False, alpha_threshold=0,
                            subplots_kwargs={}, de_kwargs={}, hist_kwargs={}, axes_kwargs={}, line_kwargs={},
                            labels_kwargs={}, grid_kwargs={}, scatter_kwargs={}, grouping_kwargs={}, axvline_kwargs={},
                            add_empty_plots_like=None, label_fontsize=12, show_legend=False):
    data = ensure_rec(data)
    empty_columns =[]
    if add_empty_plots_like is not None:
        columns = data.dtype.names
        data2 = ensure_rec(add_empty_plots_like)
        columns2 = data2.dtype.names
        new_data = np.zeros(len(data),dtype=data2.dtype)
        for c in columns2:
            if c in columns:
                new_data[c] = data[c]
            else:
                new_data[c] = data2[c][np.random.randint(0,len(data2),len(data))]
                empty_columns.append(c)
        data = new_data
    if params != 'all':
        data = data[params]
    columns = data.dtype.names

    #needed for plotting chains with different automatic limits
    current_ranges = {}
    current_ticks = {}
    def find_alpha(column, empty_columns):
        if column in empty_columns:
            return 0
        else:
            return 1

    try:
        grouping_indices = np.asarray(grouping_kwargs['n_per_group'])[:-1]
        ind = 0
        for g in grouping_indices:
            ind += g
            columns = np.insert(np.array(columns,dtype='<U32'), ind, 'EMPTY')
            ind+=1
    except:
        pass
    hw_ratios = np.ones_like(columns,dtype=float)
    for i,l in enumerate(columns):
        if l=='EMPTY':
            hw_ratios[i] = grouping_kwargs['empty_ratio']

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
        fig, _ = plt.subplots(nrows=n_box, ncols=n_box, figsize=(sum(hw_ratios)*size, sum(hw_ratios)*size),
                              gridspec_kw = {'height_ratios': hw_ratios, 'width_ratios': hw_ratios}, **subplots_kwargs)
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)
        for axc in ax.ravel():
            axc.axis('off')
    else:
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)

    # round to get nicer ticks
    def round_to_significant_digits(number, significant_digits):
        try:
            return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) -1 )
        except:
            return number

    def find_optimal_ticks(range_of_param, n_ticks=3):
        diff = range_of_param[1]-range_of_param[0]
        ticks = np.zeros(n_ticks)

        # mathematical center and tick interval
        diff_range = diff/(n_ticks+1)
        center = range_of_param[0] + diff/2

        #first significant digit for rounding
        significant_digit = math.floor(np.log10(diff_range))

        for i in range(10*n_ticks):
            rounded_center = np.around(center, -significant_digit + i)
            if abs(rounded_center-center)/diff < 0.05:
                break
        for i in range(10*n_ticks):
            rounded_diff_range = np.around(diff_range, -significant_digit)
            start = rounded_center - (n_ticks-1)/2 * rounded_diff_range
            for i in range(n_ticks):
                if n_ticks%2==0:
                    ticks[i] = np.around(start + i*rounded_diff_range, -significant_digit+1)
                else:
                    ticks[i] = np.around(start + i*rounded_diff_range, -significant_digit)
            #check if ticks are inside parameter space and not too close to each other
            if (ticks[0]<range_of_param[0]) or (ticks[-1]>range_of_param[1]) or ((ticks[0]-range_of_param[0])>1.2*rounded_diff_range) or ((range_of_param[1]-ticks[-1])>1.2*rounded_diff_range):
                significant_digit-=1
            else:
                break
        if significant_digit == math.floor(np.log10(diff_range)) - 10*n_ticks:
            for i in range(n_ticks):
                start = center - (n_ticks-1)/2 * diff_range
                ticks[i] = np.around(start + i*diff_range, -significant_digit)
        return ticks

    for c in columns:
        if c == 'EMPTY':
            current_ranges[c] = (np.nan, np.nan)
        else:
            current_ranges[c] = (np.amin(data[c])-1e-6, np.amax(data[c])+1e-6)

    # Bins for histograms
    hist_binedges = {c: np.linspace(*current_ranges[c], num=n_bins + 1) for c in columns}
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

    def get_best_lims(new_xlims, new_ylims, old_xlims, old_ylims):
        xlims = (np.min([new_xlims[0], old_xlims[0]]) , np.max([new_xlims[1], old_xlims[1]]))
        ylims = (np.min([new_ylims[0], old_ylims[0]]) , np.max([new_ylims[1], old_ylims[1]]))
        return xlims, ylims

    # Plot histograms
    if plot_histograms_1D:
        for i in range(n_dim):
            if columns[i]!='EMPTY':
                prob1D = get_density_grid_1D(data=data[columns[i]],
                                            prob=prob,
                                            lims=current_ranges[columns[i]],
                                            binedges=hist_binedges[columns[i]],
                                            bincenters=hist_bincenters[columns[i]],
                                            method=density_estimation_method,
                                            de_kwargs=de_kwargs)
            # prob1D = histogram_1D(data=data[columns[i]], prob=prob, binedges=hist_binedges[columns[i]], bincenters=hist_bincenters[columns[i]])

                axc = get_current_ax(ax, tri, i, i)
                if axc.lines or axc.collections:
                    old_ylims = axc.get_ylim()
                    old_xlims = axc.get_xlim()
                else:
                    old_ylims = (np.inf, 0)
                    old_xlims = (np.inf, -np.inf)
                axc.autoscale()
                axc.plot(hist_bincenters[columns[i]], prob1D, '-', color=color_hist, alpha=find_alpha(columns[i], empty_columns), label=label, **hist_kwargs)
                if fill:
                    axc.fill_between(hist_bincenters[columns[i]], np.zeros_like(prob1D), prob1D, alpha=0.1*find_alpha(columns[i], empty_columns), color=color_hist)
                try:
                    xlims = ranges[columns[i]]
                except:
                    xlims = get_best_lims(current_ranges[columns[i]], current_ranges[columns[i]], old_xlims, old_ylims)[0]
                axc.set_xlim(xlims)
                axc.set_ylim(0, max(old_ylims[1], axc.get_ylim()[1]))

    if scatter_vline_1D:
        for i in range(n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, i, i)
                if np.size(data[columns[i]])>1:
                    for d in data[columns[i]]:
                        axc.axvline(d, color=color, **axvline_kwargs)
                else:
                    axc.axvline(data[columns[i]], color=color, **axvline_kwargs)
    # data
    for i, j in zip(*tri_indices):
        if columns[i]!='EMPTY' and columns[j]!='EMPTY':
            axc = get_current_ax(ax, tri, i, j)
            if axc.lines or axc.collections:
                old_ylims = axc.get_ylim()
                old_xlims = axc.get_xlim()
            else:
                old_ylims = (np.inf, -np.inf)
                old_xlims = (np.inf, -np.inf)
            if func=='contour_cl':
                contour_cl(axc, data=data, ranges=current_ranges, columns=columns,
                           i=i, j=j, fill=fill, color=color, de_kwargs=de_kwargs, line_kwargs=line_kwargs,
                           prob=prob, density_estimation_method=density_estimation_method,
                           label=label, alpha=min( (find_alpha(columns[i], empty_columns), find_alpha(columns[j], empty_columns) )),
                           )
            if func=='density_image':
                density_image(axc, data=data, ranges=current_ranges, columns=columns,
                              i=i, j=j, fill=fill, color=color, cmap=cmap, de_kwargs=de_kwargs,
                              prob=prob, density_estimation_method=density_estimation_method,
                              label=label, alpha_for_low_density=alpha_for_low_density,
                              alpha_threshold=alpha_threshold)
            elif func=='scatter':
                axc.scatter(data[columns[j]], data[columns[i]], c=color, cmap=cmap, label=label,
                            alpha=min( (find_alpha(columns[i], empty_columns), find_alpha(columns[j], empty_columns) )), **scatter_kwargs)
            elif func=='scatter_prob':
                sorting = np.argsort(prob)
                axc.scatter(data[columns[j]][sorting], data[columns[i]][sorting], c=prob[sorting], label=label, **scatter_kwargs)
            elif func=='scatter_density':
                scatter_density(axc, points1=data[columns[j]], points2=data[columns[i]], n_bins=n_bins, lim1=current_ranges[columns[j]], lim2=current_ranges[columns[i]], norm_cols=False, n_points_scatter=-1, cmap=cmap, label=label)
            current_ranges[columns[j]], current_ranges[columns[i]] = get_best_lims(current_ranges[columns[j]], current_ranges[columns[i]], old_xlims, old_ylims)
            try:
                xlims = ranges[columns[j]]
            except:
                xlims = current_ranges[columns[j]]
            try:
                ylims = ranges[columns[i]]
            except:
                ylims = current_ranges[columns[i]]
            axc.set_xlim(xlims)
            axc.set_ylim(ylims)
            axc.get_yaxis().set_major_formatter(FormatStrFormatter('%.3e'))
            axc.get_xaxis().set_major_formatter(FormatStrFormatter('%.3e'))


    # ticks
    n = n_dim-1
    # ticks = lambda i: np.linspace(ranges[columns[i]][0], ranges[columns[i]][1], 5)[1:-1]
    def get_ticks(i):
        try:
            return ticks[columns[i]]
        except:
            return current_ticks[columns[i]]

    # delete all ticks
    for axc in ax.ravel():
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axc.grid(False)

    for c in columns:
        if c not in current_ticks:
            if c=='EMPTY':
                current_ticks[c] = np.zeros(n_ticks)
            else:
                try:
                    current_ticks[c] = find_optimal_ticks((ranges[c][0], ranges[c][1]), n_ticks)
                except:
                    current_ticks[c] = find_optimal_ticks((current_ranges[c][0], current_ranges[c][1]), n_ticks)
    # ticks
    if tri[0]=='l':
        for i in range(1, n_dim): # rows
            for j in range(0, i): # columns
                if columns[i]!='EMPTY' and columns[j]!='EMPTY':
                    axc = get_current_ax(ax, tri, i, j)
                    axc.yaxis.tick_left()
                    axc.set_yticks(get_ticks(i))
                    #axc.tick_params(direction="in")
        for i in range(1, n_dim): # rows
            for j in range(0,i+1): # columns
                if columns[i]!='EMPTY' and columns[j]!='EMPTY':
                    axc = get_current_ax(ax, tri, i, j)
                    axc.xaxis.tick_bottom()
                    axc.set_xticks(get_ticks(j))
                    #axc.tick_params(direction="in")
    elif tri[0]=='u':
        for i in range(0, n_dim-1): # rows
            for j in range(i+1, n_dim): # columns
                if columns[i]!='EMPTY' and columns[j]!='EMPTY':
                    axc = get_current_ax(ax, tri, i, j)
                    axc.yaxis.tick_right()
                    axc.set_yticks(get_ticks(i))
                    #axc.tick_params(direction="in")
        for i in range(0, n_dim-1): # rows
            for j in range(0, n_dim): # columns
                if columns[i]!='EMPTY' and columns[j]!='EMPTY':
                    axc = get_current_ax(ax, tri, i, j)
                    axc.xaxis.tick_top()
                    axc.set_xticks(get_ticks(j))
                #axc.tick_params(direction="in")


    def fmt_e(x):
        return grid_kwargs['tickformat'].format(x).replace('e+0', 'e+').replace('e-0', 'e-')

    # ticklabels
    if tri[0]=='l':
        # y tick labels
        for i in range(1, n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, i, 0)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                ticklabels = [t for t in get_ticks(i)]
                axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family=grid_kwargs['font_family'])
        # x tick labels
        for i in range(0, n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, n, i)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                ticklabels = [t for t in get_ticks(i)]
                axc.set_xticklabels(ticklabels, rotation=90, fontsize=grid_kwargs['fontsize_ticklabels'], family=grid_kwargs['font_family'])
    elif tri[0]=='u':
        # y tick labels
        for i in range(0, n_dim-1):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, i, n)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family=grid_kwargs['font_family'])
        # x tick labels
        for i in range(0, n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, 0, i)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                axc.set_xticklabels(ticklabels, rotation=90, fontsize=grid_kwargs['fontsize_ticklabels'], family=grid_kwargs['font_family'])


    # grid
    if tri[0]=='l':
        for i in range(1,n_dim):
            for j in range(i):
                if columns[i]!='EMPTY' and columns[j]!='EMPTY':
                    axc = get_current_ax(ax, tri, i, j)
                    axc.grid(grid)
    elif tri[0]=='u':
        for i in range(0,n_dim-1):
            for j in range(i+1,n_dim):
                if columns[i]!='EMPTY' and columns[j]!='EMPTY':
                    axc = get_current_ax(ax, tri, i, j)
                    axc.grid(grid)

    # Axes labels
    if labels is None:
        labels = columns
    else:
        try:
            ind = 0
            for g in grouping_indices:
                ind += g
                labels = np.insert(labels, ind, 'EMPTY')
                ind += 1
        except:
            pass

    legend_lines, legend_labels = ax[0,0].get_legend_handles_labels()
    if tri[0]=='l':
        labelpad = 10
        for i in range(n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, i, 0)
                axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
                axc.yaxis.set_label_position("left")
                axc = get_current_ax(ax, tri, n, i)
                axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
                axc.xaxis.set_label_position("bottom")
        if legend_lines and show_legend: #only print legend when there are labels for it
            fig.legend(legend_lines, legend_labels, bbox_to_anchor=(1, 1), bbox_transform=ax[0,n_dim-1].transAxes, fontsize=label_fontsize)
    elif tri[0]=='u':
        labelpad = 20
        for i in range(n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, i, n)
                axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
                axc.yaxis.set_label_position("right")
                axc = get_current_ax(ax, tri, 0, i)
                axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
                axc.xaxis.set_label_position("top")
        if legend_lines and show_legend: #only print legend when there are labels for it
            fig.get_legend().remove()
            fig.legend(legend_lines, legend_labels, bbox_to_anchor=(1, 1), bbox_transform=ax[n_dim-1,0].transAxes, fontsize=label_fontsize)



    plt.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels()
    fig.align_xlabels()

    for axc in ax.flatten():
        for c in axc.collections:
            if isinstance(c, mpl.collections.QuadMesh):
                #rasterize density images to avoid ugly aliasing when saving as a pdf
                c.set_rasterized(True)

    return fig
