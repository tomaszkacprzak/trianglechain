import pylab as plt, numpy as np, scipy, warnings, math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from functools import partial
from scipy.stats import median_absolute_deviation
from sklearn.preprocessing import MinMaxScaler
import matplotlib


def safe_normalise(p):

    # fix to use arrays
    if np.sum(p)!=0:
        p = p/np.sum(p)
    return p

# round to get nicer ticks
def round_to_significant_digits(number, significant_digits):
    try:
        return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) - 1)
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

def get_best_lims(new_xlims, new_ylims, old_xlims, old_ylims):
    xlims = (np.min([new_xlims[0], old_xlims[0]]) , np.max([new_xlims[1], old_xlims[1]]))
    ylims = (np.min([new_ylims[0], old_ylims[0]]) , np.max([new_ylims[1], old_ylims[1]]))
    return xlims, ylims

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

    if np.sum(prob2d)>0:
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
        prob1D[hist_counts==0]

    prob1D = safe_normalise(prob1D)
    prob1D = np.ma.array(prob1D, mask=prob1D==0)
    return prob1D


def get_density_grid_1D(data, binedges, bincenters, lims, prob=None, method='smoothing', de_kwargs={}):

    if method=='gaussian_mixture':

        if prob is not None:
            ind = np.random.choice(a=len(prob), size=10000, p=safe_normalise(prob), replace=True)
            data_ = data[ind][:,np.newaxis]
        else:
            data_ = data[:,np.newaxis]

        from trianglechain.TransformedGaussianMixture import TransformedGaussianMixture
        from sklearn.mixture import GaussianMixture
        clf = TransformedGaussianMixture(param_bounds=[lims], n_components=20, covariance_type='full')
        clf.fit(data_)
        logL = clf.score_samples(bincenters[:,np.newaxis])

        de = np.exp(logL - np.max(logL))
        de = safe_normalise(de)

    elif method=='smoothing':
        from scipy import signal
        from scipy.ndimage import convolve
        from scipy import ndimage

        prob1D = histogram_1D(data=data, prob=prob, binedges=binedges)
        data_pixel = pixel_coords(data, [[binedges[0], binedges[-1]]], n_pix_img=len(bincenters))
        if prob is not None:
            ind = np.random.choice(prob1D.shape[0], p=prob1D, size=1000)
            data_pixel = data_pixel[0,ind]

        sig_pix = get_smoothing_sigma(data_pixel, prob1D.shape[0])*de_kwargs['smoothing_parameter1D']
        n_pix = max(3,int(np.ceil(sig_pix*10))) # 10 sigma smoothing
        kernel = signal.gaussian(n_pix, sig_pix)
        de = scipy.ndimage.convolve(prob1D, kernel, mode='reflect')
        de = safe_normalise(de)

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
            ind = np.random.choice(a=len(prob), size=10000, p=safe_normalise(prob), replace=True)
            data_panel = data_panel[:,ind]

        bounds = [ranges[columns[j]], ranges[columns[i]]]

        from trianglechain.TransformedGaussianMixture import TransformedGaussianMixture
        from sklearn.mixture import GaussianMixture
        clf = TransformedGaussianMixture(param_bounds=bounds, n_components=10, covariance_type='full')
        clf.fit(data_panel.T)
        logL = clf.score_samples(gridpoints.T)

        de = np.exp(logL - np.max(logL)).reshape(x_grid.shape)
        de = safe_normalise(de)

    elif method=='smoothing':

        prob2d = histogram_2D(data_panel, prob, bins_x, bins_y)
        prob2d = safe_normalise(prob2d)
        from scipy import signal
        data_panel_pixel = pixel_coords(data_panel, [ranges[columns[j]], ranges[columns[i]]], n_pix_img=de_kwargs['n_points'])
        if prob is not None:
            ids = np.random.choice(a=len(prob), p=prob, size=1000, replace=True)
            data_panel_pixel = data_panel_pixel[:,ids]

        if de_kwargs['smoothing_sigma'] is None:
            sig_pix = get_smoothing_sigma(data_panel_pixel)*de_kwargs['smoothing_parameter2D']
            n_pix= int(np.ceil(sig_pix*5))
        else:
            sig_pix = de_kwargs['smoothing_sigma']

        from scipy.ndimage import gaussian_filter
        de = gaussian_filter(prob2d, sigma=sig_pix)


    elif method=='median_filter':

        from scipy import ndimage
        prob2d = histogram_2D(data_panel, prob, bins_x, bins_y)
        prob2d = safe_normalise(prob2d)
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
    de = safe_normalise(de)
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
    axc.pcolormesh(x_grid, y_grid, kde, cmap=cmap, vmin=0, label=label, shading='auto')

def get_confidence_levels(de, levels, n_levels_check):

    lvl_max = 0.99
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

    def _get_paler_colors(color_rgb, n_levels, pale_factor=None):

        solid_contour_palefactor = 0.6

        # convert a color into an array of colors for used in contours
        color = matplotlib.colors.colorConverter.to_rgb(color_rgb)
        pale_factor = pale_factor or solid_contour_palefactor
        cols = [color]
        for _ in range(1, n_levels):
            cols = [[c * (1 - pale_factor) + pale_factor for c in cols[0]]] + cols
        return cols


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
        colors = _get_paler_colors(color_rgb=color, n_levels=len(de_kwargs['levels']))

        for l, lvl in enumerate(levels_contour):
            if fill:
                axc.contourf(x_grid, y_grid, de, levels=[lvl, np.inf], colors=[colors[l]], label=label, alpha=0.85*alpha, **line_kwargs)
                #axc.contour(x_grid, y_grid, de, levels=[lvl, np.inf], colors=[colors[l]], alpha=0.5*alpha, **line_kwargs)
            else:
                axc.contour(x_grid, y_grid, de, levels=[lvl, np.inf], colors=color, alpha=alpha, label=label, **line_kwargs)

def scatter_density(axc, points1, points2, n_bins=50, lim1=None, lim2=None, norm_cols=False, n_points_scatter=-1, colorbar=False, label=None, **kwargs):

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
        sc = axc.scatter(points1_box[sorting], points2_box[sorting], c=c[sorting], label=label,  **kwargs)

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

def find_alpha(column, empty_columns):
    if column in empty_columns:
        return 0
    else:
        return 1
