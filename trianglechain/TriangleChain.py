import pylab as plt, numpy as np, scipy, warnings, math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from functools import partial
from scipy.stats import median_absolute_deviation
from sklearn.preprocessing import MinMaxScaler
from trianglechain.utils_plots import *
from trianglechain.BaseChain import BaseChain

class TriangleChain(BaseChain):

    def __init__(self, fig=None, size=4, **kwargs):

        super().__init__(fig=fig, size=size, **kwargs)

        self.add_plotting_functions(self.add_plot)

    def add_plot(self, data, plottype, prob=None, color='b', cmap=plt.cm.plasma, tri='lower', plot_histograms_1D=True):
        
        self.fig = plot_triangle_maringals(fig=self.fig, 
                                           size=self.size,
                                           func=plottype,
                                           cmap=cmap,
                                           data=data,
                                           prob=prob,
                                           tri=tri,
                                           color=color,
                                           plot_histograms_1D=plot_histograms_1D, 
                                           **self.kwargs)
        return self.fig

def plot_triangle_maringals(data, prob=None, func='contour_cl', tri='lower',
                            single_tri=True, color='b', cmap=plt.cm.plasma,
                            ranges={}, ticks={}, n_bins=20, fig=None, size=4,
                            fill=True, grid=False, labels=None, plot_histograms_1D=True,
                            density_estimation_method='smoothing', n_ticks=3,
                            alpha_for_low_density=False, alpha_threshold=0,
                            subplots_kwargs={}, de_kwargs={}, hist_kwargs={}, axes_kwargs={},
                            labels_kwargs={}, grid_kwargs={}, scatter_kwargs={}, grouping_kwargs={}):
    data = ensure_rec(data)

    columns = data.dtype.names


    try:
        grouping_indices = np.cumsum(np.asarray(grouping_kwargs['n_per_group']))[:-1]
        columns = np.insert(columns, grouping_indices + 1, 'EMPTY')
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
            return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) - 1)
        except:
            return number

    def find_optimal_ticks(range_of_param, n_ticks = 3):
        diff = range_of_param[1]-range_of_param[0]
        ticks = np.zeros(n_ticks)

        # mathematical center and tick interval
        diff_range = diff/(n_ticks+1)
        center = range_of_param[0] + diff/2

        # nicely rounded tick interval
        rounded_diff_range = round_to_significant_digits(diff_range, 1)
        if abs(rounded_diff_range-diff_range)/diff_range > 0.199:
            rounded_diff_range = round_to_significant_digits(diff_range, 2)

        # decimal until which ticks are rounded
        decimal_to_round = math.floor(np.log10(rounded_diff_range))
        if n_ticks&2==0:
            decimal_to_round-=1

        # nicely rounded center value
        rounded_center = np.around(center, -decimal_to_round)

        start = rounded_center - (n_ticks-1)/2 * rounded_diff_range
        for i in range(n_ticks):
            ticks[i] = np.around(start + i*rounded_diff_range, -decimal_to_round)
        return ticks

    eps = 1e-6
    for c in columns:
        if c not in ranges:
            ranges[c] = (np.nan, np.nan) if c == 'EMPTY' else (np.amin(data[c])-eps, np.amax(data[c])+eps)
        if c not in ticks:
            ticks[c] = np.zeros(n_ticks) if c == 'EMPTY' else find_optimal_ticks((ranges[c][0], ranges[c][1]), n_ticks)
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
            if columns[i]!='EMPTY':
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
                axc.set_ylim(bottom=0)


    # data
    for i, j in zip(*tri_indices):
        if columns[i]!='EMPTY' and columns[j]!='EMPTY':
            axc = get_current_ax(ax, tri, i, j)

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
        axc.grid(grid)


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
                #ticklabels = [fmt_e(t) for t in get_ticks(i)]
                ticklabels = [t for t in get_ticks(i)]
                axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family=grid_kwargs['font_family'])
        # x tick labels
        for i in range(0, n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, n, i)
                #ticklabels = [fmt_e(t) for t in get_ticks(i)]
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
        if len(labels.keys()) == 0:
            labels = columns
        else:   
            try:
                labels = np.insert(labels, grouping_indices + 1, 'EMPTY')
            except:
                pass


    if tri[0]=='l':
        labelpad = 10
        for i in range(n_dim):
            if columns[i]!='EMPTY':
                axc = get_current_ax(ax, tri, i, 0)

                try:
                    axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
                except:
                    import ipdb; ipdb.set_trace()
                axc.yaxis.set_label_position("left")
                axc = get_current_ax(ax, tri, n, i)
                axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
                axc.xaxis.set_label_position("bottom")
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



    plt.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels()
    fig.align_xlabels()

    return fig

