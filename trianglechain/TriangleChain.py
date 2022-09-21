import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from trianglechain.utils_plots import (
    prepare_columns,
    setup_grouping,
    get_labels,
    get_hw_ratios,
    setup_figure,
    update_current_ranges,
    update_current_ticks,
    set_limits,
    delete_all_ticks,
    add_vline,
    ensure_rec,
    get_old_lims,
    get_best_old_lims,
    find_alpha,
)
from trianglechain.make_subplots import (
    contour_cl,
    density_image,
    scatter_density,
    plot_1d,
)
from trianglechain.BaseChain import BaseChain
from tqdm.auto import tqdm, trange

from ekit import logger as logger_utils

LOGGER = logger_utils.init_logger(__name__)


class TriangleChain(BaseChain):
    def __init__(self, fig=None, size=4, **kwargs):

        super().__init__(fig=fig, size=size, **kwargs)

        self.add_plotting_functions(self.add_plot)

    def add_plot(
        self,
        data,
        plottype,
        prob=None,
        color="b",
        cmap=plt.cm.plasma,
        tri="lower",
        plot_histograms_1D=True,
        **kwargs,
    ):

        from copy import deepcopy

        kwargs_copy = deepcopy(self.kwargs)
        kwargs_copy.update(kwargs)

        self.fig, self.ax = plot_triangle_marginals(
            fig=self.fig,
            size=self.size,
            func=plottype,
            cmap=cmap,
            data=data,
            prob=prob,
            tri=tri,
            color=color,
            plot_histograms_1D=plot_histograms_1D,
            **kwargs_copy,
        )
        return self.fig, self.ax


def plot_triangle_marginals(
    data,
    prob=None,
    params="all",
    func="contour_cl",
    tri="lower",
    single_tri=True,
    color="b",
    cmap=plt.cm.plasma,
    cmap_vmin=0,
    cmap_vmax=None,
    ranges={},
    ticks={},
    n_bins=20,
    fig=None,
    size=4,
    fill=True,
    grid=False,
    labels=None,
    plot_histograms_1D=True,
    show_values=False,
    bestfit_method="mode",
    levels_method="hdi",
    credible_interval=0.68,
    lnprobs=None,
    scatter_vline_1D=False,
    label=None,
    density_estimation_method="smoothing",
    n_ticks=3,
    alpha1D=1,
    alpha2D=1,
    alpha_for_low_density=False,
    alpha_threshold=0,
    label_levels1D=0.68,
    subplots_kwargs={},
    de_kwargs={},
    hist_kwargs={},
    axes_kwargs={},
    line_kwargs={},
    labels_kwargs={},
    grid_kwargs={},
    scatter_kwargs={},
    grouping_kwargs={},
    axvline_kwargs={},
    add_empty_plots_like=None,
    label_fontsize=12,
    show_legend=False,
    orientation=None,
    colorbar=False,
    colorbar_label=None,
    colorbar_ax=[0.735, 0.5, 0.03, 0.25],
    normalize_prob=True,
):

    ###############################
    # prepare data and setup plot #
    ###############################
    data = ensure_rec(data)
    data, columns, empty_columns = prepare_columns(
        data, params=params, add_empty_plots_like=add_empty_plots_like
    )

    # needed for plotting chains with different automatic limits
    current_ranges = {}
    current_ticks = {}

    # setup everything that grouping works properly
    columns, grouping_indices = setup_grouping(columns, grouping_kwargs)
    labels = get_labels(labels, columns, grouping_indices)
    hw_ratios = get_hw_ratios(columns, grouping_kwargs)

    n_dim = len(columns)
    if single_tri:
        n_box = n_dim
    else:
        n_box = n_dim + 1

    if prob is not None:
        if normalize_prob:
            prob = prob / np.sum(prob)
        else:
            # for example to plot an additional parameter in parameter space
            prob_label = prob
            prob = None

    if tri[0] == "l":
        tri_indices = np.tril_indices(n_dim, k=-1)
    elif tri[0] == "u":
        tri_indices = np.triu_indices(n_dim, k=1)
    else:
        raise Exception("tri={} should be either lower or upper".format(tri))

    # Create figure if necessary and get axes
    fig, ax, old_tri = setup_figure(
        fig, n_box, hw_ratios, size, colorbar, subplots_kwargs
    )
    if old_tri is not None and old_tri != tri:
        double_tri = True
    else:
        double_tri = False

    # get ranges for each parameter (if not specified, max/min of data is used)
    update_current_ranges(current_ranges, ranges, columns, data)

    # Bins for histograms
    hist_binedges = {
        c: np.linspace(*current_ranges[c], num=n_bins + 1) for c in columns
    }
    hist_bincenters = {
        c: (hist_binedges[c][1:] + hist_binedges[c][:-1]) / 2 for c in columns
    }

    if len(color) == len(data):
        color_hist = "k"
    else:
        color_hist = color

    def get_current_ax(ax, tri, i, j):

        if tri[0] == "u":
            if single_tri:
                axc = ax[i, j]
            else:
                axc = ax[i, j + 1]
        elif tri[0] == "l":
            if single_tri:
                axc = ax[i, j]
            else:
                axc = ax[i + 1, j]

        if i==j and not plot_histograms_1D and not scatter_vline_1D:
            # in this case axis should not be turned on in this call
            pass
        else:
            # turn on ax sinces it is used
            axc.axis("on")
        return axc

    #################
    # 1D histograms #
    #################
    if plot_histograms_1D:
        disable_progress_bar = True
        if show_values:
            LOGGER.info("Computing bestfits and levels")
            disable_progress_bar = False
        for i in trange(n_dim, disable=disable_progress_bar):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, i)
                plot_1d(
                    axc,
                    column=columns[i],
                    param_label=labels[i],
                    data=data,
                    prob=prob,
                    ranges=ranges,
                    current_ranges=current_ranges,
                    hist_binedges=hist_binedges,
                    hist_bincenters=hist_bincenters,
                    density_estimation_method=density_estimation_method,
                    de_kwargs=de_kwargs,
                    show_values=show_values,
                    color_hist=color_hist,
                    empty_columns=empty_columns,
                    alpha1D=alpha1D,
                    label=label,
                    hist_kwargs=hist_kwargs,
                    fill=fill,
                    lnprobs=lnprobs,
                    levels_method=levels_method,
                    bestfit_method=bestfit_method,
                    credible_interval=credible_interval,
                    label_fontsize=label_fontsize,
                )

    if scatter_vline_1D:
        for i in range(n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, i)
                add_vline(axc, columns[i], data, color, axvline_kwargs)

    #################
    # 2D histograms #
    #################
    for i, j in tqdm(zip(*tri_indices), total=len(tri_indices[0])):
        if columns[i] != "EMPTY" and columns[j] != "EMPTY":
            axc = get_current_ax(ax, tri, i, j)
            old_xlims, old_ylims = get_old_lims(axc)

            if double_tri:
                if tri=="lower":
                    other_tri = "upper"
                else:
                    other_tri = "lower"
                axc_mirror = get_current_ax(ax, other_tri, j, i)
                old_xlims_mirror, old_ylims_mirror = get_old_lims(axc_mirror)
                old_xlims, old_ylims = get_best_old_lims(
                    old_xlims, old_ylims_mirror, old_ylims, old_xlims_mirror
                )

            if func == "contour_cl":
                contour_cl(
                    axc,
                    data=data,
                    ranges=current_ranges,
                    columns=columns,
                    i=i,
                    j=j,
                    fill=fill,
                    color=color,
                    de_kwargs=de_kwargs,
                    line_kwargs=line_kwargs,
                    prob=prob,
                    density_estimation_method=density_estimation_method,
                    label=label,
                    alpha=min(
                        (
                            find_alpha(columns[i], empty_columns, alpha2D),
                            find_alpha(columns[j], empty_columns, alpha2D),
                        )
                    ),
                )
            if func == "density_image":
                density_image(
                    axc,
                    data=data,
                    ranges=current_ranges,
                    columns=columns,
                    i=i,
                    j=j,
                    fill=fill,
                    color=color,
                    cmap=cmap,
                    de_kwargs=de_kwargs,
                    vmin=cmap_vmin,
                    vmax=cmap_vmax,
                    prob=prob,
                    density_estimation_method=density_estimation_method,
                    label=label,
                    alpha_for_low_density=alpha_for_low_density,
                    alpha_threshold=alpha_threshold,
                )
            elif func == "scatter":
                axc.scatter(
                    data[columns[j]],
                    data[columns[i]],
                    c=color,
                    cmap=cmap,
                    label=label,
                    alpha=min(
                        (
                            find_alpha(columns[i], empty_columns, alpha2D),
                            find_alpha(columns[j], empty_columns, alpha2D),
                        )
                    ),
                    **scatter_kwargs,
                )
            elif func == "scatter_prob":
                if normalize_prob:
                    _prob = prob
                else:
                    _prob = prob_label
                sorting = np.argsort(_prob)
                axc.scatter(
                    data[columns[j]][sorting],
                    data[columns[i]][sorting],
                    c=_prob[sorting],
                    label=label,
                    cmap=cmap,
                    **scatter_kwargs,
                )
            elif func == "scatter_density":
                scatter_density(
                    axc,
                    points1=data[columns[j]],
                    points2=data[columns[i]],
                    n_bins=n_bins,
                    lim1=current_ranges[columns[j]],
                    lim2=current_ranges[columns[i]],
                    norm_cols=False,
                    n_points_scatter=-1,
                    cmap=cmap,
                    label=label,
                )
            set_limits(
                axc,
                ranges,
                current_ranges,
                columns[i],
                columns[j],
                old_xlims,
                old_ylims,
            )
            if double_tri:
                set_limits(
                    axc_mirror,
                    ranges,
                    current_ranges,
                    columns[j],
                    columns[i],
                    old_ylims,
                    old_xlims,
                )

    #########
    # ticks #
    #########
    def get_ticks(i):
        try:
            return ticks[columns[i]]
        except Exception:
            return current_ticks[columns[i]]

    def plot_yticks(axc, i, length=10, direction="in"):
        axc.yaxis.set_ticks_position("both")
        axc.set_yticks(get_ticks(i))
        axc.tick_params(direction=direction, length=length)

    def plot_xticks(axc, i, j, length=10, direction="in"):
        if i != j:
            axc.xaxis.set_ticks_position("both")
        axc.set_xticks(get_ticks(j))
        axc.tick_params(direction=direction, length=length)

    delete_all_ticks(ax)
    update_current_ticks(
        current_ticks, columns, ranges, current_ranges, n_ticks
    )

    if tri[0] == "l" or double_tri:
        local_tri = "lower"
        for i in range(1, n_dim):  # rows
            for j in range(0, i):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, local_tri, i, j)
                    plot_yticks(axc, i)

        for i in range(0, n_dim):  # rows
            for j in range(0, i + 1):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, local_tri, i, j)
                    plot_xticks(axc, i, j)
    if tri[0] == "u" or double_tri:
        local_tri="upper"
        for i in range(0, n_dim):  # rows
            for j in range(i + 1, n_dim):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, local_tri, i, j)
                    plot_yticks(axc, i)
        for i in range(0, n_dim):  # rows
            for j in range(i, n_dim):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, local_tri, i, j)
                    plot_xticks(axc, i, j)

    def fmt_e(x):
        return (
            grid_kwargs["tickformat"]
            .format(x)
            .replace("e+0", "e+")
            .replace("e-0", "e-")
        )

    # ticklabels
    def plot_tick_labels(axc, xy, i, tri, grid_kwargs):
        # ticklabels = [fmt_e(t) for t in get_ticks(i)]
        ticklabels = [t for t in get_ticks(i)]
        if xy == "y":
            axc.set_yticklabels(
                ticklabels,
                rotation=0,
                fontsize=grid_kwargs["fontsize_ticklabels"],
                family=grid_kwargs["font_family"],
            )
            if tri[0] == "u":
                axc.yaxis.tick_right()
                axc.yaxis.set_ticks_position("both")
                axc.yaxis.set_label_position("right")
        elif xy == "x":
            axc.set_xticklabels(
                ticklabels,
                rotation=90,
                fontsize=grid_kwargs["fontsize_ticklabels"],
                family=grid_kwargs["font_family"],
            )
            if tri[0] == "u":
                axc.xaxis.tick_top()
                axc.xaxis.set_ticks_position("both")
                axc.xaxis.set_label_position("top")

    if tri[0] == "l" or double_tri:
        local_tri="lower"
        # y tick labels
        for i in range(1, n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, local_tri, i, 0)
                plot_tick_labels(axc, "y", i, local_tri, grid_kwargs)
        # x tick labels
        for i in range(0, n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, local_tri, n_dim - 1, i)
                plot_tick_labels(axc, "x", i, local_tri, grid_kwargs)
    if tri[0] == "u" or double_tri:
        local_tri="upper"
        # y tick labels
        for i in range(0, n_dim - 1):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, local_tri, i, n_dim - 1)
                plot_tick_labels(axc, "y", i, tri, grid_kwargs)
        # x tick labels
        for i in range(1, n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, local_tri, 0, i)
                plot_tick_labels(axc, "x", i, tri, grid_kwargs)

    ########
    # grid #
    ########
    if tri[0] == "l":
        for i in range(1, n_dim):
            for j in range(i):
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, tri, i, j)
                    if grid:
                        axc.grid(zorder=0, linestyle="--")
                    axc.set_axisbelow(True)
    elif tri[0] == "u":
        for i in range(0, n_dim - 1):
            for j in range(i + 1, n_dim):
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, tri, i, j)
                    if grid:
                        axc.grid(zorder=0, linestyle="--")
                    axc.set_axisbelow(True)

    ###########
    # legends #
    ###########
    legend_lines, legend_labels = ax[0, 0].get_legend_handles_labels()
    if tri[0] == "l":
        labelpad = 10
        for i in range(n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, 0)

                axc.set_ylabel(
                    labels[i],
                    **labels_kwargs,
                    rotation=90,
                    labelpad=labelpad,
                )
                axc.yaxis.set_label_position("left")
                axc = get_current_ax(ax, tri, n_dim - 1, i)
                axc.set_xlabel(
                    labels[i], **labels_kwargs, rotation=0, labelpad=labelpad
                )
                axc.xaxis.set_label_position("bottom")
        if legend_lines and show_legend:
            # only print legend when there are labels for it
            fig.legend(
                legend_lines,
                legend_labels,
                bbox_to_anchor=(1, 1),
                bbox_transform=ax[0, n_dim - 1].transAxes,
                fontsize=label_fontsize,
            )
    elif tri[0] == "u":
        labelpad = 20
        for i in range(n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, n_dim - 1)
                axc.set_ylabel(
                    labels[i], **labels_kwargs, rotation=90, labelpad=labelpad
                )
                axc.yaxis.set_label_position("right")
                axc = get_current_ax(ax, tri, 0, i)
                axc.set_xlabel(
                    labels[i], **labels_kwargs, rotation=0, labelpad=labelpad
                )
                axc.xaxis.set_label_position("top")
        if legend_lines and show_legend:
            # only print legend when there are labels for it
            fig.legend(
                legend_lines,
                legend_labels,
                bbox_to_anchor=(1, 1),
                bbox_transform=ax[n_dim - 1, 0].transAxes,
                fontsize=label_fontsize,
            )

    if colorbar:
        try:
            cmap_vmin = min(prob_label)
            cmap_vmax = max(prob_label)
        except Exception:
            pass
        norm = mpl.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # ticks = np.linspace(amin, amax, 3)
        cbar = fig.colorbar(sm, cax=fig.add_axes(colorbar_ax))
        cbar.ax.tick_params(labelsize=grid_kwargs["fontsize_ticklabels"])
        cbar.set_label(colorbar_label, fontsize=label_fontsize)

    plt.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels()
    fig.align_xlabels()

    for axc in ax.flatten():
        for c in axc.collections:
            if isinstance(c, mpl.collections.QuadMesh):
                # rasterize density images to avoid ugly aliasing
                # when saving as a pdf
                c.set_rasterized(True)

    return fig, ax
