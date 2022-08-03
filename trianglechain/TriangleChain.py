import pylab as plt
import numpy as np
import math
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from trianglechain.utils_plots import (
    ensure_rec,
    get_density_grid_1D,
    find_alpha,
    get_best_lims,
    round_to_significant_digits,
    contour_cl,
    density_image,
    scatter_density,
    find_optimal_ticks
)
from trianglechain.BaseChain import BaseChain
from trianglechain import limits, bestfit
from tqdm import tqdm, trange

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

        self.fig, self.ax = plot_triangle_maringals(
            fig=self.fig,
            size=self.size,
            func=plottype,
            cmap=cmap,
            data=data,
            prob=prob,
            tri=tri,
            color=color,
            **kwargs_copy,
        )
        return self.fig, self.ax


def plot_triangle_maringals(
    data,
    prob=None,
    params="all",
    func="contour_cl",
    tri="lower",
    single_tri=True,
    color="b",
    cmap=plt.cm.plasma,
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
):
    data = ensure_rec(data)
    empty_columns = []
    if add_empty_plots_like is not None:
        columns = data.dtype.names
        data2 = ensure_rec(add_empty_plots_like)
        columns2 = data2.dtype.names
        new_data = np.zeros(len(data), dtype=data2.dtype)
        for c in columns2:
            if c in columns:
                new_data[c] = data[c]
            else:
                new_data[c] = data2[c][
                    np.random.randint(0, len(data2), len(data))
                ]
                empty_columns.append(c)
        data = new_data
    if params != "all":
        data = data[params]
    columns = data.dtype.names

    # needed for plotting chains with different automatic limits
    current_ranges = {}
    current_ticks = {}

    try:
        grouping_indices = np.asarray(grouping_kwargs["n_per_group"])[:-1]
        ind = 0
        for g in grouping_indices:
            ind += g
            columns = np.insert(np.array(columns, dtype="<U32"), ind, "EMPTY")
            ind += 1
    except Exception:
        pass

    # Axes labels
    if labels is None:
        labels = columns
    else:
        try:
            ind = 0
            for g in grouping_indices:
                ind += g
                labels = np.insert(labels, ind, "EMPTY")
                ind += 1
        except Exception:
            pass

    hw_ratios = np.ones_like(columns, dtype=float)
    for i, label in enumerate(columns):
        if label == "EMPTY":
            hw_ratios[i] = grouping_kwargs["empty_ratio"]

    n_dim = len(columns)
    if single_tri:
        n_box = n_dim
    else:
        n_box = n_dim + 1

    if prob is not None:
        prob = prob / np.sum(prob)

    if tri[0] == "l":
        tri_indices = np.tril_indices(n_dim, k=-1)
    elif tri[0] == "u":
        tri_indices = np.triu_indices(n_dim, k=1)
    else:
        raise Exception("tri={} should be either l or u".format(tri))

    # Create figure if necessary and get axes
    if fig is None:
        fig, _ = plt.subplots(
            nrows=n_box,
            ncols=n_box,
            figsize=(sum(hw_ratios) * size, sum(hw_ratios) * size),
            gridspec_kw={
                "height_ratios": hw_ratios,
                "width_ratios": hw_ratios,
            },
            **subplots_kwargs,
        )
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)
        for axc in ax.ravel():
            axc.axis("off")
    else:
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)

    eps = 1e-6
    for c in columns:
        if c not in ranges:
            current_ranges[c] = (
                (np.nan, np.nan)
                if c == "EMPTY"
                else (np.amin(data[c]) - eps, np.amax(data[c]) + eps)
            )
        else:
            current_ranges[c] = ranges[c]

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
        axc.axis("on")
        return axc

    # Plot histograms
    if not plot_histograms_1D:
        for i in range(n_dim):
            axc = get_current_ax(ax, tri, i, i)
            if not axc.lines or not axc.collections:
                axc.set_visible(False)
    else:
        disable_progress_bar = True
        if show_values:
            LOGGER.info("Computing bestfits and levels")
            disable_progress_bar = False
        for i in trange(n_dim, disable=disable_progress_bar):
            if columns[i] != "EMPTY":
                prob1D = get_density_grid_1D(
                    data=data[columns[i]],
                    prob=prob,
                    lims=current_ranges[columns[i]],
                    binedges=hist_binedges[columns[i]],
                    bincenters=hist_bincenters[columns[i]],
                    method=density_estimation_method,
                    de_kwargs=de_kwargs,
                )
                # prob1D = histogram_1D(data=data[columns[i]],
                #                       prob=prob,
                #                       binedges=hist_binedges[columns[i]],
                #                       bincenters=hist_bincenters[columns[i]])

                axc = get_current_ax(ax, tri, i, i)
                if axc.lines or axc.collections:
                    old_ylims = axc.get_ylim()
                    old_xlims = axc.get_xlim()
                else:
                    old_ylims = (np.inf, 0)
                    old_xlims = (np.inf, -np.inf)
                axc.autoscale()
                axc.plot(
                    hist_bincenters[columns[i]],
                    prob1D,
                    "-",
                    color=color_hist,
                    alpha=find_alpha(columns[i], empty_columns),
                    label=label,
                    **hist_kwargs,
                )
                if fill:
                    axc.fill_between(
                        hist_bincenters[columns[i]],
                        np.zeros_like(prob1D),
                        prob1D,
                        alpha=0.1 * find_alpha(columns[i], empty_columns),
                        color=color_hist,
                    )
                try:
                    xlims = ranges[columns[i]]
                except Exception:
                    xlims = get_best_lims(
                        current_ranges[columns[i]],
                        current_ranges[columns[i]],
                        old_xlims,
                        old_ylims,
                    )[0]
                axc.set_xlim(xlims)
                axc.set_ylim(0, max(old_ylims[1], axc.get_ylim()[1]))
                if show_values:
                    lower, upper = limits.get_levels(
                        data[columns[i]],
                        lnprobs,
                        levels_method,
                        credible_interval,
                    )
                    bf = bestfit.get_bestfit(
                        data[columns[i]], lnprobs, bestfit_method
                    )
                    uncertainty = (upper - lower) / 2
                    first_significant_digit = math.floor(np.log10(uncertainty))
                    u = round_to_significant_digits(uncertainty, 3) * 10 ** (
                        -first_significant_digit + 2
                    )
                    if u > 100 and u < 354:
                        significant_digits_to_round = 2
                    elif u < 949:
                        significant_digits_to_round = 1
                    else:
                        significant_digits_to_round = 2
                        uncertainty = 1000 / 10 ** (
                            -first_significant_digit + 2
                        )
                    rounding_digit = -(
                        math.floor(np.log10(uncertainty))
                        - significant_digits_to_round
                        + 1
                    )
                    if rounding_digit > 0:
                        frmt = "{{:.{}f}}".format(rounding_digit)
                    else:
                        frmt = "{:.0f}"
                    str_bf = f"{frmt}".format(np.around(bf, rounding_digit))
                    low = f"{frmt}".format(
                        np.around(bf - lower, rounding_digit)
                    )
                    up = f"{frmt}".format(
                        np.around(upper - bf, rounding_digit)
                    )
                    axc.set_title(
                        r"{} $= {}^{{+{} }}_{{-{} }}$".format(
                            labels[i], str_bf, up, low
                        ),
                        fontsize=label_fontsize,
                    )

    if scatter_vline_1D:
        for i in range(n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, i)
                if np.size(data[columns[i]]) > 1:
                    for d in data[columns[i]]:
                        axc.axvline(d, color=color, **axvline_kwargs)
                else:
                    axc.axvline(
                        data[columns[i]], color=color, **axvline_kwargs
                    )
    # data
    for i, j in tqdm(zip(*tri_indices), total=len(tri_indices[0])):
        if columns[i] != "EMPTY" and columns[j] != "EMPTY":
            axc = get_current_ax(ax, tri, i, j)
            if axc.lines or axc.collections:
                old_ylims = axc.get_ylim()
                old_xlims = axc.get_xlim()
            else:
                old_ylims = (np.inf, -np.inf)
                old_xlims = (np.inf, -np.inf)
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
                            find_alpha(columns[i], empty_columns),
                            find_alpha(columns[j], empty_columns),
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
                            find_alpha(columns[i], empty_columns),
                            find_alpha(columns[j], empty_columns),
                        )
                    ),
                    **scatter_kwargs,
                )
            elif func == "scatter_prob":
                sorting = np.argsort(prob)
                axc.scatter(
                    data[columns[j]][sorting],
                    data[columns[i]][sorting],
                    c=prob[sorting],
                    label=label,
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
            (
                current_ranges[columns[j]],
                current_ranges[columns[i]],
            ) = get_best_lims(
                current_ranges[columns[j]],
                current_ranges[columns[i]],
                old_xlims,
                old_ylims,
            )
            try:
                xlims = ranges[columns[j]]
            except Exception:
                xlims = current_ranges[columns[j]]
            try:
                ylims = ranges[columns[i]]
            except Exception:
                ylims = current_ranges[columns[i]]
            axc.set_xlim(xlims)
            axc.set_ylim(ylims)
            axc.get_yaxis().set_major_formatter(FormatStrFormatter("%.3e"))
            axc.get_xaxis().set_major_formatter(FormatStrFormatter("%.3e"))

    # ticks
    n = n_dim - 1
    # ticks = lambda i: np.linspace(
    #     ranges[columns[i]][0],
    #     ranges[columns[i]][1],
    #     5
    # )[1:-1]

    def get_ticks(i):
        try:
            return ticks[columns[i]]
        except Exception:
            return current_ticks[columns[i]]

    # delete all ticks
    for axc in ax.ravel():
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axc.set_axisbelow(True)

    for c in columns:
        if c not in current_ticks:
            if c == "EMPTY":
                current_ticks[c] = np.zeros(n_ticks)
            else:
                try:
                    current_ticks[c] = find_optimal_ticks(
                        (ranges[c][0], ranges[c][1]), n_ticks
                    )
                except Exception:
                    current_ticks[c] = find_optimal_ticks(
                        (current_ranges[c][0], current_ranges[c][1]), n_ticks
                    )
    # ticks
    if tri[0] == "l":
        for i in range(1, n_dim):  # rows
            for j in range(0, i):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, tri, i, j)
                    axc.yaxis.tick_left()
                    axc.set_yticks(get_ticks(i))
                    axc.tick_params(direction="in")
        for i in range(1, n_dim):  # rows
            for j in range(0, i + 1):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, tri, i, j)
                    axc.xaxis.tick_bottom()
                    axc.set_xticks(get_ticks(j))
                    axc.tick_params(direction="in")
    elif tri[0] == "u":
        for i in range(0, n_dim - 1):  # rows
            for j in range(i + 1, n_dim):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, tri, i, j)
                    axc.yaxis.tick_right()
                    axc.set_yticks(get_ticks(i))
                    axc.tick_params(direction="in")
        for i in range(0, n_dim - 1):  # rows
            for j in range(0, n_dim):  # columns
                if columns[i] != "EMPTY" and columns[j] != "EMPTY":
                    axc = get_current_ax(ax, tri, i, j)
                    axc.xaxis.tick_top()
                    axc.set_xticks(get_ticks(j))
                    axc.tick_params(direction="in")

    def fmt_e(x):
        return (
            grid_kwargs["tickformat"]
            .format(x)
            .replace("e+0", "e+")
            .replace("e-0", "e-")
        )

    # ticklabels
    if tri[0] == "l":
        # y tick labels
        for i in range(1, n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, 0)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                ticklabels = [t for t in get_ticks(i)]
                axc.set_yticklabels(
                    ticklabels,
                    rotation=0,
                    fontsize=grid_kwargs["fontsize_ticklabels"],
                    family=grid_kwargs["font_family"],
                )
        # x tick labels
        for i in range(0, n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, n, i)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                ticklabels = [t for t in get_ticks(i)]
                axc.set_xticklabels(
                    ticklabels,
                    rotation=90,
                    fontsize=grid_kwargs["fontsize_ticklabels"],
                    family=grid_kwargs["font_family"],
                )
    elif tri[0] == "u":
        # y tick labels
        for i in range(0, n_dim - 1):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, n)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                axc.set_yticklabels(
                    ticklabels,
                    rotation=0,
                    fontsize=grid_kwargs["fontsize_ticklabels"],
                    family=grid_kwargs["font_family"],
                )
        # x tick labels
        for i in range(0, n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, 0, i)
                ticklabels = [fmt_e(t) for t in get_ticks(i)]
                axc.set_xticklabels(
                    ticklabels,
                    rotation=90,
                    fontsize=grid_kwargs["fontsize_ticklabels"],
                    family=grid_kwargs["font_family"],
                )

    # grid
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

    legend_lines, legend_labels = ax[0, 0].get_legend_handles_labels()
    if tri[0] == "l":
        labelpad = 10
        for i in range(n_dim):
            if columns[i] != "EMPTY":
                axc = get_current_ax(ax, tri, i, 0)

                try:
                    axc.set_ylabel(
                        labels[i],
                        **labels_kwargs,
                        rotation=90,
                        labelpad=labelpad,
                    )
                except Exception:
                    import ipdb
                    ipdb.set_trace()
                axc.yaxis.set_label_position("left")
                axc = get_current_ax(ax, tri, n, i)
                axc.set_xlabel(
                    labels[i], **labels_kwargs, rotation=0, labelpad=labelpad
                )
                axc.xaxis.set_label_position("bottom")
        if (
            legend_lines and show_legend
        ):  # only print legend when there are labels for it
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
                axc = get_current_ax(ax, tri, i, n)
                axc.set_ylabel(
                    labels[i], **labels_kwargs, rotation=90, labelpad=labelpad
                )
                axc.yaxis.set_label_position("right")
                axc = get_current_ax(ax, tri, 0, i)
                axc.set_xlabel(
                    labels[i], **labels_kwargs, rotation=0, labelpad=labelpad
                )
                axc.xaxis.set_label_position("top")
        if (
            legend_lines and show_legend
        ):  # only print legend when there are labels for it
            fig.get_legend().remove()
            fig.legend(
                legend_lines,
                legend_labels,
                bbox_to_anchor=(1, 1),
                bbox_transform=ax[n_dim - 1, 0].transAxes,
                fontsize=label_fontsize,
            )

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
