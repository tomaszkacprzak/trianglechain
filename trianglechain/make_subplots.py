import matplotlib.pyplot as plt
import numpy as np
from trianglechain.density_estimation import (
    get_density_grid_1D,
    get_density_grid_2D,
    get_confidence_levels,
)
from trianglechain.utils_plots import (
    get_old_lims,
    find_alpha,
    get_values,
    get_best_lims,
)
from matplotlib.colors import ListedColormap
import matplotlib
import warnings


def plot_1d(
    axc,
    column,
    data,
    prob,
    ranges,
    current_ranges,
    hist_binedges,
    hist_bincenters,
    de_kwargs,
    hist_kwargs,
    empty_columns,
    show_values=False,
    label=None,
    density_estimation_method="smoothing",
    color_hist="k",
    alpha1D=1,
    fill=False,
    lnprobs=None,
    levels_method="hdi",
    bestfit_method="mean",
    credible_interval=0.68,
    label_fontsize=12,
):
    prob1D = get_density_grid_1D(
        data=data[column],
        prob=prob,
        lims=current_ranges[column],
        binedges=hist_binedges[column],
        bincenters=hist_bincenters[column],
        method=density_estimation_method,
        de_kwargs=de_kwargs,
    )

    old_xlims, old_ylims = get_old_lims(axc)
    axc.autoscale()
    axc.plot(
        hist_bincenters[column],
        prob1D,
        "-",
        color=color_hist,
        alpha=find_alpha(column, empty_columns, alpha1D),
        label=label,
        **hist_kwargs,
    )
    if fill:
        axc.fill_between(
            hist_bincenters[column],
            np.zeros_like(prob1D),
            prob1D,
            alpha=0.1 * find_alpha(column, empty_columns, alpha1D),
            color=color_hist,
        )
    try:
        xlims = ranges[column]
    except Exception:
        xlims = get_best_lims(
            current_ranges[column],
            current_ranges[column],
            old_xlims,
            old_ylims,
        )[0]
    axc.set_xlim(xlims)
    axc.set_ylim(0, max(old_ylims[1], axc.get_ylim()[1]))
    if show_values:
        add_values(
            axc,
            column,
            data,
            lnprobs,
            label=label,
            levels_method=levels_method,
            bestfit_method=bestfit_method,
            credible_interval=credible_interval,
            label_fontsize=label_fontsize,
        )


def add_values(
    axc,
    column,
    data,
    lnprobs,
    label,
    levels_method="hdi",
    bestfit_method="mean",
    credible_interval=0.68,
    label_fontsize=12,
):
    str_bf, up, low = get_values(
        column,
        data,
        lnprobs,
        levels_method=levels_method,
        bestfit_method=bestfit_method,
        credible_interval=credible_interval,
    )
    axc.set_title(
        r"{} $= {}^{{+{} }}_{{-{} }}$".format(label, str_bf, up, low),
        fontsize=label_fontsize,
    )


def density_image(
    axc,
    data,
    ranges,
    columns,
    i,
    j,
    fill,
    color,
    cmap,
    de_kwargs,
    vmin=0,
    vmax=None,
    prob=None,
    density_estimation_method="smoothing",
    label=None,
    alpha_for_low_density=False,
    alpha_threshold=0,
    color_bar=True,
):
    """
    axc - axis of the plot
    data - numpy struct array with column data
    ranges - dict of ranges for each column in data
    columns - list of columns
    i, j - pair of columns to plot
    fill - use filled contour
    color - color for the contour
    de_kwargs - dict with kde settings,
                has to have n_points, n_levels_check, levels, defaults below
    prob - if not None, then probability attached to the samples,
           in that case samples are treated as grid not a chain
    """
    kde, x_grid, y_grid = get_density_grid_2D(
        data=data,
        ranges=ranges,
        columns=columns,
        i=i,
        j=j,
        de_kwargs=de_kwargs,
        prob=prob,
        method=density_estimation_method,
    )
    if alpha_for_low_density:
        cmap_plt = plt.get_cmap(cmap)
        my_cmap = cmap_plt(np.arange(cmap_plt.N))
        cmap_threshold = int(cmap_plt.N * alpha_threshold)
        my_cmap[:cmap_threshold, -1] = np.linspace(0, 1, cmap_threshold)
        cmap = ListedColormap(my_cmap)
    axc.pcolormesh(
        x_grid,
        y_grid,
        kde,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        label=label,
        shading="auto",
    )


def contour_cl(
    axc,
    data,
    ranges,
    columns,
    i,
    j,
    fill,
    color,
    de_kwargs,
    line_kwargs,
    prob=None,
    density_estimation_method="smoothing",
    label=None,
    alpha=1,
):
    """
    axc - axis of the plot
    data - numpy struct array with column data
    ranges - dict of ranges for each column in data
    columns - list of columns
    i, j - pair of columns to plot
    fill - use filled contour
    color - color for the contour
    de_kwargs - dict with kde settings, has to have
                n_points, n_levels_check, levels, defaults below
    prob - if not None, then probability attached to the samples,
           in that case samples are treated as grid not a chain
    """

    def _get_paler_colors(color_rgb, n_levels, pale_factor=None):

        solid_contour_palefactor = 0.6

        # convert a color into an array of colors for used in contours
        color = matplotlib.colors.colorConverter.to_rgb(color_rgb)
        pale_factor = pale_factor or solid_contour_palefactor
        cols = [color]
        for _ in range(1, n_levels):
            cols = [
                [c * (1 - pale_factor) + pale_factor for c in cols[0]]
            ] + cols
        return cols

    de, x_grid, y_grid = get_density_grid_2D(
        i=i,
        j=j,
        data=data,
        prob=prob,
        ranges=ranges,
        columns=columns,
        method=density_estimation_method,
        de_kwargs=de_kwargs,
    )

    levels_contour = get_confidence_levels(
        de=de,
        levels=de_kwargs["levels"],
        n_levels_check=de_kwargs["n_levels_check"],
    )

    with warnings.catch_warnings():
        # this will suppress all warnings in this block
        warnings.simplefilter("ignore")
        colors = _get_paler_colors(
            color_rgb=color, n_levels=len(de_kwargs["levels"])
        )

        for l_i, lvl in enumerate(levels_contour):
            if fill:
                axc.contourf(
                    x_grid,
                    y_grid,
                    de,
                    levels=[lvl, np.inf],
                    colors=[colors[l_i]],
                    label=label,
                    alpha=0.85 * alpha,
                    **line_kwargs,
                )
                # axc.contour(
                #     x_grid,
                #     y_grid,
                #     de,
                #     levels=[lvl, np.inf],
                #     colors=[colors[l_i]],
                #     alpha=0.5*alpha,
                #     **line_kwargs
                # )
            else:
                axc.contour(
                    x_grid,
                    y_grid,
                    de,
                    levels=[lvl, np.inf],
                    colors=color,
                    alpha=alpha,
                    label=label,
                    **line_kwargs,
                )


def scatter_density(
    axc,
    points1,
    points2,
    n_bins=50,
    lim1=None,
    lim2=None,
    norm_cols=False,
    n_points_scatter=-1,
    label=None,
    **kwargs,
):

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

    bins_edges1 = np.linspace(min1, max1, n_bins)
    bins_edges2 = np.linspace(min2, max2, n_bins)

    hv, bv, _ = np.histogram2d(
        points1, points2, bins=[bins_edges1, bins_edges2]
    )

    if norm_cols is True:
        hv = hv / np.sum(hv, axis=0)[:, np.newaxis]

    bins_centers1 = (bins_edges1 - (bins_edges1[1] - bins_edges1[0]) / 2)[1:]
    bins_centers2 = (bins_edges2 - (bins_edges2[1] - bins_edges2[0]) / 2)[1:]

    from scipy.interpolate import griddata

    select_box = (
        (points1 < max1)
        & (points1 > min1)
        & (points2 < max2)
        & (points2 > min2)
    )
    points1_box, points2_box = points1[select_box], points2[select_box]

    x1, x2 = np.meshgrid(bins_centers1, bins_centers2)
    points = np.concatenate(
        [x1.flatten()[:, np.newaxis], x2.flatten()[:, np.newaxis]], axis=1
    )
    xi = np.concatenate(
        [points1_box[:, np.newaxis], points2_box[:, np.newaxis]], axis=1
    )

    if lim1 is not None:
        axc.set_xlim(lim1)
    if lim2 is not None:
        axc.set_ylim(lim2)

    if n_points_scatter > 0:
        select = np.random.choice(len(points1_box), n_points_scatter)
        c = griddata(
            points,
            hv.T.flatten(),
            xi[select, :],
            method="linear",
            rescale=True,
            fill_value=np.min(hv),
        )
        axc.scatter(
            points1_box[select],
            points2_box[select],
            c=c,
            label=label,
            **kwargs,
        )
    else:
        c = griddata(
            points,
            hv.T.flatten(),
            xi,
            method="linear",
            rescale=True,
            fill_value=np.min(hv),
        )
        sorting = np.argsort(c)
        axc.scatter(
            points1_box[sorting],
            points2_box[sorting],
            c=c[sorting],
            label=label,
            **kwargs,
        )
