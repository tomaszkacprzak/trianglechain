import numpy as np
import math
from scipy.stats import median_absolute_deviation
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from trianglechain import limits, bestfit


def prepare_columns(data, params="all", add_empty_plots_like=None):
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
    return data, columns, empty_columns


def setup_grouping(columns, grouping_kwargs):
    try:
        grouping_indices = np.asarray(grouping_kwargs["n_per_group"])[:-1]
        ind = 0
        for g in grouping_indices:
            ind += g
            columns = np.insert(np.array(columns, dtype="<U32"), ind, "EMPTY")
            ind += 1
        return columns, grouping_indices
    except Exception:
        return columns, None


def get_labels(labels, columns, grouping_indices):
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
    return labels


def get_hw_ratios(columns, grouping_kwargs):
    hw_ratios = np.ones_like(columns, dtype=float)
    for i, lab in enumerate(columns):
        if lab == "EMPTY":
            hw_ratios[i] = grouping_kwargs["empty_ratio"]
    return hw_ratios


def setup_figure(fig, n_box, hw_ratios, size, colorbar, subplots_kwargs):
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
            # remove all unused axs
            axc.axis("off")
    else:
        ax = np.array(fig.get_axes())
        if colorbar:
            ax = ax[:-1]
        ax = ax.reshape(n_box, n_box)
    return fig, ax


def update_current_ranges(current_ranges, ranges, columns, data):
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


def update_current_ticks(
    current_ticks, columns, ranges, current_ranges, n_ticks
):
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


def get_old_lims(axc):
    if axc.lines or axc.collections:
        old_ylims = axc.get_ylim()
        old_xlims = axc.get_xlim()
    else:
        old_ylims = (np.inf, 0)
        old_xlims = (np.inf, -np.inf)
    return old_xlims, old_ylims


def get_values(
    column,
    data,
    lnprobs,
    levels_method="hdi",
    bestfit_method="mean",
    credible_interval=0.68,
):
    lower, upper = limits.get_levels(
        data[column],
        lnprobs,
        levels_method,
        credible_interval,
    )
    bf = bestfit.get_bestfit(data[column], lnprobs, bestfit_method)
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
    uncertainty = 1000 / 10 ** (-first_significant_digit + 2)
    rounding_digit = -(
        math.floor(np.log10(uncertainty)) - significant_digits_to_round + 1
    )
    if rounding_digit > 0:
        frmt = "{{:.{}f}}".format(rounding_digit)
    else:
        frmt = "{:.0f}"
    str_bf = f"{frmt}".format(np.around(bf, rounding_digit))
    low = f"{frmt}".format(np.around(bf - lower, rounding_digit))
    up = f"{frmt}".format(np.around(upper - bf, rounding_digit))
    return str_bf, up, low


def safe_normalise(p):

    # fix to use arrays
    if np.sum(p) != 0:
        p = p / np.sum(p)
    return p


def delete_all_ticks(ax):
    for axc in ax.ravel():
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axc.set_axisbelow(True)


def round_to_significant_digits(number, significant_digits):
    try:
        return round(
            number,
            significant_digits - int(math.floor(math.log10(abs(number)))) - 1,
        )
    except Exception:
        return number


def find_optimal_ticks(range_of_param, n_ticks=3):
    diff = range_of_param[1] - range_of_param[0]
    ticks = np.zeros(n_ticks)

    # mathematical center and tick interval
    diff_range = diff / (n_ticks + 1)
    center = range_of_param[0] + diff / 2

    # first significant digit for rounding
    significant_digit = math.floor(np.log10(diff_range))

    for i in range(10 * n_ticks):
        rounded_center = np.around(center, -significant_digit + i)
        if abs(rounded_center - center) / diff < 0.05:
            break
    for i in range(10 * n_ticks):
        rounded_diff_range = np.around(diff_range, -significant_digit)
        start = rounded_center - (n_ticks - 1) / 2 * rounded_diff_range
        for i in range(n_ticks):
            if n_ticks % 2 == 0:
                ticks[i] = np.around(
                    start + i * rounded_diff_range, -significant_digit + 1
                )
            else:
                ticks[i] = np.around(
                    start + i * rounded_diff_range, -significant_digit
                )
        # check if ticks are inside parameter space and
        # not too close to each other
        if (
            (ticks[0] < range_of_param[0])
            or (ticks[-1] > range_of_param[1])
            or ((ticks[0] - range_of_param[0]) > 1.2 * rounded_diff_range)
            or ((range_of_param[1] - ticks[-1]) > 1.2 * rounded_diff_range)
        ):
            significant_digit -= 1
        else:
            break
    if significant_digit == math.floor(np.log10(diff_range)) - 10 * n_ticks:
        for i in range(n_ticks):
            start = center - (n_ticks - 1) / 2 * diff_range
            ticks[i] = np.around(start + i * diff_range, -significant_digit)
    return ticks


def get_best_lims(new_xlims, new_ylims, old_xlims, old_ylims):
    xlims = (
        np.min([new_xlims[0], old_xlims[0]]),
        np.max([new_xlims[1], old_xlims[1]]),
    )
    ylims = (
        np.min([new_ylims[0], old_ylims[0]]),
        np.max([new_ylims[1], old_ylims[1]]),
    )
    return xlims, ylims


def add_vline(axc, column, data, color, axvline_kwargs):
    if np.size(data[column]) > 1:
        for d in data[column]:
            axc.axvline(d, color=color, **axvline_kwargs)
    else:
        axc.axvline(data[column], color=color, **axvline_kwargs)


def set_limits(axc, ranges, current_ranges, col1, col2, old_xlims, old_ylims):
    current_ranges[col2], current_ranges[col1] = get_best_lims(
        current_ranges[col2],
        current_ranges[col1],
        old_xlims,
        old_ylims,
    )
    try:
        xlims = ranges[col2]
    except Exception:
        xlims = current_ranges[col2]
    try:
        ylims = ranges[col1]
    except Exception:
        ylims = current_ranges[col1]
    axc.set_xlim(xlims)
    axc.set_ylim(ylims)
    axc.get_yaxis().set_major_formatter(FormatStrFormatter("%.3e"))
    axc.get_xaxis().set_major_formatter(FormatStrFormatter("%.3e"))


def pixel_coords(x, ranges, n_pix_img):
    xt = np.atleast_2d(x.copy())
    for i in range(xt.shape[0]):
        try:
            xt[i] -= ranges[i][0]
            xt[i] /= ranges[i][1] - ranges[i][0]
        except Exception:
            import ipdb

            ipdb.set_trace()
    return xt * n_pix_img


def get_smoothing_sigma(x, max_points=5000):

    x = np.atleast_2d(x)
    from sklearn.decomposition import PCA

    if x.shape[0] == 2:
        pca = PCA()
        pca.fit(x.T)
        sig_pix = np.sqrt(pca.explained_variance_[-1])
    elif x.shape[0] == 1:
        mad = median_absolute_deviation(x, axis=1)
        sig_pix = np.min(mad)

    return sig_pix


def ensure_rec(data, column_prefix=""):

    if data.dtype.names is not None:
        return data

    else:

        n_rows, n_cols = data.shape
        dtype = np.dtype(
            dict(
                formats=[data.dtype] * n_cols,
                names=[f"{column_prefix}{i}" for i in range(n_cols)],
            )
        )
        rec = np.empty(n_rows, dtype=dtype)
        for i in range(n_cols):
            rec[f"{column_prefix}{i}"] = data[:, i]
        return rec


def find_alpha(column, empty_columns, alpha=1):
    if column in empty_columns:
        return 0
    else:
        return alpha
