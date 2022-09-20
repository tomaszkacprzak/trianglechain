from functools import partial


class BaseChain:
    def __init__(self, fig=None, size=4, **kwargs):
        """
        :param fig: matplotlib figure to use
        :param size: figures size for a new figure, for a single panel.
        All panels are square
        :param labels: labels for the parameters, default taken from columns
        :param ranges: dictionary with ranges for parameters, keys correspond
        to column names
        :param ticks: values for axis ticks, defaults taken from range with
        3 equally spaced values
        :param n_bins: number of bins for 1d histograms, default: 100
        :param fill: if to fill contours
        :param density_estimation_method: method for density estimation,
            options:
                smoothing: first create a histogram of samples, and then smooth
                it with a Gaussian kernel corresponding to the variance of the
                10% of the smallest eigenvalue of the 2D distribution
                gaussian_mixture: use Gaussian mixture to fit the 2D samples
                median_filter: use median filter on the 2D histogram
                kde: use TreeKDE, may be slow
                hist: simple 2D histogram
        :param de_kwargs: density estimation kwargs, dictionary with keys:
            n_points: number of bins for 2d histograms used to create
            contours, etc, default: n_bins
            levels: density levels for contours, the contours will enclose
            this level of probability, default: [0.68, 0.95]
            n_levels_check: number of levels to check when looking for density
            levels, default: 1000. More levels is more accurate, but slower
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
        tri.contour_cl(samples)  # plot contours at given confidence levels
        tri.density_image(samples)  # plot PDF density image
        tri.scatter(samples)  # simple scatter plot
        tri.scatter_prob(samples)  # scatter plot
                                   # with probability for each sample provided
        tri.scatter_density(samples) # scatter plot
                                     # color corresponds to probability



        """

        kwargs.setdefault("ticks", {})
        kwargs.setdefault("ranges", {})
        kwargs.setdefault("labels", None)
        kwargs.setdefault("n_bins", 100)
        kwargs.setdefault("de_kwargs", {})
        kwargs.setdefault("grid_kwargs", {})
        kwargs.setdefault("hist_kwargs", {})
        kwargs.setdefault("labels_kwargs", {})
        kwargs.setdefault("line_kwargs", {})
        kwargs.setdefault("axvline_kwargs", {})
        kwargs.setdefault("density_estimation_method", "smoothing")
        kwargs.setdefault("alpha_for_low_density", False)
        kwargs.setdefault("alpha_threshold", 0)
        kwargs.setdefault("n_ticks", 3)
        kwargs.setdefault("fill", False)
        kwargs.setdefault("grid", False)
        kwargs.setdefault("scatter_kwargs", {})
        kwargs.setdefault("grouping_kwargs", {})
        kwargs.setdefault("add_empty_plots_like", None)
        kwargs.setdefault("label_fontsize", 24)
        kwargs.setdefault("params", "all")
        kwargs.setdefault("label_levels1D", 0.68)
        kwargs.setdefault("orientation", "horizontal")
        kwargs.setdefault("colorbar", False)
        kwargs["de_kwargs"].setdefault("n_points", kwargs["n_bins"])
        kwargs["de_kwargs"].setdefault("levels", [0.68, 0.95])
        kwargs["de_kwargs"].setdefault("n_levels_check", 2000)
        kwargs["de_kwargs"].setdefault("smoothing_sigma", None)
        kwargs["de_kwargs"].setdefault("smoothing_parameter1D", 0.1)
        kwargs["de_kwargs"].setdefault("smoothing_parameter2D", 0.2)
        kwargs["de_kwargs"]["levels"].sort()
        if kwargs["fill"]:
            kwargs["line_kwargs"].setdefault("linewidths", 2)
        else:
            kwargs["line_kwargs"].setdefault("linewidths", 4)
        kwargs["grid_kwargs"].setdefault("fontsize_ticklabels", 14)
        kwargs["grid_kwargs"].setdefault("tickformat", "{: 0.2e}")
        kwargs["grid_kwargs"].setdefault("font_family", "sans-serif")
        kwargs["hist_kwargs"].setdefault("lw", 4)
        kwargs["labels_kwargs"].setdefault("fontsize", 24)
        kwargs["labels_kwargs"].setdefault("family", "sans-serif")
        kwargs["grouping_kwargs"].setdefault("n_per_group", None)
        kwargs["grouping_kwargs"].setdefault("empty_ratio", 0.2)

        self.fig = fig
        self.size = size
        self.kwargs = kwargs
        self.funcs = [
            "contour_cl",
            "density_image",
            "scatter",
            "scatter_prob",
            "scatter_density",
        ]

    def add_plotting_functions(self, func_add_plot):

        for fname in self.funcs:
            f = partial(func_add_plot, plottype=fname)
            setattr(self, fname, f)
