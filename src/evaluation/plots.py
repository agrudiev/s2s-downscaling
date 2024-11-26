import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from utils import get_pdf, get_cdf
from engineering.spectrum import get_1dpsd

# Define the contour levels and colors
CUSTOM_PRECIP_COLORS = [
    '#FFFFFF',  # White
    '#FFFFCC',  # Light Yellow
    '#C7E9B4',  # Light Green
    '#7FCDBB',  # Moderate Blue-green
    '#41B6C4',  # Moderate Blue
    '#1D91C0',  # Blue
    '#225EA8',  # Darker Blue
    '#253494',  # Dark Blue
    '#54278F',  # Purple
    '#7A0177',  # Dark
    '#C51B8A'   # Pink
]
PRECIP_CMAP = mcolors.ListedColormap(CUSTOM_PRECIP_COLORS)
CUSTOM_VALUES = [0, 0.1, 0.5, 1, 2, 2.5, 5, 10, 20, 30, 50]
CUSTOM_NORM = mcolors.BoundaryNorm(CUSTOM_VALUES, 11)
CUSTOM_CURVE_COLORS = [
    '#E69F00', # Orange
    '#D55E00', # Vermilion
    '#0072B2', # Blue
    '#009E73', # Bluish Green
    '#999999', # Gray
    '#56B4E9', # Sky Blue
    '#CC79A7', # Reddish Purple
    '#F0E442', # Yellow
    '#007D65', # Teal
    '#FF00FF', # Magenta
]
CURVE_CMAP = mcolors.ListedColormap(CUSTOM_CURVE_COLORS)


def _add_swiss_latlon_labels(ax, plons, plats):
    if plons is not None:
        ax.set_xlabel('Longitude')
        ax.set_xticks(plons)
        ax.set_xticklabels([str(c)+"º" for c in plons])
    if plats is not None:
        ax.set_ylabel('Latitude')
        ax.set_yticks(plats)
        ax.set_yticklabels([str(c)+"º" for c in plats])
        
        
def _add_swiss_xy_labels(ax, xp, yp):
    if xp is not None:
        ax.set_xlabel(r'Swiss-X ($10^5$m)')
        ax.set_xticks(xp)
        ax.set_xticklabels([str(int(c/10**5)) for c in xp])
    if yp is not None:
        ax.set_ylabel(r'Swiss-Y ($10^5$m)')
        ax.set_yticks(yp)
        ax.set_yticklabels([str(int(c/10**5)) for c in yp])
        
        
def _write_label(ax, label):
    plons = None
    plats = None
    xp = None
    yp = None
    
    # TODO: generalize this for any region in the world (not only Switzerland)
    if label[0] == "lon":
        plons = [6, 8, 10]
    elif label[0] == "x":
        xp = [2550000.0, 2750000.0]
    if label[1] == "lat":
        plats = [46, 47.5]
    elif label[1] == "y":
        yp = [1110000.0, 1250000.0]
    
    if label[0] == "lon" or label[1] == "lat":
        _add_swiss_latlon_labels(ax, plons=plons, plats=plats)
    
    elif label[0] == "x" or label[1] == "y":
        _add_swiss_xy_labels(ax, xp=xp, yp=yp)


def _plot_2maps(
    arrays, 
    titles, 
    extents, 
    projections,
    cmap,
    norm,
    vmin,
    vmax,
    cbar_label,
    figsize=(8, 2.3),
):
    fig = plt.figure(figsize=figsize, dpi=500)
    axes = [None, None]
    
    axis_labels = (
        ("lon", "lat"), 
        ("lon", None)
    )
    
    fig, axs = plt.subplots(
        1, 2,
        figsize=figsize,
        dpi=300,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            norm=norm,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        ax.set_frame_on(False)
        _write_label(ax, axis_labels[i])
    
    fig.subplots_adjust(
        left=0.1,
        right=0.85,
        bottom=0.03,
        top=1.03,
        wspace=0.1,
    )
    
    cbar_ax = fig.add_axes([0.89, 0.1, 0.02, 1-2*0.1])
    fig.colorbar(img, cax=cbar_ax, label=cbar_label)

    return fig, axes


def _plot_3maps(
    arrays, 
    titles, 
    extents, 
    projections,
    cmap,
    norm,
    vmin,
    vmax,
    cbar_label,
    figsize=(11, 2.3),
):
    fig = plt.figure(figsize=figsize, dpi=500)
    axes = [None, None]
    
    axis_labels = (
        ("lon", "lat"),
        ("lon", None),
        ("lon", None),
    )
    
    fig, axs = plt.subplots(
        1, 3,
        figsize=figsize,
        dpi=300,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            norm=norm,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        ax.set_frame_on(False)
        _write_label(ax, axis_labels[i])
    
    fig.subplots_adjust(
        left=0.1,
        right=0.85,
        bottom=0.03,
        top=1.03,
        wspace=0.1,
    )
    
    cbar_ax = fig.add_axes([0.89, 0.1, 0.02, 1-2*0.1])
    fig.colorbar(img, cax=cbar_ax, label=cbar_label)

    return fig, axes


def _plot_4maps(
    arrays, 
    titles, 
    extents, 
    projections,
    cmap,
    norm,
    vmin,
    vmax,
    cbar_label,
    figsize=(8, 4),
):
    fig, axs = plt.subplots(
        2, 2,
        figsize=figsize,
        dpi=300,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    
    axis_labels = (
        (None, "lat"),
        (None, None),
        ("lon", "lat"),
        ("lon", None)
    )

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            norm=norm,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        ax.set_frame_on(False)
        _write_label(ax, axis_labels[i])

    fig.subplots_adjust(
        left=0.1,
        right=0.85,
        bottom=0.03,
        top=1.03,
        wspace=0.1,
        hspace=-0.2,
    )
    
    # Create a single axis for the colorbar
    cbar_ax = fig.add_axes([0.89, 0.5-0.5/2, 0.02, 1/2])
    fig.colorbar(img, cax=cbar_ax, label=cbar_label)

    return fig, axs


def _plot_6maps(
    arrays, 
    titles, 
    extents, 
    projections,
    cmap,
    norm,
    vmin,
    vmax,
    cbar_label,
    figsize=(11, 4),
):
    fig, axs = plt.subplots(
        2, 3,
        figsize=figsize,
        dpi=300,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    
    axis_labels = (
        (None, "lat"),
        (None, None),
        (None, None),
        ("lon", "lat"),
        ("lon", None),
        ("lon", None),
    )

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        ax.set_frame_on(False)
        _write_label(ax, axis_labels[i])

    fig.subplots_adjust(
        left=0.1,
        right=0.85,
        bottom=0.03,
        top=1.03,
        wspace=0.1,
        hspace=-0.2,
    )
    
    # Create a single axis for the colorbar
    cbar_ax = fig.add_axes([0.89, 0.5-0.5/2, 0.02, 1/2])
    fig.colorbar(img, cax=cbar_ax, label=cbar_label)

    return fig, axs


def _plot_9maps(
    arrays,
    extents, 
    projections,
    cmap,
    norm,
    vmin,
    vmax,
    cbar_label,
    figsize=(11, 11),
):
    fig, axs = plt.subplots(
        3, 3,
        figsize=figsize,
        dpi=300,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    
    axis_labels = (
        (None, "lat"),
        (None, None),
        (None, None),
        (None, "lat"),
        (None, None),
        (None, None),
        ("lon", "lat"),
        ("lon", None),
        ("lon", None),
    )

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_frame_on(False)
        _write_label(ax, axis_labels[i])

    fig.subplots_adjust(
        left=0.1,
        right=0.85,
        bottom=0.03,
        top=1.03,
        wspace=0.1,
        hspace=-0.3,
    )
    
    # Create a single axis for the colorbar
    cbar_ax = fig.add_axes([0.89, 0.5-0.5/3, 0.02, 1/3])
    fig.colorbar(img, cax=cbar_ax, label=cbar_label)

    return fig, axs


def plot_maps(
    arrays,
    titles,
    extents,
    projections=None,
    cmap=PRECIP_CMAP,
    norm=CUSTOM_NORM,
    vmin=None,
    vmax=None,
    cbar_label='Precipitation (mm/h)',
):
    """
    General function to plot any even number of maps.

    ## Parameters:
    - arrays (tuple): tuple of data arrays. Should be an even number of elements.
    - titles (tuple): tuple of titles for the maps. Should match the length of arrays.
    - extents (tuple): tuple of extents for the maps. Should match the length of arrays.
    - projections (tuple, optional): tuple of projections for the maps. Should match the length of arrays. Defaults to PlateCarree.
    - cmap (matplotlib colormap, optional): colormap for the maps.
    - norm (matplotlib.colors.Normalize, optional): normalization for colormap.
    - vmin (int, optional): minimum value for colormap. Default is None. Use it if norm is None.
    - vmax (int, optional): maximum value for colormap. Default is None. Use it if norm is None.
    - cbar_label (str, optional): label for the colorbar. Default is 'Precipitation (mm/h)'.

    ## Returns:
    - fig (matplotlib figure object)
    - axs (matplotlib axes object)
    """
    nplots = len(arrays)
    projections = projections or (ccrs.PlateCarree(),) * nplots
    
    if nplots == 2:
        fig, axes = _plot_2maps(
            arrays,
            titles,
            extents,
            projections,
            cmap,
            norm,
            vmin,
            vmax,
            cbar_label,
        )
    elif nplots == 3:
        fig, axes = _plot_3maps(
            arrays,
            titles,
            extents,
            projections,
            cmap,
            norm,
            vmin,
            vmax,
            cbar_label,
        )
    elif nplots == 4:
        fig, axes = _plot_4maps(
            arrays,
            titles,
            extents,
            projections,
            cmap,
            norm,
            vmin,
            vmax,
            cbar_label,
        )
    elif nplots == 6:
        fig, axes = _plot_6maps(
            arrays,
            titles,
            extents,
            projections,
            cmap,
            norm,
            vmin,
            vmax,
            cbar_label,
        )
    elif nplots == 9:
        fig, axes = _plot_9maps(
            arrays,
            extents,
            projections,
            cmap,
            norm,
            vmin,
            vmax,
            cbar_label,
        )
    else:
        raise ValueError("Only 2, 3, 4, 6 or 9 maps are supported.")
    
    return fig, axes


def plot_cdfs(
    arrays,
    labels,
    n_quantiles=100,
    colors=None,
    ls=None,
    cmap=CURVE_CMAP,
    xlim_max=60,
):
    """
    Plot the cumulative distribution functions (CDFs) of multiple arrays.

    ## Parameters:
    - arrays (tuple of arrays): tuple of arrays to plot CDFs for.
    - labels (tuple of str): tuple of labels for the arrays. Should match the length of arrays.
    - n_quantiles (int, optional): number of quantiles to divide the data into. Default is 100.
    - colors (tuple of str, optional): tuple of colors for each plot. If provided, should match the length of arrays.
    - ls (tuple of str, optional): tuple of linestyles for each plot. If provided, should match the length of arrays.
    - cmap (str, optional): colormap to use for plotting. Default is a custom colormap.
    - xlim_max (float, optional): maximum value for the x-axis. Default is 60.

    ## Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object)
    """
    if len(arrays) != len(labels):
        raise ValueError("The number of arrays and labels must be the same.")
    
    # Colors
    if colors is not None:
        if len(colors) != len(arrays):
            raise ValueError("The number of colors must match the number of data arrays if colors are provided.")
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(arrays))]
        
    # Linestyle
    if ls is not None:
        if len(ls) != len(arrays):
            raise ValueError("The number of linestyles must match the number of data arrays if linestyles are provided.")
    else:
        ls = ('-',) * len(arrays)
    
    cmap = plt.get_cmap(cmap)
    global_max = max(np.nanmax(arr) for arr in arrays)
    global_min = max(min(np.nanmin(arr) for arr in arrays), 0)
    
    wide = abs(global_max - global_min) / n_quantiles
    bins = np.arange(global_min, global_max + wide, wide)

    cdfs = [get_cdf(arr, bins) for arr in arrays]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (cdf, label) in enumerate(zip(cdfs, labels)):
        ax.plot(bins, cdf, label=label, color=colors[i], ls=ls[i])
    
    ax.set_frame_on(False)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.set_xlabel('Precipitation (mm/h)')
    ax.set_ylabel('Cumulative distribution function')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='both', color='white')
    ax.get_yaxis().set_tick_params(which='both', color='white')
    ax.legend(fontsize='large')
    plt.xlim(0, xlim_max)
    plt.tight_layout()
    
    return fig, ax


def plot_pdfs(arrays, labels, n_quantiles=100, colors=None, cmap=CURVE_CMAP):
    """
    Plot the probability density functions (PDFs) of multiple arrays.

    ## Parameters:
    - arrays (tuple of arrays): tuple of arrays to plot PDFs for.
    - labels (tuple of str): tuple of labels for the arrays. Should match the length of arrays.
    - n_quantiles (int, optional): number of quantiles to divide the data into. Default is 100.
    - colors (tuple of str, optional): tuple of colors for each plot. If provided, should match the length of arrays.
    - ls (tuple of str, optional): tuple of linestyles for each plot. If provided, should match the length of arrays.
    - cmap (str, optional): colormap to use for plotting (default: 'Dark2')

    ## Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object)
    """
    if len(arrays) != len(labels):
        raise ValueError("The number of arrays and labels must be the same.")
    
    # Colors
    if colors is not None:
        if len(colors) != len(arrays):
            raise ValueError("The number of colors must match the number of data arrays if colors are provided.")
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(arrays))]
        
    # Linestyle
    if ls is not None:
        if len(ls) != len(arrays):
            raise ValueError("The number of linestyles must match the number of data arrays if linestyles are provided.")
    else:
        ls = ('-',) * len(arrays)
    
    global_max = max(np.nanmax(arr) for arr in arrays)
    global_min = min(np.nanmin(arr) for arr in arrays)
    
    wide = abs(global_max - global_min) / n_quantiles
    bins = np.arange(global_min, global_max + wide, wide)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (arr, label) in enumerate(zip(arrays, labels)):
        pdf = get_pdf(arr, bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # Get the center of each bin
        
        ax.plot(bin_centers, pdf, label=label, color=colors[i], ls=ls[i])
    
    ax.set_xlabel('Precipitation (mm/h)')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.xlim(global_min, 60)
    
    return fig, ax


def plot_pp(
    arrays, 
    labels,
    n_quantiles=200,
    colors=None,
    ls=None,
    cmap=CURVE_CMAP,
    xlim_max=5,
):
    """
    Plot the precipitation intensity distribution of multiple arrays.

    ## Parameters:
    - arrays (tuple of arrays): tuple of arrays to plot for.
    - labels (tuple of str): tuple of labels for the arrays. Should match the length of arrays.
    - n_quantiles (int, optional): number of quantiles to divide the data into. Default is 100.
    - colors (tuple of str, optional): tuple of colors for each plot. If provided, should match the length of arrays.
    - ls (tuple of str, optional): tuple of linestyles for each plot. If provided, should match the length of arrays.
    - cmap (str, optional): colormap to use for plotting (default: custom colormap)
    - xlim_max (float, optional): maximum value for the x-axis. Default is 5.

    ## Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object)
    """
    if len(arrays) != len(labels):
        raise ValueError("The number of arrays and labels must be the same.")
    
    # Colors
    if colors is not None:
        if len(colors) != len(arrays):
            raise ValueError("The number of colors must match the number of data arrays if colors are provided.")
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(arrays))]
        
    # Linestyle
    if ls is not None:
        if len(ls) != len(arrays):
            raise ValueError("The number of linestyles must match the number of data arrays if linestyles are provided.")
    else:
        ls = ('-',) * len(arrays)
    
    global_max = max(np.nanmax(arr) for arr in arrays)
    global_min = max(min(np.nanmin(arr) for arr in arrays), 0)  # Ensure non-negative
    
    wide = abs(global_max - global_min) / n_quantiles
    bins = np.arange(global_min, global_max + wide, wide)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (arr, label) in enumerate(zip(arrays, labels)):
        pdf = get_pdf(arr, bins)
        bin_centers = bins[:-1]
        
        # Calculate precipitation intensity distribution
        precip_intensity_dist = pdf * bin_centers
        
        ax.plot(bin_centers, precip_intensity_dist, label=label, color=colors[i], ls=ls[i])
    
    plt.xlim(0, xlim_max)
    ax.set_xlabel('Precipitation (mm/h)')
    ax.set_ylabel('Precipitation Intensity Distribution (mm/h)')
    ax.legend()
    ax.set_frame_on(False)
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='both', color='white')
    ax.get_yaxis().set_tick_params(which='both', color='white')
    
    return fig, ax
    
    
def plot_psds(
    arrays,
    labels,
    spatial_lengths,
    min_threshold=None,
    max_threshold=None,
    colors=None,
    ls=None,
    cmap=CURVE_CMAP,
    lambda_star=None,
    psd_star=None,
    data_std=1,
    rotation_angle=0,
):
    """
    Plot multiple PSDs on the same figure.

    Parameters:
    - arrays (tuple of arrays): tuple of data arrays to compute PSDs for.
    - labels (tuple of str): tuple of labels for the data arrays. Should match the length of arrays.
    - spatial_lengths (tuple of tuples): tuple of (x_length, y_length) tuples for each data array. Should match the length of arrays.
    - colors (tuple of str, optional): tuple of colors for each plot. If provided, should match the length of arrays.
    - ls (tuple of str, optional): tuple of linestyles for each plot. If provided, should match the length of arrays.
    - min_threshold (float, optional): minimum threshold value to mask out low PSD values
    - max_threshold (float, optional): maximum threshold value to mask out high PSD values
    - cmap (str, optional): colormap to use for plotting (default: 'Dark2')
    - lambda_star (float, optional): point of intersection of curves (to be plotted as a vertical line).
    - data_std (float, optional): standard deviation of the data. Default is 1.
    - rotation_angle (float, optional): angle of rotation for the 1D-PSD condensation. Default is np.pi/4.

    Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object)
    """
    if len(arrays) != len(labels) or len(arrays) != len(spatial_lengths):
        raise ValueError("The number of data arrays, labels, and spatial lengths must be the same.")
    
    # Colors
    if colors is not None:
        if len(colors) != len(arrays):
            raise ValueError("The number of colors must match the number of data arrays if colors are provided.")
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(arrays))]
        
    # Linestyle
    if ls is not None:
        if len(ls) != len(arrays):
            raise ValueError("The number of linestyles must match the number of data arrays if linestyles are provided.")
    else:
        ls = ('-',) * len(arrays)
    
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (data, label, (x_length, y_length)) in enumerate(zip(arrays, labels, spatial_lengths)):
        k, psd = get_1dpsd(data, x_length, y_length, data_std=data_std, rotation_angle=rotation_angle)
        wavelengths = 2*np.pi / k
        
        if min_threshold is not None:
            mask = (psd >= min_threshold)
        else:
            mask = np.ones_like(psd, dtype=bool)
            
        if max_threshold is not None:
            mask &= (psd <= max_threshold)
        
        plt.loglog(wavelengths[mask], psd[mask], label=label, color=colors[i], ls=ls[i])
        
    ax.legend(fontsize='medium')
    
    if lambda_star is not None:
        ax.axvline(lambda_star, color='black', linestyle='--', label=r'$\lambda^*$')
        ax.text(lambda_star*1.1, 1e-5, r'$\lambda^\star$', fontsize='large')
    
    if psd_star is not None:
        ax.axhline(psd_star, color='black', linestyle='--', label=r'$\sigma^*$')
        ax.text(1e1, psd_star*1.1, r'$\sigma^\star$', fontsize='large')
        
    plt.ylim(1e-10,None)
    
    ax.set_frame_on(False)
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power spectral density")
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='both', color='white')
    ax.get_yaxis().set_tick_params(which='both', color='white')
    plt.tight_layout()
    
    return fig, ax
