import os
import cooler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# Ensure basepair formatting
bp_formatter = EngFormatter('b')

# Defining available color schemes
juicebox_cmap = LinearSegmentedColormap.from_list(
    "juicebox", ["#FFFFFF", "#FFD9A8", "#FF6600", "#990000"], N=256
)
juicebox_cmap.set_bad(color="white")

higlass_cmap = LinearSegmentedColormap.from_list(
    "higlass", ["#FFFFFF", "#FFFF00", "#FF4500", "#800080"], N=256
)
higlass_cmap.set_bad(color="white")

# Dict for color scheme access
cmaps = {
    "juicebox": juicebox_cmap,
    "higlass": higlass_cmap
}

def format_ticks(ax, x=True, y=True, rotate=True):
    '''
        Formats axes for a contact map to use basepairs instead of
        genomic positions in bins.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes     
            Axes to be converted to basepairs.
        x : bool     
            Controls if the x axis is reformated.
            Default = True.
        y : bool 
            Controls if the y axis is reformated.
            Default = False.
        rotate : bool 
            Controls if the x axis is rotated 45 degrees.
            Default = True.

        Returns
        -------
        True if operations were succesful.
    '''

    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)
    
    return True

def hic_vis(clr: cooler.api.Cooler, title: str = None, data_type: str = "raw", cmap: str = None):
    '''
        Takes a cooler object and optionally a title and color map and 
        creates a genome-wide contact map.

        Parameters
        ----------
        clr : cooler.api.Cooler 
            A cooler object for a single cell Hi-C dataset.
        title : str      
            Desired figure title.
            Default = None, uses a default title.
        data_type : str 
            Type of data parsed. Affects the colorbar title.
            Default = raw.
        cmap : str 
            Color map for visualization. Options include standard color 
            scheme (None, default), juicebox and higlass.

        Returns
        -------
        stats : dict[str, float]    
            Summary statistics for non-zero entries (sum, max, min).
    '''
    # Cmap doesn't exist
    if cmap != None and cmap not in cmaps.keys():
        raise ValueError(
            f"Invalid map '{cmap}'. Choose from {cmaps.keys()} or leave as None"    # Not very elegant, but works
        )
    
    # Choose colormap (select from dict if not default) 
    if cmap != None:
        cmap = cmaps[cmap]

    # Fetching statistics for non-zero entries
    pixels = clr.matrix(balance=False, as_pixels=True)[:]
    vsum = pixels['count'].sum()
    vmax = pixels['count'].max()
    vmin = pixels['count'].min()

    # Getting chromosome starts
    chromstarts = []
    for i in clr.chromnames:
        chromstarts.append(clr.extent(i)[0])

    # Handling title
    if title == None:
        title = "Genome-Wide Contact Map"

    # Defining subplot and norm
    f, ax = plt.subplots(figsize=(7,6))
    norm = LogNorm(vmax=vmax)

    # Constructing and showing figure
    im = ax.matshow(
        clr.matrix(balance=False)[:], 
        norm=norm,
        cmap = cmap
        )
    plt.colorbar(im, ax=ax ,fraction=0.046, pad=0.04, label=f'{data_type} counts')
    ax.set_xticks(chromstarts)
    ax.set_xticklabels(clr.chromnames)
    ax.set_yticks(chromstarts)
    ax.set_yticklabels(clr.chromnames)
    ax.xaxis.tick_bottom()
    ax.set_title(title)

    # Stats
    stats = {"sum": vsum, "max": vmax, "min": vmin}

    return stats

def hic_vis_region(clr: cooler.api.Cooler, chrom: str, start: int = 0,
                    end: int = None, title: str = None,
                    data_type: str = "raw", cmap: str = None, ax=None):
    '''
        Takes a cooler object, specified chromosome and region and optionally 
        a title and color map and creates a contact map for this region.

        Parameters
        ----------
        clr : cooler.api.Cooler 
            A cooler object for a single cell Hi-C dataset.
        chrom : str      
            The chromosome that should be visualized.
        start : int 
            The start of the desired region in bp.
            Default = 0.
        end : int 
            The end of the desired region in bp.
            Default = None, sets region to the end of the chromosome.
        title : str      
            Desired figure title.
            Default = None, uses a default title.
        data_type : str 
            Type of data parsed. Affects the colorbar title.
            Default = raw.
        cmap : str 
            Color map for visualization. Options include standard color 
            scheme (None, default), juicebox and higlass.
            Default = None.
        ax
            Optional argument to allow function to add to subfigures
            rather than create a whole new figure.
            Default = None.

        Returns
        -------
        stats : dict[str, float]    
            Summary statistics for non-zero entries (sum, max, min).
    '''
    # Cmap doesn't exist
    if cmap != None and cmap not in cmaps.keys():
        raise ValueError(
            f"Invalid map '{cmap}'. Choose from {cmaps.keys()} or leave as None" 
        )
    
    # Choose colormap (select from dict if not default) 
    if cmap != None:
        cmap = cmaps[cmap]

    # Initial chromosome data
    chrom_len = clr.chromsizes[chrom]
    
    # Handling start and end
    if end is not None and start >= end:
        raise ValueError(f"Start ({start}) must be less than end ({end})")
    
    if start < 0 or start >= chrom_len:
        raise ValueError(f"start coordinate {start} is out of bounds for chromosome '{chrom}' (0-{chrom_len})")

    if end == None:
        end = chrom_len
    elif end <= 0 or end > chrom_len:
        raise ValueError(f"end coordinate {end} is out of bounds for chromosome '{chrom}' (0-{chrom_len})")
    
    region = (chrom, start, end)

    # Fetching statistics for non-zero entries
    pixels = clr.matrix(balance=False, as_pixels=True).fetch(region)
    vsum = pixels['count'].sum()
    vmax = pixels['count'].max()
    vmin = pixels['count'].min()

    # Handling title
    if title == None:
        title = f"{chrom}:{start:,}-{end:,}"
    
    # Axis creation if needed
    created_fig = False
    if ax is None:
        f, ax = plt.subplots(figsize=(7,6))
        created_fig = True

    # Plotting
    norm = LogNorm(vmax=vmax)
    im = ax.matshow(
        clr.matrix(balance=False).fetch(region),
        norm=norm,
        extent=(start, end, end, start),
        cmap = cmap
        )
    ax.set_title(title)
    plt.colorbar(im, ax=ax ,fraction=0.046, pad=0.04, label=f'{data_type} counts')
    ax.set(xlabel="position, Mb", ylabel="position, Mb")
    format_ticks(ax)
    
    if created_fig:
        plt.tight_layout()

    # Stats
    stats = {"sum": vsum, "max": vmax, "min": vmin}

    return stats

def hic_vis_region_zoom(
        clr: cooler.api.Cooler, 
        chrom: str,
        title: str|None=None,
        chr_label: str|None = None, 
        cmap: str|None=None, 
        textsize: int = 12,
        ax: plt.Axes|None=None,
        **kwargs
    ) -> None:
    '''
        Visualizes the specified region from a provided cooler
        with minimal visual details, zooming in on regions with values.

        Parameters
        ----------
        clr : cooler.api.Cooler 
            A cooler object for a single cell Hi-C dataset.
        chrom : str      
            The chromosome that should be visualized.
        title : str      
            Desired figure title.
            Default = None, uses a default title.
        chr_label : str
            Label on the y-axis.
            Default = None, uses no label.
        cmap : str 
            Color map for visualization. Options include standard color 
            scheme (None, default), juicebox and higlass.
            Default = None.
        textsize : int
            Size of the label text.
            Default = 14.
        ax : plt.Axes
            Optional argument to allow function to add to subfigures
            rather than create a whole new figure.
            Default = None.
        
        Other Parameters
        ----------------
        **kwargs
            Arguments passed to imshow().
        
        Returns
        -------
        ax : plt.Axes
            Axes object containing the plot.
    '''
    # Cmap doesn't exist
    if cmap != None and cmap not in cmaps.keys():
        raise ValueError(
            f"Invalid map '{cmap}'. Choose from {cmaps.keys()} or leave as None" 
        )
    
    # Choose colormap (select from dict if not default) 
    if cmap != None:
        cmap = cmaps[cmap]
    
    # Fetching chromosome
    matrix = clr.matrix(balance=False).fetch(chrom)
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Find the first and last bins that contain at least one non-zero value
    nonzero_bins = np.where(matrix.any(axis=1))[0]
    if nonzero_bins.size == 0:
        raise ValueError(f"Contact matrix for {chrom} contains only zero values.")
    first, last = nonzero_bins[0], nonzero_bins[-1] + 1
    matrix = matrix[first:last, first:last]

    # Initialize figure if not passed
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Arguments for imshow
    imshow_kwargs = dict(
        cmap=cmap,
        norm=LogNorm(vmax=matrix.max()),
        aspect="auto",
    )
    imshow_kwargs.update(kwargs)

    # Visualizing
    img = ax.imshow(matrix, **imshow_kwargs)

    # Formatting axes
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if chr_label is not None:
        ax.set_ylabel(chr_label, rotation='vertical', size=textsize)

    return ax

def hic_vis_matrix(matrix : np.ndarray, title: str, data_type: str="raw",
                   cmap: str=None, ax=None):
    '''
        Visualization function for edited contact maps in the form of a matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix resembling a contact map.
        title : str      
            Desired figure title.
        data_type : str 
            Type of data parsed. Affects the colorbar title.
            Default = raw.
        cmap : str 
            Color map for visualization. Options include standard color 
            scheme (None, default), juicebox and higlass.
            Default = None.
        ax
            Optional argument to allow function to add to subfigures
            rather than create a whole new figure.
            Default = None.

        Returns
        -------
        vmax : float    
            Maximum value in the input matrix.
    '''
    # Cmap doesn't exist
    if cmap != None and cmap not in cmaps.keys():
        raise ValueError(
            f"Invalid map '{cmap}'. Choose from {cmaps.keys()} or leave as None"    # Not very elegant, but works
        )
    
    # Choose colormap (select from dict if not default) 
    if cmap != None:
        cmap = cmaps[cmap]
    
    # Fetching statistics
    vmax = matrix.max()

    # Axis creation if needed
    created_fig = False
    if ax is None:
        f, ax = plt.subplots(figsize=(7,6))
        created_fig = True
    
    # Plotting
    norm = LogNorm(vmax=vmax)
    im = ax.matshow(
        matrix,
        norm=norm,
        cmap=cmap
        )
    ax.set_title(title)
    plt.colorbar(im, ax=ax ,fraction=0.046, pad=0.04, label=f'{data_type} counts')
    ax.set(xlabel="position, Mb", ylabel="position, Mb")
    ax.xaxis.tick_bottom()
    
    if created_fig:
        plt.tight_layout()

    return vmax