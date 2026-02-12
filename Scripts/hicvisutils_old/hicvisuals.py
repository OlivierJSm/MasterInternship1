import os
import hictkpy as htk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

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
    "higlass": higlass_cmap,
}

# General function for visualization
def hic2vis(file_path: str, chrom: str, start: int = None, end: int = None, 
            resolution: int = None, cmap: str = None, title: str = None ):
    '''
        Takes a .hic/.cool/.mcool file, specified chromosome, range on said chromosome, resolution in 
        bp, color map and title and visualizes a contact map for these specifications.

        Arguments:
        file_path   --  Path to .hic/.cool/.mcool file (str).
        chrom       --  Desired chromosome to be visualized (str).
        start       --  Start of contacts on specified chromosome for contact map (int).
                        Default = None.
        end         --  End of contacts on specified chromosome for contact map (int).
                        Default = None.
        resolution  --  Bin size for contact map. Ignored for single resolution .cool files (int).
                        Default = None
        cmap        --  Color map for visualization. Options include standard color 
                        scheme (None, default), juicebox and higlass (str).
        title       --  Figure title (str).
                        Default = None, gives default figure title.
        
        Returns:
        File imported via hictkpy.File()
    '''
    # TODO: error for if files have different extension
    # TODO: more elegant implementation to automatically convert to kb or Mb for resolution
    # TODO: more elegant implementation for axes in cases of windows smaller than Mb scale

    # Cmap doesn't exist
    if cmap != None and cmap not in cmaps.keys():
        raise ValueError(
            f"Invalid map '{cmap}'. Choose from {cmaps.keys()} or leave as None"    # Not very elegant, but works
        )
    
    # File doesn't exist
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file does not exist: '{file_path}'")
    
    # Choose colormap (select from dict if not default) 
    if cmap != None:
        cmap = cmaps[cmap]

    # Import file
    if resolution != None:  # Hic or mcool file
        contacts = htk.File(file_path, resolution)
    else:   # Cool file
        contacts = htk.File(file_path)
        resolution = contacts.resolution()

    # Get chromosome details, define start and end
    chrom_len = contacts.chromosomes()[chrom]
    
        # Set defaults if empty, check boundaries if filled
    if start is not None and end is not None and start >= end:
        raise ValueError(f"Start ({start}) must be less than end ({end})")
    
    if start == None:
        start = 0
    elif start < 0 or start >= chrom_len:
        raise ValueError(f"start coordinate {start} is out of bounds for chromosome '{chrom}' (0-{chrom_len})")

    if end == None:
        end = chrom_len - 1
    elif end <= 0 or end > chrom_len:
        raise ValueError(f"end coordinate {end} is out of bounds for chromosome '{chrom}' (0-{chrom_len})")

    # Handling figure title
    if title == None:
        title = f"Interactions on chr{chrom} ({resolution/1000:.0f} kb bins, {start:,}-{end:,})"

    # Define region to visualize
    region = f"{chrom}:{str(start)}-{str(end+1)}"   # End + 1 to end the diagram at an even tick with even numbers

    # Fetching and coversion to NDArray
    sel = contacts.fetch(region, join=True)
    map = sel.to_numpy()

    # Plotting
    plt.imshow(map, norm=LogNorm(), cmap=cmap)

        # Naming axis labels
    plt.xlabel(f"Genomic coordinate on {chrom} (Mb)")
    plt.ylabel(f"Genomic coordinate on {chrom} (Mb)")

        # Define axis object
    ax = plt.gca()

        # Get number of bins in the plotted region
    n_bins = map.shape[0]

        # Bin indices go from 0 to n_bins-1
    bin_indices = range(n_bins)

        # Genomic positions corresponding to bins
    genomic_positions = [start + i * resolution for i in bin_indices]

        # Choose ~6 evenly spaced ticks within the genomic range
    num_ticks = 6
    step = max(1, n_bins // (num_ticks - 1))
    tick_bins = list(range(0, n_bins, step))
    if tick_bins[-1] != n_bins - 1:  # ensure last tick at end
        tick_bins.append(n_bins - 1)

    tick_labels = [f"{genomic_positions[i] / 1e6:.1f}" for i in tick_bins]

        # Apply ticks
    ax.set_xticks(tick_bins)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_bins)
    ax.set_yticklabels(tick_labels)

        # Colorbar, title and showing
    plt.colorbar(label="Interaction counts (log scale)")
    plt.title(title)
    plt.show()

    return contacts

def sc_hic2vis(cool_path: str, chrom: str, start: int = None, end: int = None, title: str = None):
    '''
        Takes a .cool file with single-cell data, specified chromosome, range on said chromosome
        and title and visualizes a contact map for these specifications.

        Arguments:
        cool_path   --  Path to .cool file (str).
        chrom       --  Desired chromosome to be visualized (str).
        start       --  Start of contacts on specified chromosome for contact map (int).
                        Default = None.
        end         --  End of contacts on specified chromosome for contact map (int).
                        Default = None.
        title       --  Figure title (str).
                        Default = None, gives default figure title.
        
        Returns:
        File imported via hictkpy.File()
    '''
    # File doesn't exist
    if not os.path.exists(cool_path):
        raise FileNotFoundError(f"The specified file does not exist: '{cool_path}'")

    # Importing file
    contacts = htk.File(cool_path)
    resolution = contacts.resolution()

    # Get chromosome details, define start and end
    chrom_len = contacts.chromosomes()[chrom]
    
        # Set defaults if empty, check boundaries if filled
    if start is not None and end is not None and start >= end:
        raise ValueError(f"Start ({start}) must be less than end ({end})")
    
    if start == None:
        start = 0
    elif start < 0 or start >= chrom_len:
        raise ValueError(f"start coordinate {start} is out of bounds for chromosome '{chrom}' (0-{chrom_len})")

    if end == None:
        end = chrom_len
    elif end <= 0 or end > chrom_len:
        raise ValueError(f"end coordinate {end} is out of bounds for chromosome '{chrom}' (0-{chrom_len})")

    # Handling figure title
    if title == None:
        title = f"Interactions on chromosome {chrom} ({resolution/1000:.0f} kb bins, {start:,}-{end:,})"

    # Define region to visualize
    region = f"{chrom}:{str(start)}-{str(end)}"

    # Fetching and coversion to NDArray
    sel = contacts.fetch(region, join=True, normalization='balanced')
    map = sel.to_numpy()

    # Constructing dot plot for less empty space
    x, y = np.where(map > 0)
    print(x)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=100000, color='black')
    plt.xlabel(f"Genomic coordinate")
    plt.ylabel(f"Genomic coordinate")
    plt.title(title)
    plt.show()

    return sel