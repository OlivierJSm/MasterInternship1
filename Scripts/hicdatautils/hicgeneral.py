import os
import cooler
import pandas as pd
import numpy as np
from pathlib import Path

def import_cool(file_path: str, resolution: int = None):
    '''
        Takes a .cool/.mcool file and returns a cooler object.

        Parameters
        ----------
        file_path : str
            Path to the .cool/.mcool file.
        resolution : int 
            Desired resolution to be imported for .mcool files.
            Default = None, for single resolution .cool files

        Returns
        -------
        clr : cooler.api.Cooler
            Cooler object for imported file.
    '''
    # Initial Errors    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file does not exist: '{file_path}'")

    # Open file
    if file_path.endswith(".mcool"):
        # Error if no resolution
        if resolution == None:
            raise ValueError("resolution must be specified for .mcool files")
        
        uri = f"{file_path}::resolutions/{resolution}"
        clr = cooler.Cooler(uri)
    elif file_path.endswith(".cool"):
        clr = cooler.Cooler(file_path)
    else:
        raise ValueError("Invalid file extension. Please provide .cool or .mcool file.")

    return clr

def import_cool_dir(dir: str, resolution: int = None, include: list[str] = None):
    '''
        Imports all .cool/.mcool files in the given directory as cooler
        objects.

        Parameters
        ----------
        dir : str       
            The path to the folder with .cool/.mcool files.
        resolution : int
            The resolution at which to import .mcool files.
            Default = None, for single resolution .cool files.
        include : list[str]
            A list of filenames to include when importing.
            If none, imports all cool files.
        
        Returns
        -------
        clrs : list[cooler.api.Cooler]
            Array of cooler objects.
    '''
    # Get .cool and .mcool files in directory
    folder = Path(dir)

    # Import all
    files = [f for f in folder.glob("*.cool")] + [f for f in folder.glob("*.mcool")]

    # Select subset for include
    if include is not None:
        valid = set(Path(v).stem for v in include)
        files = [f for f in files if f.stem in valid]

    # Convert to string and import
    files = [str(f) for f in files]
    clrs = [import_cool(f, resolution) for f in files]

    # Import all files as cooler objects
    return clrs

def fetch_region(clr: cooler.api.Cooler, chrom: str, start: int = None, end: str = None,
                 balanced: bool = False):
    '''
        Fetches the desired intrachromosomal region from a cooler object.

        Parameters
        ----------
        clr : cooler.api.Cooler
            Cooler object.
        chrom : str     
            Desired chromosome.
        start : int      
            Start of the region in bp.
        end : int 
            End of the region in bp.
        balanced : bool 
            Controls whether balanced reads will be used from the
            cooler object if present.
            Default = False.
        
        Returns
        -------
        matrix : np.ndarray
            Matrix with read counts per genomic bin for the requested region.
    '''
    # Get chromosome size for defaults
    chrom_sizes = clr.chromsizes
    if chrom not in chrom_sizes:
        raise ValueError(f'Chromosome \'{chrom}\' not found in Cooler object.')
    chrom_len = chrom_sizes[chrom]

    # Defaults
    if start is None:
        start = 0
    if end is None:
        end = chrom_len
    
    # Check valid start and end
    if not (0 <= start < end <= chrom_len):
        raise ValueError(f'Invalid region provided: start={start}, end={end}, chromosome length={chrom_len}')
    
    # Allign start and end to nearest bins
    bin_size = clr.binsize
    start = (start // bin_size) * bin_size
    end = ((end + bin_size - 1) // bin_size) * bin_size
    end = min(end, chrom_len)

    # Fetch matrix
    region_str = f'{chrom}:{start}-{end}'
    matrix = clr.matrix(balance=balanced).fetch(region_str)
    
    return matrix

def get_chrs(clr: cooler.api.Cooler, chr_exclude: set = None):
    '''
        Takes a cooler object and returns the (1) chromosome names
        and (2) chromosome lengths.

        Parameters
        ----------
        clr : cooler.api.Cooler         
            Cooler object (cooler.api.Cooler).
        chr_exclude : set[str] 
            Chromosome names to exclude from the chromosome names and chromosome 
            lengths.

        Returns
        -------
        chr_names : list[str]
            Chromosome names (list)
        chr_lengths : list[int]
            Chromosome lengths (list)
    '''
    # Get chromosomes and lengths
    chrom_table = clr.chromsizes  # pandas Series: index=chrom names, values=lengths
    
    if chr_exclude is not None:
        missing = set(chr_exclude) - set(chrom_table.index)
        if missing:
            raise ValueError(f"Chromosomes not found in file: {missing}")
        chrom_table = chrom_table.drop(chr_exclude)

    chr_names = list(chrom_table.index)
    chr_lengths = list(chrom_table.values)

    return chr_names, chr_lengths

def standardize_coolers(clr1: cooler.api.Cooler, clr2: cooler.api.Cooler, out_path1: str,
                        out_path2: str):
    '''
        Takes two cooler files and creates new .cool files such that they share
        equivalent chromosome lengths. Additional, removes any chromosomes not
        shared by both datasets.

        Parameters
        ----------
        clr1 : cooler.api.Cooler   
            Cooler file for a dataset.
        clr2 : cooler.api.Cooler
            Cooler file for a dataset.

        Returns
        -------
        st_clr1 : cooler.api.Cooler
            Standardized cooler file.
        st_clr2 : cooler.api.Cooler
            Standardized cooler file.
    '''
    # TODO: Write better check for count data types and implement in cooler creation
    
    # Check binsizes
    if clr1.binsize != clr2.binsize:
        raise ValueError(f"Bin sizes differ: {clr1.binsize} vc {clr2.binsize}.")

    # Load bin tables
    bins1 = clr1.bins()[:]
    bins2 = clr2.bins()[:]

    # Merge bins for intersection
    merged_bins = pd.merge(
        bins1.reset_index(), bins2.reset_index(),
        on = ["chrom", "start", "end"], how="inner", sort=False,
        suffixes=("_1", "_2")
    )
    if merged_bins.empty:
        raise ValueError("No overlapping bins found between datasets.")
    
    # Map old indices to new indices
    mapping1 = {old: new for new, old in enumerate(merged_bins["index_1"])}
    mapping2 = {old: new for new, old in enumerate(merged_bins["index_2"])}

    # Filter and reset bins
    bins1_filtered = bins1.loc[merged_bins["index_1"]].reset_index(drop=True)
    bins2_filtered = bins2.loc[merged_bins["index_2"]].reset_index(drop=True)

    # Filter pixels
    pixels1_filtered = clr1.pixels()[:]
    pixels1_filtered = pixels1_filtered[
        pixels1_filtered["bin1_id"].isin(mapping1) &
        pixels1_filtered["bin2_id"].isin(mapping1)
    ].copy()
    pixels1_filtered["bin1_id"] = pixels1_filtered["bin1_id"].map(mapping1).astype(int)
    pixels1_filtered["bin2_id"] = pixels1_filtered["bin2_id"].map(mapping1).astype(int)

    pixels2_filtered = clr2.pixels()[:]
    pixels2_filtered = pixels2_filtered[
        pixels2_filtered["bin1_id"].isin(mapping2) &
        pixels2_filtered["bin2_id"].isin(mapping2)
    ].copy()
    pixels2_filtered["bin1_id"] = pixels2_filtered["bin1_id"].map(mapping2).astype(int)
    pixels2_filtered["bin2_id"] = pixels2_filtered["bin2_id"].map(mapping2).astype(int)

    # Write new cooler files
    cooler.create_cooler(out_path1, bins1_filtered, pixels1_filtered, ordered=True, dtypes={'count': 'float64'})
    cooler.create_cooler(out_path2, bins2_filtered, pixels2_filtered, ordered=True, dtypes={'count': 'float64'})
    st_clr1 = cooler.Cooler(out_path1)
    st_clr2 = cooler.Cooler(out_path2)

    return st_clr1, st_clr2

def standardize_coolers_ref(clr: cooler.api.Cooler, ref: cooler.api.Cooler, out_path: str):
    '''
        Trims an input cooler object to match the chromosome lengths of the
        provided reference cooler object. Writes the trimmed file to the 
        out_path.

        Parameters
        ----------
        clr : cooler.api.Cooler
            Cooler object to be trimmed.
        ref : cooler.api.Cooler
            Reference cooler object with shorter chromosomes.
        out_path : str
            The export path for the .cool file.

        Returns
        -------
        clr_out : cooler.api.Cooler
            Trimmed cooler object.
    '''
    # Check binsizes
    if clr.binsize != ref.binsize:
        raise ValueError(f"Bin sizes differ: {clr.binsize} vc {ref.binsize}.")

    # Load bin tables
    bins = clr.bins()[:]
    bins_ref = ref.bins()[:]

    # Merge bins for intersection
    merged_bins = pd.merge(
        bins.reset_index(), bins_ref.reset_index(),
        on = ["chrom", "start", "end"], how="inner", sort=False,
        suffixes=("_1", "_2")
    )
    if merged_bins.empty:
        raise ValueError("No overlapping bins found between datasets.")
    
    # Map old indices to new indices
    mapping = {old: new for new, old in enumerate(merged_bins["index_1"])}

    # Filter and reset bins
    bins_filtered = bins.loc[merged_bins["index_1"]].reset_index(drop=True)

    # Filter pixels
    pixels_filtered = clr.pixels()[:]
    pixels_filtered = pixels_filtered[
        pixels_filtered["bin1_id"].isin(mapping) &
        pixels_filtered["bin2_id"].isin(mapping)
    ].copy()
    pixels_filtered["bin1_id"] = pixels_filtered["bin1_id"].map(mapping).astype(int)
    pixels_filtered["bin2_id"] = pixels_filtered["bin2_id"].map(mapping).astype(int)

    # Write new cooler files
    cooler.create_cooler(out_path, bins_filtered, pixels_filtered, ordered=True, dtypes={'count': 'float64'})
    clr_out = cooler.Cooler(out_path)

    return clr_out

def get_cool_name(clr: cooler.api.Cooler):
    '''
        Extracts the file name for a cooler object.

        Parameters
        ----------
        clrs : cooler.api.Cooler
            Cooler objects.

        Returns
        -------
        filename : str
            Name of the cooler file.
    '''
    url = clr.uri
    filename = Path(url.split("::")[0].split(".")[0]).name

    return filename

def standardize_coolers_bulk(clrs: list[cooler.api.Cooler], ref: cooler.api.Cooler, out_dir: str):
    '''
        Takes a list of cooler objects and trims their lengths to match the provided
        reference cooler object. Outputs all trimmed .cool files into the output
        directory.

        Parameters
        ----------
        clrs : list[cooler.api.Cooler]
            List of cooler objects to be trimmed.
        ref : cooler.api.Cooler       
            Reference cooler object.
        out_dir : str    
            Output directory for trimmed files.
    '''
    # Loop through every cooler and standardize to reference
    for clr in clrs:
        filename = get_cool_name(clr)
        standardize_coolers_ref(clr, ref, f"{out_dir}/{filename}_trimmed.cool")
    
    return

def bin_contact_map(matrix : np.ndarray, bin_size : int):
    '''
        Bins a Hi-C-like contact map to smaller total bin numbers
        by binning bin_size bins together.

        Parameters 
        ----------
        matrix : np.ndarray
            Square matrix to be binned.
        bin_size : int
            Number of bins that will become one bin on one axis.
            Any bins outside of this range will be deleted.
        
        Returns
        -------
        binned_matrix : np.ndarray
            Rebinned matrix.
    '''
    # Assure square matrix
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("Input matrix is not symmetrical!")
    
    # Assure valid bin_size
    if bin_size < 0:
        raise ValueError(f"Invalid bin_size: {bin_size}")
    
    # Binning
    m = n // bin_size
    binned_matrix = matrix[:m*bin_size, :m*bin_size].reshape(m, bin_size, m, bin_size).sum(axis=(1, 3))

    return binned_matrix

def remap_pixels(clr: cooler.api.Cooler, old_map: dict, new_map: dict):
    '''
        FUNCTION IS DEPRECATED, USE STANDARDIZE_COOLERS
    
        Takes a cooler files, a dictionary mapping each bin to an index for
        the original cooler file and a dictionary with a similar mapping but
        for a harmonized cooler files and returns a pixels dataframe with
        harmonized indices for bin1_id and bin2_id.

        Arguments:
        clr     --  Cooler object (cooler.api.Cooler).
        old_map --  Dictionary mapping genomic coordinates to indices for 
                    original cooler file (dict).
        new_map --  Dictionary mapping genomic coordinates to indices for
                    harmonized cooler file (dict).
        
        Returns:
        Harmonized pixels dataframe (pandas.DataFrame).
    '''
    # TODO: more elegant solution for floats (check if input data is normalized in main function?)

    # Get the pixels
    pixels = clr.pixels()[:]
    
    # Create reverse mapping to translate index to genomic region
    reverse_old = {v: k for k, v in old_map.items()}

    # Map old bins to coordinates and map back to new indices
    pixels["bin1_id"] = pixels["bin1_id"].map(lambda i: new_map.get(reverse_old.get(i)))
    pixels["bin2_id"] = pixels["bin2_id"].map(lambda i: new_map.get(reverse_old.get(i)))

    # Drop pixels where mapping failed (regions not shared between old and new map)
    pixels = pixels.dropna(subset=["bin1_id", "bin2_id"])
    pixels["bin1_id"] = pixels["bin1_id"].astype(int)
    pixels["bin2_id"] = pixels["bin2_id"].astype(int)

    # Ensure floats for counts in case of normalized data
    pixels["count"] = pixels["count"].astype(float)

    return pixels

def harmonize_coolers(clr1: cooler.api.Cooler, clr2: cooler.api.Cooler, out_path1: str,
                        out_path2: str):
    '''
        FUNCTION IS DEPRECATED, USE STANDARDIZE_COOLERS. IT DOES THE SAME THING
        BUT FASTER

        Takes two cooler files and creates new .cool files such that they share
        equivalent chromosome lengths. Additional, removes any chromosomes not
        shared by both datasets.

        Arguments:
        clr1    --  Cooler file for a dataset (cooler.api.Cooler).
        clr2    --  Cooler file for a dataset (cooler.api.Cooler).

        Returns:
        Standardized cooler file (cooler.api.Cooler).
        Standardized cooler file (cooler.api.Cooler).
    '''
    # TODO: Write better check for count data types and implement in cooler creation
    # TODO: Fix bug of removing empty data

    # Check binsizes
    if clr1.binsize != clr2.binsize:
        raise ValueError(f"Bin sizes differ: {clr1.binsize} vc {clr2.binsize}.")
    bin_size= clr1.binsize

    # Load bin tables
    bins1 = clr1.bins()[:]
    bins2 = clr2.bins()[:]

    # Identify shared chromosomes
    shared_chroms = sorted(set(bins1['chrom']) & set(bins2['chrom']))
    if not shared_chroms:
        raise ValueError("No shared chromosomes found between the datasets.")
    
    bins1 = bins1[bins1['chrom'].isin(shared_chroms)]
    bins2 = bins2[bins2['chrom'].isin(shared_chroms)]

    # Trim chromosomes to shared full bin length
    chrom_ends = {}
    for chrom in shared_chroms:
        end1 = bins1.loc[bins1['chrom'] == chrom, 'end'].max()
        end2 = bins2.loc[bins2['chrom'] == chrom, 'end'].max()
        min_end = min(end1, end2)

        # Snap to nearest full bin
        min_end = (min_end // bin_size) * bin_size
        chrom_ends[chrom] = min_end

    bins1 = bins1[bins1.apply(lambda row: row["end"] <= chrom_ends[row["chrom"]], axis=1)]
    bins2 = bins2[bins2.apply(lambda row: row["end"] <= chrom_ends[row["chrom"]], axis=1)]

    # Construct unified bin table
    all_bins = (
        pd.concat([bins1, bins2])
        .drop_duplicates(subset=["chrom", "start", "end"])
        .reset_index(drop=True)
        .sort_values(["chrom", "start", "end"])    # Fix incorrect mapping of chromosomes
    )

        # Create maps from genomic indices to original index 
    bins1_indexed = bins1.reset_index().set_index(["chrom", "start", "end"])["index"].to_dict()
    bins2_indexed = bins2.reset_index().set_index(["chrom", "start", "end"])["index"].to_dict()

        # Create map from genomic indices to new index
    unified_map = {
        (row.chrom, row.start, row.end): i for i, row in all_bins.iterrows()
    }

    # Remapping pixels
    pixels1 = remap_pixels(clr1, bins1_indexed, unified_map)
    pixels2 = remap_pixels(clr2, bins2_indexed, unified_map)

    # Creating new coolers
    cooler.create_cooler(out_path1, all_bins, pixels1, ordered=True, dtypes={'count': 'float64'})
    cooler.create_cooler(out_path2, all_bins, pixels2, ordered=True, dtypes={'count': 'float64'})
    
    return cooler.Cooler(out_path1), cooler.Cooler(out_path2)