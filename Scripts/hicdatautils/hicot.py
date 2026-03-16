import os
import cooler
import pandas as pd
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from .hicgeneral import fetch_region, get_cool_name, bin_contact_map, compile_hic_reads
from ot import dist, solve, dist_batch, solve_batch
from hicrep.utils import readMcool, cool2pixels, coolerInfo, getSubCoo, trimDiags, upperDiagCsr, meanFilterSparse

def get_diag_csr(fcool1, fcool2, chrName, bin_size = None, h=1, d_bp_max=1_000_000):
    '''
        Takes two .cool/.mcool files and returns the sparse csr matrices for both
        files for the specified chromosome, providing the upper diagonals. Diagonals
        have been normalized for total read counts. Depending on h, the diagonals may
        have additionally been mean filtered. Code adapted from hicrep on Github.

        Parameters
        ----------
        fcool1 : str 
            Path to a .cool/.mcool file.
        fcool2 : str    
            Path to a .cool/.mcool file.
        chrName : str   
            Name of the chromosome from which the diagonals should be extracted.
        bin_size : int  
            Bin size in both mcool files that will be used to compute the SCC.
            If inputs are .cool files, this parameter is ignored.
            Default = None.
        h : int
            The smoothing parameter used for the 2D mean filter smoothing.
            Default = 1.
        d_bp_max : int 
            The maximum genomic distance between contacts that will be considered for
            extracting the diagonals.
            Default = 1_000_000
        
        Returns
        -------
        matrix1_diags : sp.csr_matrix
            Selected diagonals for fcool1.
        matrix2_diags : sp.csr_matrix
            Selected diagonals for fcool2.
    '''
    # TODO: rewrite to just fetch 1 diagonal?

    # Extract file extension
    ext1 = os.path.splitext(fcool1)[1]
    ext2 = os.path.splitext(fcool2)[1]

    # Value errors
        # TODO: rewrite to function def
    if not isinstance(fcool1, str):
        raise TypeError(f"fcool1 must be a string, got {type(fcool1)}")
    if not isinstance(fcool2, str):
        raise TypeError(f"fcool2 must be a string, got {type(fcool2)}")
    if not isinstance(bin_size, int):
        raise TypeError(f"bin_size must be an integer, got {type(bin_size)}")
    if not isinstance(h, int):
        raise TypeError(f"h must be an interger, got {type(h)}")
    if not isinstance(d_bp_max, int):
        raise TypeError(f"d_bp_max must be an interger, got {type(d_bp_max)}")
    if not isinstance(chrName, str):
        raise TypeError(f"chrName must be a string, got {type(chrName)}")
    
    # File errors
    if not os.path.exists(fcool1):
        raise FileNotFoundError(f"The specified file does not exist: '{fcool1}'")
    if not os.path.exists(fcool2):
        raise FileNotFoundError(f"The specified file does not exist: '{fcool2}'")

    # Extension errors
    allowed_files = (".cool", ".mcool")
    if any(ext not in allowed_files for ext in (ext1, ext2)):
        raise ValueError(f"One or both files are not .cool or .mcool: {ext1} and {ext2}")
    if ext1 != ext2:
        raise ValueError(f"File types do not match: {ext1} vs {ext2}")
    
    # Handling .cool vs .mcool
    if ext1 == ".cool":
        cool1, bin_size_1 = readMcool(fcool1, -1)
        cool2, bin_size_2 = readMcool(fcool2, -1)
        bin_size = bin_size_1
    else:
        # Raise error if no bin size is specified
        if bin_size == None:
            raise ValueError("No bin size specified for .mcool file")
        
        cool1, bin_size_1 = readMcool(fcool1, bin_size)
        cool2, bin_size_2 = readMcool(fcool2, bin_size)

    # Handling binsizes
    bins1 = cool1.bins()
    bins2 = cool2.bins()

    assert coolerInfo(cool1, 'nbins') == coolerInfo(cool2, 'nbins'),\
        f"Input cool files have different number of bins"
    assert (cool1.chroms()[:] == cool2.chroms()[:]).all()[0],\
        f"Input file have different chromosome names"
    
    # Handling d_bp_max
    if d_bp_max == -1:
        # Upper bound
        d_max = coolerInfo(cool1, 'nbins')
    else:
        d_max = d_bp_max // bin_size + 1
    assert d_max > 1, f"Input d_bp_max is smaller than bin_size"

    # Conversion to pixels format
    pixel1 = cool2pixels(cool1)
    pixel2 = cool2pixels(cool2)

    # Get total number of contacts to normalize
        # TODO: figure out if we normalize by total count or not? Do we only do this per stratum?
    total_contacts1 = coolerInfo(cool1, 'sum')
    total_contacts2 = coolerInfo(cool2, 'sum')

    # Fetching desired region of pixels maps
    sub_matrix1 = getSubCoo(pixel1, bins1, chrName)
    assert sub_matrix1.size > 0, "Contact matrix 1 of chromosome %s is empty" % (chrName)
    assert sub_matrix1.shape[0] == sub_matrix1.shape[1],\
        "Contact matrix 1 of chromosome %s is not square" % (chrName)
    sub_matrix2 = getSubCoo(pixel2, bins2, chrName)
    assert sub_matrix2.size > 0, "Contact matrix 2 of chromosome %s is empty" % (chrName)
    assert sub_matrix2.shape[0] == sub_matrix2.shape[1],\
       "Contact matrix 2 of chromosome %s is not square" % (chrName)
    assert sub_matrix1.shape == sub_matrix2.shape,\
        "Contact matrices of chromosome %s have different input shape" % (chrName)

    # Trim diagonals, saves on memory
    n_diags = sub_matrix1.shape[0] if d_max < 0 else min(d_max, sub_matrix1.shape[0])
    matrix1 = trimDiags(sub_matrix1, n_diags, False)
    matrix2 = trimDiags(sub_matrix2, n_diags, False)
    del sub_matrix1
    del sub_matrix2

    # Normalizing by contacts
    matrix1 = matrix1.astype(float) / total_contacts1
    matrix2 = matrix2.astype(float) / total_contacts2

    # Mean filtering, still done to maintain structures
    matrix1 = meanFilterSparse(matrix1, h)
    matrix2 = meanFilterSparse(matrix2, h)

    # Get csr_matrix
    matrix1_diags = upperDiagCsr(matrix1, n_diags)
    matrix2_diags = upperDiagCsr(matrix2, n_diags)

    return matrix1_diags, matrix2_diags

def distance_weights(matrix: np.ndarray, function: str = 'linear', d0: int = 0, slope: float = 0.2,
                     alpha: float = 1.0, beta: float = 1.5):
    '''
        OUTDATED FUNCTION, USE dist_weights INSTEAD
        
        Takes a symmetrical numpy array and applies weighting based on distance from
        the major diagonal. Has three different weighting options (linear, power-law
        and exponential).

        Parameters
        ----------
        matrix : np.ndarray
            Symmetrical matrix.
        function : str  
            Type of weighting funciton to use (str).
            Options: 'linear', 'power' and 'sigmoid'.
            Default: 'linear'
        d0 : int
            Number of diagonals from the major diagonal to surpress.
            Default = 0.
        slope : float
            The slope of the linear function.
            Default = 0.2
        alpha : float
            Steepness of the sigmoid for the sigmoid function.
            Default = 1.0.
        beta : float       
            Power exponent for the power function.
            Default = 1.5.

        Returns
        -------
        dist_matrix : np.ndarray
            Weight adjusted matrix.
    '''
    # TODO: refine parameters (and distance scaling) for real contact maps.

    # Ensure square matrix
    height = matrix.shape[0]
    width = matrix.shape[1]
    ndim = len(matrix.shape)

    if height != width:
        raise ValueError(f'Input matrix not square ({height}x{width})')
    if ndim != 2:
        raise ValueError(f'Input matrix is not 2D (number of dimensions: {ndim})')

    # Extract distances from diagonal
    i = np.arange(height)[:, None]
    j = np.arange(height)[None, :]
    d = np.abs(i + j - height)

    # Weight functions
    if function == 'linear':
        weights = np.maximum(0, slope*d - d0)
    elif function == 'power':
        weights = np.maximum(0, d - d0) ** beta
    elif function == 'sigmoid':
        weights = 1 / (1 + np.exp(-alpha * (d - d0)))
        weights[d <= d0] = 0
    else:
        raise ValueError(f'Invalid weighting function provided: {function}')
    
    # Calculation
    dist_matrix = matrix * weights
    
    return dist_matrix

def diag_dist(matrix: np.ndarray):
    '''
        Takes a symmetrical matrix and calculates the distance of each point 
        from the major diagonal.

        Parameters
        ----------
        matrix : np.ndarray
            Symmetrical matrix.
        
        Returns
        -------
        diag_dists : np.ndarray
            Matrix giving the distance from the major diagonal
            for each point in the input matrix.
    '''
    # Pulling height, width and dimensions
    h = matrix.shape[0]
    w = matrix.shape[1]
    ndim = len(matrix.shape)

    # Ensuring proper formatting
    if h != w:
        raise ValueError(f'Input matrix not square ({h}x{w})')
    if ndim != 2:
        raise ValueError(f'Input matrix is not 2D (number of dimensions: {ndim})')
    
    # Extracting distances from major diagonal
    i = np.arange(h)[:, None]
    j = np.arange(h)[None, :]
    diag_dists = np.abs(i - j)

    return diag_dists

def dist_weights(matrix: np.ndarray, d0: int=0, d1: int=1, gamma: float=1.0):
    '''
        Takes a symmetrical matrix and returns a matrix with multiplication 
        factors to scale with distance from the major diagonal using a linear function.

        Parameters
        ----------
        matrix : np.ndarray
            Symmetrical matrix.
        d0 : int
            The number of diagonals starting from the major diagonal that
            are ignored.
            Default = 1.
        d1 : int
            The number of diagonals starting from the maximum distance diagonal
            that are scaled to x1.0.
            Default = 1
        gamma : float
            Variable that controls the rate at which the ramp scales from x0.0
            to x1.0. Larger than 1.0 gives a slower rise, while smaller gives a
            faster rise.
            Default = 1.0.

        Returns
        -------
        mult_matrix : np.ndarray
            Matrix with scaled data, from x0.0 to x1.0.
        weights : np.ndarray
            Matrix with used weights for each index.
    '''
    n = matrix.shape[0]

    # Checks for arguments
    if d0 < 0:
        raise ValueError(f'd0 must be higher than 0 (currently: d0={d0}).')
    if d0 >= (n-d1):
        raise ValueError(f'd0 must be smaller than n-d1 (currently: d0={d0}, n-d1={n-d1}).')
    if d1 > n:
        raise ValueError(f'd1 must be smaller than the number of diagonals (currently: d1={d1}, n={n}).')
    if gamma <= 0 :
        raise ValueError(f'gamma must be larger than 0 (currently: gamma={gamma})')
    
    # Calculate diagonal number for d1
    d1 = n - d1

    # Compute distance matrix
    dists = diag_dist(matrix)

    # Compute weights
    weights = np.zeros_like(dists, dtype=float)
    ramp = (dists - d0) / (d1 - d0)
    weights = np.clip(ramp, 0.0, 1.0) ** gamma

    mult_matrix = matrix * weights

    return mult_matrix

def hic_mask_threshold(matrix : np.ndarray, cut_off : float, type : str = "raw",
                       norm : str = "none"):
    '''
        Returns the values and corresponding coordinates of a Hi-C
        matrix that are above a given threshold (raw of percentile based).

        Parameters
        ----------
        matrix : np.ndarray
            Symmetrical matrix corresponding to a Hi-C map.
        value : float
            The value used to determine the threshold.
        cut_off : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        
        Returns
        -------
        matrix_values : np.ndarray
            A list of values in the input matrix above the threshold.
        matrix_coords : np.ndarray
            A list of coordinates for all values in matrix_values.
    '''
    types = ["raw", "percentile"]
    norms = ["none", "max", "sum"]
    
    # Errors
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Provided matrix is not symmetrical")
    if type not in types:
        raise ValueError(f"Provided threshold type is invalid: {type}")
    if norm not in norms:
        raise ValueError(f"Provided normalization type is invalid: {norm}")
    
    # Extracting raw value for percentile
    if type == "percentile" and cut_off != 0:                       # Don't use percentile if cut_off = 0
        cut_off = np.percentile(matrix[matrix > 0], cut_off)        # Ensure percentile of only non-zero values
    
    # Extracting points
    matrix_mask = np.triu(matrix, k=1) > cut_off
    matrix_coords = np.column_stack(np.where(matrix_mask))
    matrix_values = matrix[matrix_mask]

    # Applying normalization
    if norm == "max":
        matrix_values = matrix_values / matrix_values.max()
    if norm == "sum":
        matrix_values = matrix_values / matrix_values.sum()

    return matrix_values, matrix_coords

def hic_calculate_deviance(
        coolers : list[cooler.Cooler],
        chrom : str
    ):

    '''
        Takes a list of coolers with identical structure and calculates
        the deviance for each bin.

        Adapted from scry (https://github.com/kstreet13/scry)

        Parameters
        ----------
        coolers : list[cooler.Cooler]
            List of coolers to consider
        chrom : str
            Chromosome to consider.

        Returns
        -------
        deviance : pd.DataFrame
            Dataframe describing deviance per genomic coordinate,
            order in a descending order.
    '''
    # Compile reads
    compiled_reads = compile_hic_reads(coolers, chrom)
    
    # Convert to numpy for faster calculations
    X = compiled_reads.to_numpy(dtype=np.float64)

    # Compute sums per cell
    sz = X.sum(axis=0)

    # Constructing saturated term
    P = X / sz

    logP = np.zeros_like(P)
    mask = P > 0
    logP[mask] = np.log(P[mask])

    log1P = np.log1p(-P)

    ll_sat = np.sum(
        X * logP + (sz - X) * log1P,
        axis=1
    )

    # Constructing null term
    sz_sum = np.sum(sz)
    feature_sums = X.sum(axis=1)

    p = feature_sums / sz_sum 
    
    logp = np.zeros_like(p)
    mask = p > 0
    logp[mask] = np.log(p[mask])

    log1p = np.log1p(-p)

    ll_null = feature_sums * logp + (sz_sum - feature_sums) * log1p

    # Compute deviance
    deviance_np = 2 * (ll_sat - ll_null)
    deviance_np[np.isnan(deviance_np)] = 0.0

    # Convert to dataframe
    deviance = pd.DataFrame(deviance_np, index=compiled_reads.index, columns=["deviances"])

    # Ordering by deviance
    deviance = deviance.sort_values(by=["deviances"], ascending=False)

    return deviance

def hic_extract_weights(
        map : np.ndarray,
        coords : np.ndarray  
    ):
    '''
        Returns the weigths (read counts) from a given map representing
        (a part of) a chromosome contact matrix.

        Parameters
        ----------
        map : np.ndarray
            (Part of) a chromosome contact matrix.
        coords : list[tuple[int, int]]
            Array of coordinates.
        
        Returns
        -------
        values : list[float]
            List of values associated with the provided coordinates,
            in the same order as coords.
    '''
    values = map[coords[:, 0], coords[:, 1]]

    return values

def hic_ot_source_to_targets(source: np.ndarray, targets: list[np.ndarray], cut_off: float = 0, 
                             type: str = 'raw', norm: str = 'none' , unbalanced: float = None,
                             reg: float = None, reg_type: str = "entropy", return_size: bool = False):
    '''
        Takes a source matrix and list of target matrices and performs optimal
        transport from the source to all designated target matrices.

        Parameters
        ----------
        source : np.ndarray
            The source matrix
        targets : list[np.ndarray]
            List of target matrices
        cut_off : float
            The cut-off value for setting a threshold in the source and target datasets.
            Default = 0.
        type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "entropy"
        return_size : bool
            Determines if the size of the cost matrix is returned with the function.

        Returns
        -------
        ot_values : list[float]
            A list ot values when transporting the source to the targets.
        avg_m_size : float
            The average number of elements in M.
            Only returned if return_size = True.
    '''

    # Extracting values and OT
        # Source
    source_values, source_coords = hic_mask_threshold(source, cut_off, type, norm)

        # Targets
    ot_values = []
    if return_size:
        m_sizes = []

    for target in targets:
            # Extracting values
        values, coords = hic_mask_threshold(target, cut_off, type, norm)
        
            # Cost matrix
        M = dist(source_coords, coords)
        if return_size:
            m_sizes.append(M.shape[0] * M.shape[1])

            # OT
        if unbalanced == None:
            values = values * (source_values.sum() / values.sum()) # Sum equalizing if exact
            ot_value = solve(M, source_values, values, reg=reg, reg_type=reg_type).value
        else:
            ot_value = solve(M, source_values, values, unbalanced=unbalanced, reg=reg, reg_type=reg_type).value
 
        ot_values.append(ot_value)    

    if return_size:
        return ot_values, np.average(m_sizes)
    
    return ot_values

def hic_ot(source: np.ndarray, target: np.ndarray, thres: float = 0, 
            thres_type: str='raw', norm: str='none' , unbalanced: float|None=None,
            reg: float|None=None, reg_type: str="entropy") -> float:
    '''
        Performs OT with the specified parameters from a source
        hic contact map to a target hic contact map.

        Parameters
        ----------
        source : np.ndarray
            The source matrix
        target : np.ndarray
            The target matrix
        thres : float
            The cut-off value for setting a threshold in the source and target datasets.
            Default = 0.
        thres_type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "entropy"

        Returns
        -------
        ot_value : float
            The OT loss value for the OT solution. 
    '''
    # Masking and thresholding
    source_values, source_coords = hic_mask_threshold(source, thres, thres_type, norm)
    target_values, target_coords = hic_mask_threshold(target, thres, thres_type, norm)

    # Computing cost matrix
    M = dist(source_coords, target_coords)
    M = M / M.max()

    # OT
    if unbalanced is None:
        #target_values = target_values * (source_values.sum() / target_values.sum()) # Sum equalizing if exact
        ot_value = solve(M, source_values, target_values, reg=reg, reg_type=reg_type).value
    else:
        ot_value = solve(M, source_values, target_values, unbalanced=unbalanced, reg=reg, reg_type=reg_type).value

    return ot_value

def hic_ot_bulk(sources: list[np.ndarray], targets: list[np.ndarray], thres: float = 0, 
                thres_type: str='raw', norm: str='none' , unbalanced: float|None=None,
                reg: float|None=None, reg_type: str="entropy") -> list[list[float]]:
    '''
        Performs OT with the specified parameters from source
        hic contact maps to target hic contact maps.

        Parameters
        ----------
        sources : list[np.ndarray]
            List of source maps
        targets : list[np.ndarray]
            List of target maps
        thres : float
            The cut-off value for setting a threshold in the source and target datasets.
            Default = 0.
        thres_type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "entropy"

        Returns
        -------
        ot_values_list : list[list[float]]
            Lists of OT values. Each entry in the parent list
            corresponds to the indices of the source matrices.
    '''
    # Masking and thresholding
    masked_sources_values = []
    masked_sources_coords = []
    masked_targets_values = []
    masked_targets_coords = []

    for source in sources:
        values, coords = hic_mask_threshold(source, thres, thres_type, norm)
        masked_sources_values.append(values)
        masked_sources_coords.append(coords)

    for target in targets:
        values, coords = hic_mask_threshold(target, thres, thres_type, norm)
        masked_targets_values.append(values)
        masked_targets_coords.append(coords)
    
    # OT
    ot_values_list = []

    # Go through all sources
    for source_values, source_coords in zip(masked_sources_values, masked_sources_coords):
        ot_values = []

        # Go through all targets
        for target_values, target_coords in zip (masked_targets_values, masked_targets_coords):
            # Cost matrix
            M = dist(source_coords, target_coords)
            #M = M / M.max()

            # OT
            if unbalanced is None:
                target_values = target_values * (source_values.sum() / target_values.sum()) # Sum equalizing if exact
                ot_value = solve(M, source_values, target_values, reg=reg, reg_type=reg_type).value
            else:
                ot_value = solve(M, source_values, target_values, unbalanced=unbalanced, reg=reg, reg_type=reg_type).value
            
            # Append to list
            ot_values.append(ot_value)
        
        # Append to results list
        ot_values_list.append(ot_values)

    return ot_values_list

def hic_ot_bulk_clr(sources: list[cooler.Cooler], targets: list[cooler.Cooler], chrom: str, res: int|None=None, 
                    thres: float = 0, thres_type: str='raw', norm: str='none' , unbalanced: float|None=None,
                    reg: float|None=None, reg_type: str="kl") -> pd.DataFrame:
    '''
        Performs OT with the specified parameters from source
        hic contact maps to target hic contact maps. Inputs are coolers
        to prevent double calculations.

        Parameters
        ----------
        sources : list[cooler.Cooler]
            List of source coolers.
        targets : list[cooler.Cooler]
            List of target coolers.
        chrom : str
            Chromosome to compare.
        res : int
            Resolution to bin data to.
            Default = None, uses native resolution.
        thres : float
            The cut-off value for setting a threshold in the source and target datasets.
            Default = 0.
        thres_type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "kl"

        Returns
        -------
        ot_results : pd.DataFrame
            OT results in a dataframe with cooler names as
            row and column indices.
    '''
    # Getting chromosome and cooler names
    source_chrs = []
    source_names = []

    for source in sources:
        source_chrs.append(fetch_region(source, chrom))
        source_names.append(get_cool_name(source))
    
    target_chrs = []
    target_names = []

    for target in targets:
        target_chrs.append(fetch_region(target, chrom))
        target_names.append(get_cool_name(target))

    # Binning
    if res is not None:
        sources_binned = []
        for source in source_chrs:
            sources_binned.append(bin_contact_map(source, res))
        
        targets_binned = []
        for target in target_chrs:
            targets_binned.append(bin_contact_map(target, res))
    else:
        sources_binned = source_chrs
        targets_binned = target_chrs

    del source_chrs
    del target_chrs

    # Generating data
        # Create empty dict
    ot_results = {}

        # Fetch all source and target coords and values
    masked_sources_values = []
    masked_sources_coords = []
    masked_targets_values = []
    masked_targets_coords = []

    for source in sources_binned:
        values, coords = hic_mask_threshold(source, thres, thres_type, norm)
        masked_sources_values.append(values)
        masked_sources_coords.append(coords)

    for target in targets_binned:
        values, coords = hic_mask_threshold(target, thres, thres_type, norm)
        masked_targets_values.append(values)
        masked_targets_coords.append(coords)

        # Loop through all sources (coords, values, names)
    for i, s_name in enumerate(source_names):
        # Loop through all targets (coords, values, names)
        for j, t_name in enumerate(target_names):
            print(f"Starting OT ({i*len(target_names)+j}/{len(target_names)**2-1})")
            # Key for dict
            key = tuple(sorted((s_name, t_name)))

            # Check if the same and assign 0
            if s_name == t_name:
                ot_value = 0
            # Check if comparison has already been performed
            elif key in ot_results:
                ot_value = ot_results[key]
            # Perform compariuson
            else:
                # Cost matrix
                M = dist(masked_sources_coords[i], masked_targets_coords[j])

                # OT
                if unbalanced is None:
                    target_values = target_values * (masked_sources_values[i].sum() / masked_sources_values[j].sum()) # Sum equalizing if exact
                    ot_value = solve(M, masked_sources_values[i], masked_targets_values[j], reg=reg, reg_type=reg_type).value
                else:
                    ot_value = solve(M, masked_sources_values[i], masked_targets_values[j], unbalanced=unbalanced, reg=reg, reg_type=reg_type).value
                
            # Add to dict
            ot_results[key] = ot_value
    
    # Constructing DataFrame
    ot_values = pd.DataFrame(
        index=source_names,
        columns=target_names,
        data=np.nan
    )

    for (a, b), ot_value in ot_results.items():
        if a in ot_values.index and b in ot_values.columns:
            ot_values.loc[a, b] = ot_value
        if b in ot_values.index and a in ot_values.columns:
            ot_values.loc[b, a] = ot_value

    return ot_values

def hic_ot_bulk_deviance(
        coolers: list[cooler.Cooler], 
        chrom: str,
        top_contacts : int, 
        norm: str='none', 
        unbalanced: float|None=None,
        reg: float|None=None, 
        reg_type: str="kl"
        ) -> pd.DataFrame:
    '''
        Performs pairwise OT with the specified parameters on the
        provided coolers. Inputs are coolers to prevent double calculations. 
        Selects contacts to consider based on deviance.

        TODO:
            - Add binning.
            - Add error/warning if maps are not of equal size.

        Parameters
        ----------
        coolers : list[cooler.Cooler]
            List of coolers to compare.
        chrom : str
            Chromosome to compare.
        top_contacts : int
            Top number of contacts to consider, based on deviance.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "kl"

        Returns
        -------
        ot_results : pd.DataFrame
            OT results in a dataframe with cooler names as
            row and column indices.
    '''
    # Errors
    if norm not in ["none", "sum", "max"]:
        raise ValueError(f"Invalid normalization type: {norm}")

    # Getting chromosome and cooler names
    chrs = []
    names = []

    for cooler in coolers:
        chrs.append(fetch_region(cooler, chrom))
        names.append(get_cool_name(cooler))
    
    # Select top contacts
    deviance = hic_calculate_deviance(coolers, chrom).head(top_contacts)
    coords = np.asarray([list(ij) for ij in deviance.index.values])
    del deviance

    # Defining cost matrix
    M = dist(coords, coords)

    # Extracting weights
    values = []
    eps = 1e-15
    for chr_map in chrs:
        weights = hic_extract_weights(chr_map, coords)
        weights = np.maximum(weights, eps) # 0 values break OT, may introduce bias.
        if norm == "max":
            weights = weights / np.max(weights)
        if norm == "sum":
            weights = weights / np.sum(weights)
        values.append(weights)
    
    del coords

    # OT
        # Create empty dict
    ot_results = {}

        # Loop through all sources (coords, values, names)
    for i, s_name in enumerate(names):
        # Loop through all targets (coords, values, names)
        for j, t_name in enumerate(names):
            print(f"Starting OT ({i*len(names)+j}/{len(names)**2-1})")

            # Key for dict
            key = tuple(sorted((s_name, t_name)))

            # Check if the same and assign 0
            if s_name == t_name:
                ot_value = 0
            # Check if comparison has already been performed
            elif key in ot_results:
                ot_value = ot_results[key]
            # Perform compariuson
            else:
                if unbalanced is None:
                    if norm != 'sum':
                        raise ValueError("Exact OT requires sum normalization.")
                    ot_value = solve(M, values[i], values[j], reg=reg, reg_type=reg_type).value
                else:
                    ot_value = solve(M, values[i], values[j], unbalanced=unbalanced, reg=reg, reg_type=reg_type).value
                
            # Add to dict
            ot_results[key] = ot_value
    
    # Constructing DataFrame
    ot_values = pd.DataFrame(
        index=names,
        columns=names,
        data=np.nan
    )

    for (a, b), ot_value in ot_results.items():
        if a in ot_values.index and b in ot_values.columns:
            ot_values.loc[a, b] = ot_value
        if b in ot_values.index and a in ot_values.columns:
            ot_values.loc[b, a] = ot_value

    return ot_values

def hic_ot_bulk_deviance_parallel(
        coolers: list[cooler.Cooler], 
        chrom: str,
        top_contacts : int, 
        norm: str='none', 
        unbalanced: float|None=None,
        reg: float|None=None, 
        reg_type: str="kl"
    ) -> pd.DataFrame:
    '''
        Performs pairwise OT with the specified parameters on the
        provided coolers. Inputs are coolers to prevent double calculations. 
        Selects contacts to consider based on deviance. Uses parallelization
        to speed up calculations.

        TODO:
            - Add binning.
            - Add error/warning if maps are not of equal size.

        Parameters
        ----------
        coolers : list[cooler.Cooler]
            List of coolers to compare.
        chrom : str
            Chromosome to compare.
        top_contacts : int
            Top number of contacts to consider, based on deviance.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "kl"

        Returns
        -------
        ot_results : pd.DataFrame
            OT results in a dataframe with cooler names as
            row and column indices.
    '''
    # Errors
    if norm not in ["none", "sum", "max"]:
        raise ValueError(f"Invalid normalization type: {norm}")

    # Getting chromosome and cooler names
    chrs = []
    names = []

    for cooler in coolers:
        chrs.append(fetch_region(cooler, chrom))
        names.append(get_cool_name(cooler))
    
    # Select top contacts
    deviance = hic_calculate_deviance(coolers, chrom).head(top_contacts)
    coords = np.asarray([list(ij) for ij in deviance.index.values])
    del deviance

    # Defining cost matrix
    M = dist(coords, coords)

    # Extracting weights
    values = []
    eps = 1e-15
    for chr_map in chrs:
        weights = hic_extract_weights(chr_map, coords)
        weights = np.maximum(weights, eps) # 0 values break OT, may introduce bias.
        values.append(weights)
    
    values = np.vstack(values)
    if norm == "max":
        values = values / values.max(axis=1, keepdims=True)
    if norm == "sum":
        values = values / values.sum(axis=1, keepdims=True)

    del coords

    # Parallelized OT
    pairs = combinations(range(len(names)), 2)

        # Pre-configured partial function to speed up calls
    if unbalanced is None:
        if norm != 'sum':
            raise ValueError("Balanced OT requires sum normalization.")
        ot_solver = partial(
            solve,
            M,
            reg=reg,
            reg_type=reg_type
        )
    else:
        ot_solver = partial(
            solve,
            M,
            unbalanced=unbalanced,
            reg=reg,
            reg_type=reg_type
        )

        # Function for single comparison
    def compute_pair(pair):
        i, j = pair
        return i, j, ot_solver(values[i], values[j]).value

        # Compute OT losses parallelized
    results = Parallel(n_jobs=-1, batch_size='auto')(
        delayed(compute_pair)(pair) 
        for pair in tqdm(pairs, total=len(names)*(len(names)-1)//2)
    )

    ot_results = pd.DataFrame(index=names, columns=names, data=0.0)

    for i, j, val in results:
        ot_results.iloc[i, j] = val
        ot_results.iloc[j, i] = val

    return ot_results

def hic_ot_bulk_threshold_parallel(
        coolers: list[cooler.Cooler], 
        chrom: str,
        thres: float=0.0,
        thres_type: str="raw",
        norm: str='none', 
        unbalanced: float|None=None,
        reg: float|None=None, 
        reg_type: str="kl" 
    ) -> pd.DataFrame:
    '''
        Performs pairwise OT with the specified parameters on the
        provided coolers. Inputs are coolers to prevent double calculations. 
        Selects contacts to consider based on thresholding. Uses parallelization
        to speed up calculations.

        TODO:
            - Add binning.
            - Add error/warning if maps are not of equal size.

        Parameters
        ----------
        coolers : list[cooler.Cooler]
            List of coolers to compare.
        chrom : str
            Chromosome to compare.
        thres : float
            The cut-off value for setting a threshold in the source and target datasets.
            Default = 0.
        thres_type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "kl"

        Returns
        -------
        ot_results : pd.DataFrame
            OT results in a dataframe with cooler names as
            row and column indices.
    '''
    # Getting chromosome and cooler names
    chrs = []
    names = []

    for cooler in coolers:
        chrs.append(fetch_region(cooler, chrom))
        names.append(get_cool_name(cooler))

    # Fetching relevant data
        # Fetch all source and target coords and values
    masked_values = []
    masked_coords = []

    for chr_map in chrs:
        values, coords = hic_mask_threshold(chr_map, thres, thres_type, norm)
        masked_values.append(np.asarray(values))
        masked_coords.append(np.asarray(coords))
    
    # Parallelized OT
    pairs = combinations(range(len(names)), 2)

        # Pre-configured partial function to speed up calls
    if unbalanced is None:
        if norm != 'sum':
            raise ValueError("Balanced OT requires sum normalization.")
        ot_solver = partial(
            solve,
            reg=reg,
            reg_type=reg_type
        )
    else:
        ot_solver = partial(
            solve,
            unbalanced=unbalanced,
            reg=reg,
            reg_type=reg_type
        )

        # Function for single comparison
    def compute_pair(pair):
        i, j = pair
        M = dist(masked_coords[i], masked_coords[j])
        return i, j, ot_solver(M, masked_values[i], masked_values[j]).value

        # Compute OT losses parallelized
    results = Parallel(n_jobs=-1, batch_size='auto')(
        delayed(compute_pair)(pair) 
        for pair in tqdm(pairs, total=len(names)*(len(names)-1)//2)
    )

    ot_results = pd.DataFrame(index=names, columns=names, data=0.0)

    for i, j, val in results:
        ot_results.iloc[i, j] = val
        ot_results.iloc[j, i] = val

    return ot_results

def hic_ot_optim(
        coolers: list[cooler.Cooler], 
        chrom: str,
        selection: str,
        norm: str='none', 
        unbalanced: float|None=None,
        reg: float|None=None, 
        reg_type: str="kl",
        top_contacts: int|None=None, 
        thres: float|None=None,
        thres_type: str|None=None
    ) -> pd.DataFrame:
    '''
        Performs pairwise OT with the specified parameters on the
        provided coolers. Inputs are coolers to prevent double calculations. 
        Selects contacts to consider based on either deviance or a set threshold.

        TODO:
            - Add binning.
            - Add error/warning if maps are not of equal size.

        Parameters
        ----------
        coolers : list[cooler.Cooler]
            List of coolers to compare.
        chrom : str
            Chromosome to compare.
        selection: str
            Type of selection criteria to select contacts for OT.
            Options include "deviance" and "threshold".
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        unbalanced : float
            Unbalanced parameter if UOT should be used.
            Default = None, uses balanced OT instead.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "kl"
        top_contacts : int
            Top number of contacts to consider, based on deviance.
            Default = None, disables deviance selection.
        thres : float
            The cut-off value for setting a threshold.
            Default = None, disables threshold selection
        thres_type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = None, disables threshold selection.

        Returns
        -------
        ot_results : pd.DataFrame
            OT results in a dataframe with cooler names as
            row and column indices.
    '''
    # Errors
        # Normalization
    if norm not in ["none", "sum", "max"]:
        raise ValueError(f"Invalid normalization type: {norm}")
    
    if selection == "deviance":
        ot_results = hic_ot_bulk_deviance_parallel(
            coolers,
            chrom,
            top_contacts,
            norm,
            unbalanced,
            reg,
            reg_type
        )
    elif selection == "threshold":
        ot_results = hic_ot_bulk_threshold_parallel(
            coolers,
            chrom,
            thres,
            thres_type,
            norm,
            unbalanced,
            reg,
            reg_type
        )
    else:
        raise ValueError(f"Invalid selection criterium: {selection}")
    return ot_results

def batch_ot(sources: list[np.ndarray], targets: list[np.ndarray], thres: float = 0, 
                thres_type: str = 'raw', norm: str = 'none', reg: float = None, reg_type: str = "entropy"):
    '''
        Takes a list of source matrices and a list of target matrices, performs
        OT from these sources to targets, and returns the OT loss values.

        CURRENTLY NOT WORKING, ALSO CANNOT SUPPORT UNBALANCED
        MAY NOT EVER WORK, AS MATRICES OF IDENTICAL SHAPE ARE REQUIRED
    Parameters
        ----------
        sources : list[np.ndarray]
            List of source matrices
        targets : list[np.ndarray]
            List of target matrices
        thres : float
            The threshold value for setting a threshold in the source and target datasets.
            Default = 0.
        thres_type : str
            The kind of threshold to be set.
            Options include 'raw' and 'percentile'.
            Default = 'raw'.
        norm : str
            Type of normalization to use. Options include "none",
            "max" and "sum".
            Default = 'none'.
        reg : float
            Entropic regularization term.
            Default = None, uses exact OT instead.
        reg_type : str
            Type of regularization used.
            Default = "entropy"

        Returns
        -------
        ot_values : list[float]
            A list ot values when transporting the source to the targets.
    '''
    # Performing masking and coord extraction
    src_val_list = []
    src_coord_list = []
    for source in sources:
        vals, coords = hic_mask_threshold(source, thres, thres_type, norm)
        src_val_list.append(vals)
        src_coord_list.append(coords)
    src_val_list = np.stack(src_val_list)
    src_coord_list = np.stack(src_coord_list)

    trgt_val_list = []
    trgt_coord_list = []
    for target in targets:
        vals, coords = hic_mask_threshold(target, thres, thres_type, norm)
        trgt_val_list.append(vals)
        trgt_coord_list.append(coords)
    trgt_val_list = np.stack(src_val_list)
    trgt_coord_list = np.stack(src_coord_list)

    # Calculating cost matrices
    M_batch = dist_batch(src_coord_list, trgt_coord_list)

    # Calculating OT losses
    ot_values = solve_batch(
        M=M_batch,
        reg=reg,
        reg_type=reg_type,
        a=src_val_list,
        b=trgt_val_list,
        ).value

    return ot_values

def hic_ot_chr_value(src_clr: np.ndarray, trg_clr: np.ndarray, unbalanced: float = 5e-2,
           reg: float = 1e-1):
    '''
        NO LONGER MAINTAINED, USE OTHER FUNCTIONS
        
        Takes a source and target data array representing a contact map
        for a specific chromosome and performs exact/Sinkhorn unbalanced
        optimal transport based on the arguments provided.

        Parameters
        ----------
        src_clr : np.ndarray    
            Source array.
        trg_clr : np.ndarray    
            Target array.
        unbalanced : float
            Unbalanced parameter for unbalanced optimal transport.
            Default = 5e-2.
        reg : float        
            Sinkhorn regularization term.
            Default = 1e-1.
        
        Returns
        -------
        ot_value : np.float64
            Unbalanced optimal transport value calculated by POT.
    '''
    # Checks for the input matrices
        # Ensure same shapes
    if src_clr.shape != trg_clr.shape:
        raise ValueError(f'Input matrices are of differing sizes')

        # Ensure square matrix
    height = src_clr.shape[0]
    width = src_clr.shape[1]
    ndim = len(src_clr.shape)

    if height != width:
        raise ValueError(f'Input matrices are not square ({height}x{width})')
    if ndim != 2:
        raise ValueError(f'Input matrices are not 2D (number of dimensions: {ndim})')

    # Defining masks to save on computation time
    mask_src = np.triu(src_clr, k=1) > 0
    mask_trg = np.triu(trg_clr, k=1) > 0
    coords_src = np.column_stack(np.where(mask_src))
    coords_trg = np.column_stack(np.where(mask_trg))

    # Computing cost matrix
    C = dist(coords_src, coords_trg, metric='sqeuclidean')

    # Subsetting nonzero data
    a = src_clr[mask_src]
    b = trg_clr[mask_trg]

    # Unbalanced optimal transport
    ot = solve(C, a, b, unbalanced=unbalanced, reg=reg)
    ot_value = ot.value

    return ot_value