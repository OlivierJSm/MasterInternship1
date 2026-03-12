import pandas as pd
import numpy as np
import cooler
from .hicgeneral import fetch_region, get_cool_name
from .hicot import hic_calculate_deviance, hic_extract_weights

def long_transform(
        ot_data: pd.DataFrame, 
        metadata: pd.DataFrame,
        name_col: str = "cell_name",
        type_col: str = "cell_type"
    ) -> pd.DataFrame:
    '''
        Takes a file-to-file comparisons matrix with OT results and
        transforms these results into a long format, also providing
        row and column cell types/lines.

        Parameters
        ----------
        ot_data : pd.DataFrame
            File-to-file format OT results.
        metadata : pd.DataFrame
            Metadata providing the cell type/line per file name.
        name_col : str
            Name of the column providing cell names in the metadata.
            Default = "cell_name".
        type_col : str
            Name of the column providing cell types/lines in the metadata.
            Default = "cell_type".
        
        Returns
        -------
        ot_results_long : pd.DataFrame
            Long format dataframe additionally giving row and column
            cell lines.
    '''
    # Mapping file to type
    type_lookup = metadata.set_index(name_col)[type_col]

    # Long format
    ot_results_long = ot_data.reset_index().rename(columns={"index": "row_file"})
    ot_results_long = ot_results_long.melt(id_vars="row_file",
                        var_name="col_file",
                        value_name="value")
    
    # Mapping celltypes
    ot_results_long["row_cell_type"] = ot_results_long["row_file"].map(type_lookup)
    ot_results_long["col_cell_type"] = ot_results_long["col_file"].map(type_lookup)

    # Inter and intra
    ot_results_long["relationship"] = (
        ot_results_long["row_cell_type"] == ot_results_long["col_cell_type"]
        ).map({True: "intra", False: "inter"})

    return ot_results_long

def ot_mass_deviance(
        clrs: list[cooler.Cooler],
        chr : str,
        top: int,
        norm: str="max"
    ) -> pd.DataFrame:
    '''
        Takes a list of coolers with their corresponding metadata and
        returns a dataframe providing the mass considered in deviance-guided
        OT per cell name.

        Parameters
        ----------
        clrs : list[cooler.Cooler]
            List of coolers considered
        chr : str
            Chromosome to extract mass from.
        top : int
            The top number of contacts considered by deviance-guided
            selection.
        norm : str
            Type of normalization used for OT. Options include "none", "max"
            and "sum".
        
        Returns
        -------
        mass_df : pd.DataFrame
            Dataframe providing the mass considered per file.
    '''
    # Errors
    if norm not in ["none", "sum", "max"]:
        raise ValueError(f"Invalid normalization type: {norm}")

    # Extracting chromosome and names
    chrs = []
    names = []
    for clr in clrs:
        chrs.append(fetch_region(clr, chr))
        names.append(get_cool_name(clr))
    
    # Extracting masses
        # Calculating deviance and fetching coords
    deviance = hic_calculate_deviance(clrs, chr).head(top)
    coords = np.asarray([list(ij) for ij in deviance.index.values])
    del deviance
    
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
        values.append(np.sum(weights))
    
    del coords

    # Converting to dataframe
    mass_df = pd.DataFrame(
        {
            "cell_name" : names,
            "total_mass" : values
        }
    )

    return mass_df