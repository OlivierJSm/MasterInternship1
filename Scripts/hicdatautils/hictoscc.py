import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

def compare_to_scc(ot_results : list[float], map_names : list[str], scc_scores : pd.DataFrame,
                   threshold : int = None, resolution : int = None, ot_type : str = "balanced", 
                   reg_m: float = None, reg: float = None, plot: bool = False):
    ''' 
        Converts ot results to a pandas dataframe and uses 
        said dataframe to compare ot loss to scc scores.

        Parameters
        ----------
        ot_results : list[float]
            List of ot results orderded like map_names.
        map_names : list[str]
            List of names of each element compared for OT.
        scc_scores : pd.DataFrame
            Pandas dataframe containting scc scores, ordered
            like map_names.
        threshold : int
            The threshold tested. Used for figure title.
            Default = None
        resolution : int
            The resolution tested. Used for figure title.
            Default = None
        ot_type : str
            Type of ot used. Used for figure title.
            Default = "balanced"
        reg_m : float
            Unbalancing parameter used. Used for figure title.
            Default = None.
        reg : float
            Regularization parameter used. Used for figure title.
            Default = None.
        plot : bool
            Determines if the results are plotted.
            Default = False
        
        Returns
        -------
        results_df : pd.DataFrame
            Dataframe of ot results with indices based on map_names.
        rsqr : float
            R^2 values of the fit with SCC scores.
    '''

    # Convert to dataframe
    results_df = pd.DataFrame(ot_results, index=map_names, columns=map_names)

    # SCC comparison regression
    ot_results_flatten = results_df.values.flatten()
    scc_scores_flatten = scc_scores.values.flatten()
    regression = linregress(ot_results_flatten, scc_scores_flatten)
    slope = regression.slope
    intercept = regression.intercept
    rsqr = regression.rvalue**2

    # Plotting SCC comparison
    if plot:
        plt.scatter(ot_results_flatten, scc_scores_flatten, label="Data")
        plt.plot(ot_results_flatten, intercept + slope * ot_results_flatten, color='red',
                label=f'Linear fit (R\u00b2 = {rsqr:.3f})')
        plt.xlabel("OT loss")
        plt.ylabel("SCC score")
        if ot_type.lower() == "balanced":
            if reg == None:
                plt.title(f"{ot_type} OT loss to SCC score (res={resolution}Mb, threshold={threshold})")
            else:
                plt.title(f"{ot_type} OT loss (reg={reg}) to SCC score (res={resolution}Mb, threshold={threshold})")
        else:
            if reg == None:
                plt.title(f"{ot_type} OT loss to SCC score (res={resolution}Mb, threshold={threshold}, reg_m={reg_m})")
            else:
                plt.title(f"{ot_type} OT loss (reg={reg}) to SCC score (res={resolution}Mb, threshold={threshold}, reg_m={reg_m})")
        plt.legend()
        plt.grid(alpha=0.3)

    return results_df, rsqr

def compare_to_scc_df(ot_results : pd.DataFrame, scc_scores : pd.DataFrame,
                   threshold : int = None, resolution : int = None, ot_type : str = "balanced", 
                   reg_m: float = None, reg: float = None, plot: bool = False):
    ''' 
        Uses OT results dataframe to compare ot loss to scc scores.

        Parameters
        ----------
        ot_results : pd.DataFrame
            Pandas dataframe of ot results.
        scc_scores : pd.DataFrame
            Pandas dataframe containting scc scores, ordered
            like ot_results.
        threshold : int
            The threshold tested. Used for figure title.
            Default = None
        resolution : int
            The resolution tested. Used for figure title.
            Default = None
        ot_type : str
            Type of ot used. Used for figure title.
            Default = "balanced"
        reg_m : float
            Unbalancing parameter used. Used for figure title.
            Default = None.
        reg : float
            Regularization parameter used. Used for figure title.
            Default = None.
        plot : bool
            Determines if the results are plotted.
            Default = False
        
        Returns
        -------
        rsqr : float
            R^2 values of the fit with SCC scores.
    '''

    # SCC comparison regression
    ot_results_flatten = ot_results.values.flatten()
    scc_scores_flatten = scc_scores.values.flatten()
    regression = linregress(ot_results_flatten, scc_scores_flatten)
    slope = regression.slope
    intercept = regression.intercept
    rsqr = regression.rvalue**2

    # Plotting SCC comparison
    if plot:
        plt.scatter(ot_results_flatten, scc_scores_flatten, label="Data")
        plt.plot(ot_results_flatten, intercept + slope * ot_results_flatten, color='red',
                label=f'Linear fit (R\u00b2 = {rsqr:.3f})')
        plt.xlabel("OT loss")
        plt.ylabel("SCC score")
        if ot_type.lower() == "balanced":
            if reg == None:
                plt.title(f"{ot_type} OT loss to SCC score (res={resolution}Mb, threshold={threshold})")
            else:
                plt.title(f"{ot_type} OT loss (reg={reg}) to SCC score (res={resolution}Mb, threshold={threshold})")
        else:
            if reg == None:
                plt.title(f"{ot_type} OT loss to SCC score (res={resolution}Mb, threshold={threshold}, reg_m={reg_m})")
            else:
                plt.title(f"{ot_type} OT loss (reg={reg}) to SCC score (res={resolution}Mb, threshold={threshold}, reg_m={reg_m})")
        plt.legend()
        plt.grid(alpha=0.3)

    return rsqr