import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from scipy import stats
import cooler
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from . import get_cool_name

def compare_metrics(
        metric1_data : pd.DataFrame,
        metric2_data : pd.DataFrame,
        metric1_label : str = "Metric 1",
        metric2_label: str = "Metric 2",
        figsize : tuple[int, int] = (7, 6),
        textsize : int = 12,
        ax = None
    ) -> float:
    '''
        Generates a scatterplot with a linear trendline to compare
        two comparison metrics.

        Parameters
        ----------
        metric1_data : pd.DataFrame
            Pairwise comparison matrix.
        metric2_data : pd.DataFrame
            Pairwise comparison matrix, must have identical indexing
            to metric 1.
        metric1_label : str
            Label for metric 1.
            Default = "Metric 1".
        metric2_label : str
            Label for metric 2.
            Default = "Metric 1".
        figsize: tuple[int, int]
            Figure size to use.
            Default = (7, 6)
        textsize : int
            Text size to use.
            Default = 14
        ax
            Axes object to add to.
            Default = None, creates a new figure.

        Returns
        -------
        r**2 : float
            Pearson r2 value between OT and SCC values.
    '''
    # Ensure identical shapes
    if metric1_data.shape != metric2_data.shape:
        raise ValueError("Dataframes are of differing shapes!")

    # Extracting upper triangles
    mask = np.triu(np.ones(metric1_data.shape, dtype=bool))
    metric1_values = metric1_data.values[mask]
    metric2_values = metric2_data.values[mask]

    # Correspondence values
    r, p = stats.pearsonr(metric1_values, metric2_values)

    # Linear fit
    slope, intercept = stats.linregress(metric1_values, metric2_values)[0:2]
    x_line = np.linspace(metric1_values.min(), metric1_values.max(), 200)
    y_line = slope * x_line + intercept

    # Plotting
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

    ax.scatter(
        x=metric1_values,
        y=metric2_values,
        marker='s', 
        c='black'
    )

    ax.plot(
        x_line,
        y_line,
        color='red',
        linestyle=":",
        label=f"Pearson R² = {r**2:.4f})"
    )

    ax.set_xlabel(metric1_label, fontsize=textsize)
    ax.set_ylabel(metric2_label, fontsize=textsize)
    ax.legend(fontsize=textsize)
    plt.tight_layout()

    return r**2

def generate_clustermap(
        ot_data : pd.DataFrame,
        title : str|None = None,
        **kwargs
    ) -> None:
    '''
        Generates a clustermap based on the provided OT data in long
        form.

        Parameters
        ----------
        ot_data : pd.DataFrame
            OT data in long form.
        title : str
            Title of the figure.
            Default = None
        **kwargs
            Arguments to pass to sns.clustermap

        Returns
        -------
        None
    '''
    # Calculating means per cell type
    mean_matrix = ot_data.groupby(
        ['row_cell_type', 'col_cell_type']
    )["value"].mean()

    # Prepping for generating heatmap
    heatmap_data = mean_matrix.unstack()

    # Clustermap
    g = sns.clustermap(
        heatmap_data, 
        cmap='viridis',
        **kwargs)
    if title is not None:
        plt.suptitle(title, fontweight='bold')
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.xaxis.set_ticks_position('bottom')
    g.ax_heatmap.tick_params(axis='x', length=5)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    return

def generate_marginal_plot(
        ot_data : pd.DataFrame,
        mass_df : pd.DataFrame,
        cell_name_col : str = "file_name",
        title : str|None=None
    ) -> None:
    '''
        Generates a marginal plot for the OT data using the mass considered per cell.

        Parameters
        ----------
        ot_data : pd.DataFrame
            OT data in long form.
        mass_df : pd.DataFrame
            Dataframe providing the mass considered per file.
        cell_name_col : str
            Name of the column in the mass dataframe that gives the cell
            name.
        title : str
            Title of the figure.
            Default = None
        
        Returns
        -------
        None

        TODO:
        - Add support for changing figure size.
    '''
    sns.set_theme(style="white")

    # Adding mass data
    mass_map = mass_df.copy()
    mass_map = mass_map.set_index(cell_name_col)['total_mass']

    # Adding row and col masses
    long_df_mass = ot_data.copy()
    long_df_mass['row_mass'] = long_df_mass['row_file'].map(mass_map)
    long_df_mass['col_mass'] = long_df_mass['col_file'].map(mass_map)

    # Adding additional metrics
    long_df_mass['mass_diff'] = abs(long_df_mass["row_mass"] - long_df_mass['col_mass'])
    long_df_mass['log_value'] = np.log10(long_df_mass['value'])

    # Initialize grid
    g = sns.JointGrid(
        data=long_df_mass,
        x="mass_diff",
        y='log_value',
        height=6
    )

    # Joint: light scatter + contour KDE
    g.ax_joint.scatter(
        long_df_mass["mass_diff"],
        long_df_mass['log_value'],
        s=8,
        alpha=0.25
    )

    sns.kdeplot(
        data=long_df_mass,
        x="mass_diff",
        y='log_value',
        ax=g.ax_joint,
        levels=10,
        bw_adjust=0.8,
        color='black'
    )

    # Marginals
    sns.kdeplot(
        data=long_df_mass,
        x="mass_diff",
        ax=g.ax_marg_x,
        fill=True,
        bw_adjust=0.8
    )

    sns.kdeplot(
        data=long_df_mass,
        y='log_value',
        ax=g.ax_marg_y,
        fill=True,
        bw_adjust=0.8
    )

    # Axis labels
    g.set_axis_labels('Mass Difference', 'log(OT Loss)')

    # Title
    if title is not None:
        g.figure.suptitle(title, fontweight='bold')
    g.figure.tight_layout(rect=[0, 0, 1, 0.98])
    return

def generate_umap(
        ot_data: pd.DataFrame,
        metadata: pd.DataFrame,
        title: str|None=None,
        name_col: str = "cell_name",
        type_col: str = "cell_type",
        group_map: dict|None=None,
        palette : dict|None = None,
        ax = None,
        figsize : tuple[int, int]|None=None
    ) -> list :
    '''
        Generates a UMAP based on the provided OT results.

        Parameters
        ----------
        ot_data : pd.DataFrame
            OT data formatted in a pairwise manner.
        metadata : pd.DataFrame
            Metadata providing the cell type/line per file name.
        title : str
            Title for the figure.
            Default = None.
        name_col : str
            Name of the column providing cell names in the metadata.
            Default = "cell_name".
        type_col : str
            Name of the column providing cell types/lines in the metadata.
            Default = "cell_type".
        group_map : dict
            Dictionary with cell types as keys and umbrella cell types as vales. 
            Allows celltypes to be clustered together.
            Default = None
        palette : dict
            Dictonary that maps groups in metadata or in the group_map to specific
            colors.
            Default = None, automatically asigns colors.
        ax
            Axes object to add to.
            Default = None, creates a new figure.
        figsize : tuple[int, int]
            Size of the figure if ax is None.
            Default = None, uses standard figsize.

        Returns
        -------
        vec : list
            UMAP embeddings of each cell in the same order as ot_data.
        
        TODO:
        - Add support for changing text size.
    '''
    ### UMAP ###
    # Converting to ndarray
    ot_ndarray = ot_data.to_numpy()

    # Extracting cell types
    cell_types = []
    for cell in ot_data.index:
        specific_type = metadata.loc[metadata[name_col] == cell, type_col].values[0]
        specific_type = group_map.get(specific_type, specific_type) if group_map else specific_type

        cell_types.append(specific_type)

    # Generating UMAP
    vec = UMAP(n_components=2).fit_transform(ot_ndarray)

    # Figure
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title, fontweight="bold")
    elif title is not None:
        ax.set_title(title)

    if palette is None:
        sns.scatterplot(x=vec[:, 0], y=vec[:, 1], hue=cell_types, ax=ax, linewidth=0, s=20)
    else:
        sns.scatterplot(x=vec[:, 0], y=vec[:, 1], hue=cell_types, palette=palette, ax=ax, linewidth=0, s=20)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles=handles, 
              labels=labels, 
              loc='lower left', 
                )

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ### Barplots for clusters ###
    # Extract clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels_kmeans = kmeans.fit_predict(vec) + 1

    # Dataframe with cluster per original cell name
    cluster_df = pd.DataFrame({
        "cluster": labels_kmeans
    }, index=ot_data.index)

    cluster_df = cluster_df.join(
        metadata.set_index(name_col),
        how="left"
    )

    cluster_df["group"] = cluster_df["cell_type"].map(group_map)

    # Extract counts
    counts = pd.crosstab(cluster_df["cluster"], cluster_df["group"])
    fraction = counts.div(counts.sum(axis=1), axis=0)
    print(fraction)

    # Create inset axes
    ax_bar = inset_axes(ax,
                        width="42%",
                        height="25%",
                        loc="upper right",
                        bbox_to_anchor=(0.3, 0.28, 0.7, 0.7),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    
    counts.plot(
            kind="barh",
            stacked=True,
            color=[palette[c] for c in fraction.columns],
            legend=False,
            ax=ax_bar,
        )
    
    ax_bar.set_xlabel("#Cells")
    ax_bar.set_ylabel("")
    

    ### Cluster labels ###
    for cluster_id in np.unique(labels_kmeans):
        coords = vec[labels_kmeans == cluster_id]
        centroid = coords.mean(axis=0)

        ax.text(centroid[0], centroid[1],
                f"Cluster {str(cluster_id)}",
                fontsize=10,
                weight='bold')

    return vec

def generate_violin_plot(
        clrs : list[cooler.Cooler],
        metadata : pd.DataFrame,
        cell_name_col : str = "cell_name",
        cell_type_col : str = "cell_type",
        palette : dict|None = None,
        order : list[str] | None=None,
        ax = None,
        fig_size : tuple[int, int]|None = None
) -> None:
    '''
        Takes coolers and metadata information to generate violin
        plots describing the total Hi-C contacts per cell per included cell
        type.

        Parameters
        ----------
        clrs : list[cooler.Cooler]
            List of coolers to consider.
        metadata : pd.DataFrame
            Dataframe with cell type information on
            provided coolers.
        cell_name_col : str
            The name of the column with the names of cells
            in the metadata dataframe.
        cell_type_col : str
            The name of the column with the types of cells
            in the metadata dataframe.
        palette : dict
            Dictonary that maps groups in metadata or in the group_map to specific
            colors.
            Default = None, automatically asigns colors.
        order : list[str]
            The order of celltypes in the violin plot.
            Default = None, uses a default order.
        ax :
            Axes object to pass figure to. Creates a new figure
            if left empty.
        fig_size : tuple[int, int]
            Desired figure size if ax is not provided.
    '''
    # Extracting masses
    totals = []
    names = []
    for clr in clrs:
        totals.append(clr.info["sum"])
        names.append(get_cool_name(clr))

        # To df
    totals_df = pd.DataFrame({
        cell_name_col : names,
        "total" : totals
    })

        # Merge with metadata
    totals_df = totals_df.merge(metadata, on=cell_name_col)

    # Constructing figure
    if ax is None:
        if fig_size is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=fig_size)

    sns.violinplot(
        data=totals_df,
        ax=ax,
        x=cell_type_col,
        y="total",
        hue=cell_type_col,
        palette=palette,
        order=order,
        cut=0
    )
    ax.set_ylabel("Counts")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)

    # Adding counts to graphs
    counts = totals_df[cell_type_col].value_counts()
    y_max = totals_df.groupby(cell_type_col)["total"].max()
    y_min, y_max_plot = ax.get_ylim()
    ax.set_ylim(y_min, y_max_plot * 1.05)
    for i, cell_type in enumerate(ax.get_xticklabels()):
        ct = cell_type.get_text()
        ax.text(
            i,
            y_max[ct] * 1.03,
            f"n = {counts[ct]}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontstyle='italic'
        )

    return

def generate_deviance_heatmap(
        deviances : pd.DataFrame,
        colorbar : bool=False,
        title : str|None=None,
        label : str|None=None,
        textsize : int=14,
        ax = None
) -> None:
    '''
        Takes a deviance dataframe generated by
        hicot.hic_calculate_deviance() and visualizes it
        as a heatmap.

        Parameters
        ----------
        deviances : pd.DataFrame
            Deviance dataframe genetated by hicot.hic_calculate_deviance().
        colorbar : bool
            Determines whether to include a colorbar.
            Default = False.
        textsize : int
            Textsize to use.
            Default = 14.
        title : str
            Title to use.
            Default = None, uses no title.
        ax
            Axis to pass figure to.
            Default = None, creates a new figure
    '''
    # Create symmetrical matrix
    dev_reset = deviances.reset_index()
    mirrored = dev_reset.rename(columns={
        'bin_i': 'bin_j',
        'bin_j': 'bin_i'
    })

    # Combine
    combined = pd.concat([dev_reset, mirrored], ignore_index=True)
    combined = combined.drop_duplicates(subset=['bin_i', 'bin_j'])
    combined = combined.set_index(['bin_i', 'bin_j'])

    # Make image
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    heatmap_data = combined['deviances'].unstack(level='bin_j')
    im = ax.imshow(
        heatmap_data[heatmap_data > 0],
        origin='upper',
        cmap='viridis',
        aspect='auto',
    )
    if colorbar:
        plt.colorbar(im, ax=ax, label='Deviance')
    if title is not None:
        ax.set_title(title, size=textsize),
    if label is not None:
        ax.set_ylabel(label, size=textsize)
    ax.set_xticks([])
    ax.set_yticks([])

    return None