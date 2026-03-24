import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

def generate_clustermap(
        ot_data : pd.DataFrame,
        title : str
    ) -> None:
    '''
        Generates a clustermap based on the provided OT data in long
        form.

        Parameters
        ----------
        ot_data : pd.DataFrame
            OT data in long form.
        title : str
            Title of the figure

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
    sns.clustermap(heatmap_data, annot=True, fmt='.1f', cmap='viridis')
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()

    return

def generate_marginal_plot(
        ot_data : pd.DataFrame,
        mass_df : pd.DataFrame,
        title : str
    ) -> None:
    '''
        Generates a marginal plot for the OT data using the mass considered per cell.

        Parameters
        ----------
        ot_data : pd.DataFrame
            OT data in long form.
        mass_df : pd.DataFrame
            Dataframe providing the mass considered per file.
        title : str
            Title of the figure
        
        Returns
        -------
        None
    '''
    sns.set_theme(style="white")

    # Adding mass data
    mass_map = mass_df.copy()
    mass_map = mass_map.set_index('cell_name')['total_mass']

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
    g.figure.suptitle(title, fontweight='bold')
    g.figure.tight_layout(rect=[0, 0, 1, 0.98])
    return

def generate_umap(
        ot_data: pd.DataFrame,
        metadata: pd.DataFrame,
        title: str,
        name_col: str = "cell_name",
        type_col: str = "cell_type",
        ax = None
    ) -> None:
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
        name_col : str
            Name of the column providing cell names in the metadata.
            Default = "cell_name".
        type_col : str
            Name of the column providing cell types/lines in the metadata.
            Default = "cell_type".
        ax
            Axes object to add to.
            Default = None, creates a new figure.
    '''
    # Converting to ndarray
    ot_ndarray = ot_data.to_numpy()

    # Extracting cell types
    cell_types = []
    for cell in ot_data.index:
        specific_type = metadata[metadata[name_col] == cell][type_col].values[0]

    # Generating UMAP
    vec = UMAP(n_components=2).fit_transform(ot_ndarray)
    xlab = "UMAP 1"
    ylab = "UMAP 2"

    # Figure
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(title)


    sns.scatterplot(x=vec[:, 0], y=vec[:, 1], hue=cell_types, ax=ax, linewidth=0, s=20)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return