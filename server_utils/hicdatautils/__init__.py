import os
import numpy as np
from .hicscc import calc_scc_file, avg_scc, avg_scc_weighted, bulk_compare, sample_scc_dist, bulk_compare_ref, scc_compare_matrix, scc_compare_lists, scc_bulk
from .hicot import get_diag_csr, distance_weights, hic_ot_chr_value, hic_mask_threshold, hic_ot_source_to_targets, dist_weights, batch_ot, hic_ot, hic_ot_bulk, hic_ot_bulk_clr, hic_calculate_deviance, hic_extract_weights, hic_ot_bulk_deviance, hic_ot_bulk_deviance_parallel, hic_ot_bulk_threshold_parallel, hic_ot_optim, hic_ot_bulk_threshold_sequential, hic_ot_optim_multi
from .hicgeneral import import_cool, get_chrs, standardize_coolers, standardize_coolers_ref, import_cool_dir, get_cool_name, standardize_coolers_bulk, fetch_region, bin_contact_map, bulk_coarsen, compile_hic_reads, subset_clr_data, rebin_contact_map
from .hictests import generate_contact_map, generate_related_contact_map, visualize_synthetic_maps
from .hictoscc import compare_to_scc, compare_to_scc_df
from .hicdatahandling import long_transform, ot_mass_deviance, ot_mass_threshold
from .hicotvis import generate_clustermap, generate_marginal_plot, generate_umap, compare_metrics, generate_violin_plot, generate_deviance_heatmap

__all__ = ["calc_scc_file", "avg_scc", "avg_scc_weighted", "bulk_compare", "get_chrs", 
           "sample_scc_dist", "get_diag_csr", "import_cool", "standardize_coolers",
           "standardize_coolers_ref", "import_cool_dir", "get_cool_name",
           "standardize_coolers_bulk", "bulk_compare_ref", "fetch_region", 
           "distance_weights", "scc_compare_matrix", 'hic_ot_chr_value',
           "generate_contact_map", "generate_related_contact_map",
           "visualize_synthetic_maps", "bin_contact_map", "hic_mask_threshold",
           "hic_ot_source_to_targets", "compare_to_scc", "compare_to_scc_df",
           "scc_compare_lists", "dist_weights", "batch_ot", "hic_ot", "hic_ot_bulk",
           "hic_ot_bulk_clr", "bulk_coarsen", "scc_bulk", "compile_hic_reads",
           "hic_calculate_deviance", "hic_extract_weights", "hic_ot_bulk_deviance",
           "subset_clr_data", "long_transform", "ot_mass_deviance", "generate_clustermap",
           "generate_marginal_plot", "hic_ot_bulk_deviance_parallel", "hic_ot_bulk_threshold_parallel",
           "hic_ot_optim", "generate_umap", "rebin_contact_map", "hic_ot_bulk_threshold_sequential", 
           "compare_metrics", "generate_violin_plot", "ot_mass_threshold", "generate_deviance_heatmap",
           "hic_ot_optim_multi"]