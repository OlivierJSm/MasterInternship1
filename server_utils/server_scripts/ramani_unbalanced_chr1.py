### Setting directories
data_dir = '/mbshome/osmeets/MasterInternship1/Data/ramani_cool_files' # Importing was not working with relative root+'..' approach, so replace with local data
metadata_dir = f"{data_dir}/cell_line_mapping.csv"
### Imports
from hicdatautils import hic_ot_optim, import_cool_dir, subset_clr_data
import pandas as pd

### Importing coolers and metadata
clrs = import_cool_dir(data_dir)
metadata_df = pd.read_csv(metadata_dir)

### Setting parameters
save_dir = '/mbshome/osmeets/MasterInternship1/Generated Data/Server'
count = None
res = 4
thres = 0
reg_m = 0.1

### Subset
print("Starting subsetting\n")
unbalanced_subset = subset_clr_data(
    data_dir,
    metadata_df,
    count,
    iqr_rule = 1.5,
    type_col="cell_line"
)
print("Finished subsetting\n")

### OT
print("Starting OT\n")
unbalanced_results = hic_ot_optim(
    unbalanced_subset,
    "chr1",
    "threshold",
    "max",
    unbalanced=reg_m,
    thres=thres,
    thres_type="percentile",
    rebin_factor=res
)
print("Finished OT\n")

### Saving
unbalanced_results.to_csv(f"{save_dir}/ramani_unbalanced_chr1_count={count}_res={res}_thres={thres}_reg_m={reg_m}.csv")