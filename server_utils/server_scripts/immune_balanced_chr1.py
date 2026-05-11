### Setting directories
data_dir = '/mbshome/osmeets/MasterInternship1/Data/sc_data' # Importing was not working with relative root+'..' approach, so replace with local data
metadata_dir = f"{data_dir}/cell_types.tsv"
### Imports
from hicdatautils import hic_ot_optim, import_cool_dir, subset_clr_data
import pandas as pd

### Importing coolers and metadata
clrs = import_cool_dir(data_dir)
metadata_df = pd.read_csv(metadata_dir, sep="\t")

### Setting parameters
save_dir = '/mbshome/osmeets/MasterInternship1/Generated Data/Server'
count = 100
res = 1
thres = 95

### Subset
print("Starting subsetting\n")
balanced_subset = subset_clr_data(
    data_dir,
    metadata_df,
    count,
    iqr_rule = 1.5
)
print("Finished subsetting\n")

### OT
print("Starting OT\n")
balanced_results = hic_ot_optim(
    balanced_subset,
    "chr1",
    "threshold",
    "sum",
    thres=thres,
    thres_type="percentile",
    rebin_factor=res
)
print("Finished OT\n")

### Saving
balanced_results.to_csv(f"{save_dir}/immune_balanced_chr1_count={count}_res={res}_thres={thres}.csv")