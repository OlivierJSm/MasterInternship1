### Setting directories
data_dir = '/mbshome/osmeets/MasterInternship1/Data/ramani_cool_files' # Importing was not working with relative root+'..' approach, so replace with local data
metadata_dir = f"{data_dir}/cell_line_mapping.csv"
### Imports
from hicdatautils import hic_ot_optim_multi, import_cool_dir, subset_clr_data
import pandas as pd

### Importing coolers and metadata
clrs = import_cool_dir(data_dir)
metadata_df = pd.read_csv(metadata_dir)

### Setting parameters
save_dir = '/mbshome/osmeets/MasterInternship1/Generated Data/Server/'
count = None
top = 1500
reg_m = 0.1

### Subsetting
clr_subset = subset_clr_data(
    data_dir,
    metadata_df,
    count,
    iqr_rule = 1.5,
    type_col="cell_line"
)

### Generating results
sum_res, dict_res = hic_ot_optim_multi(
    coolers=clr_subset, 
    selection="deviance",
    norm="max",
    top_contacts=top,
    unbalanced=reg_m
)

### Saving
sum_res.to_csv(f"SUM_{save_dir}ramani_top={top}_count=ALL_reg_m={reg_m}.csv")
for chrom, results in dict_res.items():
    results.to_csv(f"{save_dir}ramani_top={top}_count=ALL_reg_m={reg_m}_{chrom}.csv")