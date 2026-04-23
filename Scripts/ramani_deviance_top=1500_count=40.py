### Setting directories
root="../"
data_dir = '/Users/oliviersmeets/Desktop/University/Master Internship 1/Data Handling/Data/ramani_cool_files' # Importing was not working with relative root+'..' approach, so replace with local data
metadata_dir = f"{data_dir}/cell_line_mapping.csv"
### Imports
import sys
print(sys.path)
import pandas as pd
from hicdatautils import hic_ot_optim_multi, import_cool_dir, subset_clr_data

### Importing coolers and metadata
clrs = import_cool_dir(data_dir)
metadata_df = pd.read_csv(metadata_dir)

### Setting parameters
save_dir = f"{root}Generated Data/Server/ramani_deviance_top=1500_count=40/"
count = 1
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
sum_res.to_csv(f"{save_dir}ramani_top={top}_count={count}_reg_m={reg_m}_sum.csv")
for chrom, results in dict_res.items():
    results.to_csv(f"{save_dir}ramani_top={top}_count={count}_reg_m={reg_m}_{chrom}.csv")