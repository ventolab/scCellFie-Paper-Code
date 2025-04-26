import os
import scanpy as sc
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from scipy import sparse as sp

import matplotlib.pyplot as plt

import mpl_fontkit as fk
fk.install("Lato")
fk.set_font("Lato")
plt.rcParams['font.size'] = 14


# DATA SETUP
datasets_by_tissue = "/lustre/scratch127/cellgen/cellgeni/projects/cellxgene_census_snapshot/2024-04-01/datasets_by_tissue/"
output_directory = "/nfs/team292/eg22/Metabolic-Tasks/CELLxGENE/Thresholds-Calculation/"
#output_directory = "/lustre/scratch126/cellgen/team292/eg22/Thresholds-Calculation/"
sccellfie_data_folder = '/nfs/team292/eg22/Metabolic-Tasks/Task-Data/homo_sapiens/'

ensembl2symbol = pd.read_csv(os.path.join(sccellfie_data_folder, 'Ensembl2symbol.csv'))
rxn_by_gene = pd.read_csv(os.path.join(sccellfie_data_folder, 'Rxn_by_Gene.csv'), index_col='Reaction')

with open(os.path.join(sccellfie_data_folder, 'All_metabolic_genes.json')) as fp:
    metabolic_genes = json.load(fp)
    
ensembl2symbol = ensembl2symbol.loc[ensembl2symbol['symbol'].isin(metabolic_genes)]
gene_dict = ensembl2symbol.reset_index().set_index('ensembl_id')['symbol'].to_dict()
rxn2ensembl = ensembl2symbol.loc[ensembl2symbol['symbol'].isin(list(rxn_by_gene.columns))].reset_index().set_index('ensembl_id')['symbol'].to_dict()
gene_dict.update(rxn2ensembl)

# excluded_files = ['digestive_system.h5ad', 'respiratory_system.h5ad']
manual_files = ['testis.h5ad']

# Initialize dictionaries to store results
tissue_stats_dict = {}
gene_stats_dict = {}

# Analysis setups
chunk_size = 100000

use_raw = False  # True if counts are in adata.raw.X, if they are in adata.X, set to False

# Normalization parameters
n_counts_col = 'raw_sum'  # Normally 'n_counts' column in adata.obs, but CELLxGENE uses 'raw_sum'
target_sum = 10000 # This is CP10k

# Set the temporary directory for storing batch files
temp_dir = os.path.join(output_directory, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Initialize counters
total_count = 0
batch_counter = 0
# Loop through each h5ad file in the directory / each h5ad file is an organ or system in CELLxGENE
for h5ad in [f for f in os.listdir(datasets_by_tissue) if (f.endswith(".h5ad"))]:  # & (f in manual_files)
    h5ad_path = os.path.join(datasets_by_tissue, h5ad)
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # Analysis by chunks for cells in normal condition (without disease)
    filter_pass_loc = np.array([i for i, v in enumerate((adata.obs['disease'] == 'normal').values) if v])

    # Check gene names nomenclature
    if all([g.startswith('ENS') for g in adata.var_names]):  # Check whether var_names are ensembl IDs
        ensembl_id = True
        var_names = [g for g in gene_dict.keys() if g in adata.var_names]
    else:
        ensembl_id = False
        var_names = [g for g in metabolic_genes if g in adata.var_names]

    # Intersection between metabolic genes and genes in data
    # var_names = list(set(var_names) & set(metabolic_genes))

    # Initialize variables for cumulative results per h5ad file
    sum_counts_per_gene = np.zeros(len(var_names))
    nonzero_cells_per_gene = np.zeros(len(var_names), dtype=int)
    max_values_per_gene = np.zeros(len(var_names))
    total_cells = 0

    for i in tqdm(range(0, len(filter_pass_loc), chunk_size), desc=f'Processing chunks of {h5ad}', position=0,
                  leave=True):
        idx = filter_pass_loc[i: i + chunk_size]
        adata_tmp = adata[idx, :]
        adata_tmp = adata_tmp.to_memory()
        adata_tmp = adata_tmp[:, var_names]

        # Rename gene names
        if ensembl_id:
            adata_tmp.var_names = [gene_dict[g] for g in adata_tmp.var_names]

        obs_names = adata_tmp.obs_names
        if adata_tmp.raw is not None:
            adata_tmp.raw = adata_tmp.raw[obs_names, adata_tmp.var_names].to_adata()

        n_counts = adata_tmp.obs[n_counts_col].values[:, None]

        if use_raw:
            X_view = adata_tmp.raw.X
        else:
            X_view = adata_tmp.X

        X_norm = X_view / n_counts * target_sum
        X_norm = sp.csr_matrix(X_norm, dtype=np.float32)

        if use_raw:
            adata_tmp_raw = adata_tmp.raw.to_adata()
            adata_tmp_raw.X = X_norm
            adata_tmp.raw = adata_tmp_raw
        else:
            adata_tmp.X = X_norm

        # Update cumulative results per h5ad file
        sum_counts_per_gene += np.squeeze(np.asarray(X_norm.sum(axis=0)))
        nonzero_cells_per_gene += np.squeeze(np.asarray((X_norm > 0).sum(axis=0)))
        max_values_per_gene = np.maximum(max_values_per_gene, np.squeeze(np.asarray(X_norm.max(axis=0).todense())))
        total_cells += adata_tmp.n_obs

        X_norm = X_norm.toarray().flatten()

        # Store non-zero values in a temporary file
        nonzero_values = X_norm[X_norm > 0]
        batch_file = os.path.join(temp_dir, f"batch_{batch_counter}.txt")
        np.savetxt(batch_file, nonzero_values, fmt='%.6f')

        # Update counters
        total_count += nonzero_values.shape[0]
        batch_counter += 1

    # Store the tissue-level results in the dictionary
    tissue = os.path.splitext(h5ad)[0]  # Extract the tissue name from the filename
    tissue_stats_dict[tissue] = {
        'N_Cells': total_cells,
        'CumCounts': sum_counts_per_gene,
        'N_Cells_NZ': nonzero_cells_per_gene,
        'MaxValues': max_values_per_gene
    }

## Post chunk analysis

# Set the output file for storing the sorted values
output_file = os.path.join(output_directory, "sorted_values.txt")

# Perform external sorting using the batch files
batch_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("batch_")]
os.system(f"sort -m {' '.join(batch_files)} | sort -n > {output_file}")

# Remove the temporary directory
os.system(f"rm -r {temp_dir}")

# Set the percentiles
percentiles = [10, 25, 50, 75, 90, 95]

# # Compute percentiles by reading the sorted file
# percentile_values = []
# with open(output_file, 'r') as f:
#     percentile_indices = [int(total_count * p / 100) for p in percentiles]
#     f.seek(0, 0)
#     line_n = 0
#     for i, line in tqdm(enumerate(f), total=total_count):
#         if i in percentile_indices:
#             val = float(line.strip())
#             percentile_values.append(val)
#             print(val)

#         line_n += 1
# percentiles_dict = dict(zip(percentiles, percentile_values))

def compute_percentiles(output_file, total_count, percentiles, chunksize=10**7):
    # Calculate the indices corresponding to the percentiles
    percentile_indices = [int(total_count * p / 100) for p in percentiles]
    
    # Initialize an empty list to store the percentile values
    percentile_values = []
    
    # Initialize variables to keep track of the current position and chunk size
    current_position = 0

    # Read the file in chunks
    for chunk in tqdm(pd.read_csv(output_file, chunksize=chunksize, header=None),
                      total=int(np.ceil(total_count / chunksize))
                     ):
        chunk_size = len(chunk)
        
        # Check if the current chunk contains any of the percentile indices
        for idx in percentile_indices:
            if current_position <= idx < current_position + chunk_size:
                # Calculate the relative position within the current chunk
                relative_idx = idx - current_position
                # Get the value at the relative position and add it to the list
                val = chunk.iloc[relative_idx, 0]
                percentile_values.append(val)
                print(f"Percentile {percentiles[percentile_indices.index(idx)]}: {val}")

        # Update the current position
        current_position += chunk_size
    
    # Create a dictionary of percentiles and their corresponding values
    percentiles_dict = dict(zip(percentiles, percentile_values))
    
    return percentiles_dict

percentiles_dict = compute_percentiles(output_file, total_count, percentiles)
percentile_values = [percentiles_dict[p] for p in percentiles]

# Export values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open(os.path.join(output_directory, 'Tissue_MeanGeneExpressions.json'), 'w') as fp:
    json.dump(tissue_stats_dict, fp, cls=NumpyEncoder)

with open(os.path.join(output_directory, 'Tissue_GeneNames.json'), 'w') as fp:
    json.dump(list(adata_tmp.var_names), fp, cls=NumpyEncoder)

with open(os.path.join(output_directory, 'GeneExpressionPercentiles.json'), 'w') as fp:
    json.dump(percentiles_dict, fp, cls=NumpyEncoder)

## Generate threshold table
# Extract the gene names from adata_tmp
gene_names = list(adata_tmp.var_names)

# Initialize variables to store the totals
total_sum_counts = np.zeros(len(gene_names))
total_nonzero_cells = np.zeros(len(gene_names))
total_cells_tissue = np.zeros(len(gene_names))
max_values = np.zeros(len(gene_names))

# Iterate over the tissue-level results and accumulate the totals
for tissue_stats in tissue_stats_dict.values():
    total_sum_counts += tissue_stats['CumCounts']
    total_nonzero_cells += tissue_stats['N_Cells_NZ']
    total_cells_tissue += tissue_stats['N_Cells']
    max_values = np.maximum(max_values, tissue_stats['MaxValues'])

# Compute the average by dividing the total sum counts by the total nonzero cells
average_nz_counts = np.divide(total_sum_counts, total_nonzero_cells,
                              out=np.zeros_like(total_sum_counts), where=total_nonzero_cells!=0)

average_counts = np.divide(total_sum_counts, total_cells_tissue,
                           out=np.zeros_like(total_sum_counts), where=total_cells_tissue!=0)

# Create a DataFrame with the average counts and gene names as the index
df_average_counts = pd.DataFrame({'Mean': average_counts,
                                  'NonZero-Mean': average_nz_counts,
                                  'Max': max_values
                                  },index=gene_names)

# Get the 25th and 75th percentile values
percentile_25 = percentiles_dict[25]
percentile_75 = percentiles_dict[75]

# Function to apply the new threshold logic
def apply_threshold(row):
    if (row['Max'] > percentile_25) | (row['Max'] == 0.):
        return pd.Series({
            'Mean': np.clip(row['Mean'], percentile_25, percentile_75),
            'NonZero-Mean': np.clip(row['NonZero-Mean'], percentile_25, percentile_75)
        })
    else:
        return pd.Series({
            'Mean': row['Mean'],
            'NonZero-Mean': row['NonZero-Mean']
        })

# Apply the new threshold logic to the entire DataFrame
thresholds = df_average_counts.apply(apply_threshold, axis=1)

# Rename the columns
thresholds.columns = ['sccellfie_threshold_mean', 'sccellfie_threshold_nonzero_mean']
thresholds = thresholds.join(ensembl2symbol.reset_index().set_index('symbol'))
thresholds = thresholds[['ensembl_id', 'sccellfie_threshold_nonzero_mean']]
thresholds = thresholds.rename(columns={'sccellfie_threshold_nonzero_mean' : 'sccellfie_threshold'})
thresholds.to_csv(os.path.join(output_directory,'scCellFie-Thresholds.csv'))

## Plot distribution
# Get the minimum and maximum values to plot
import subprocess
min_value = float(subprocess.check_output(['head', '-n', '1', output_file]).decode().strip())
max_value = np.ceil(percentile_values[-1]*2.) # + 1

# Create a distribution using linspace and count the frequency of values in each bin
bin_edges = np.linspace(min_value, max_value, 51)  # 50 bins
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# histogram = np.zeros(len(bin_edges) - 1) #  - 1

# with open(output_file, 'r') as f:
#     f.seek(0, 0)
#     for line in tqdm(f, total=total_count):
#         value = float(line.strip())
#         bin_index = np.clip(np.searchsorted(bin_edges, value, side='right') - 1, 0, histogram.shape[0] - 1)
#         histogram[bin_index] += 1

# More efficient option
def process_chunk(chunk, bin_edges):
    hist, _ = np.histogram(chunk, bins=bin_edges)
    return hist

def generate_histogram(output_file, bin_edges, total_count, chunksize=10**7):
    # Adjust chunk size as needed
    histogram = np.zeros(len(bin_edges) - 1, dtype=int)
    
    for chunk in tqdm(pd.read_csv(output_file, chunksize=chunksize,
                                  header=None),
                      total=int(np.ceil(total_count / chunksize))
                     ):
        chunk = chunk.squeeze()  # Convert DataFrame to Series
        histogram += process_chunk(chunk, bin_edges)
    
    return histogram

histogram = generate_histogram(output_file, bin_edges, total_count)

# Normalize the histogram to obtain frequency density
histogram_norm = histogram / (total_count * np.diff(bin_edges))

with open(os.path.join(output_directory,'Histogram.json'), 'w') as fp:
    json.dump(histogram, fp, cls=NumpyEncoder)

with open(os.path.join(output_directory,'Histogram_norm.json'), 'w') as fp:
    json.dump(histogram_norm, fp, cls=NumpyEncoder)

# Create a new figure and set the size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the histogram
ax.bar(bin_centers, histogram_norm, width=np.diff(bin_edges), edgecolor='black', linewidth=0.8)
ax.set_xlabel('Gene Expression Values')
ax.set_ylabel('Frequency Density')
ax.set_title('Distribution of Non-Zero Gene Expression Values')

# Add vertical lines for the percentiles
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
for percentile, value, color in zip(percentiles, percentile_values, colors):
    ax.axvline(x=value, color=color, linestyle='--', linewidth=1.5, label=f'{percentile}th Percentile')

# Add a legend
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

plt.savefig(os.path.join(output_directory, 'Expression_Dist.pdf'), dpi=300, bbox_inches='tight')