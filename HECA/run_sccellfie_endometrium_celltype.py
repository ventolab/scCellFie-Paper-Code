from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import sccellfie
from scanpy.readwrite import read
import os
import warnings
import json
from sccellfie.preprocessing.prepare_inputs import CORRECT_GENES

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Process directories and files.')
parser.add_argument('-i', '--input_adata', dest='input_adata', type=str, required=True, help='Input AnnData file')
parser.add_argument('-o', '--output_directory', dest='output_directory', type=str,
                    help='Root output directory. A subdirectory will be created with the basename of the AnnData file')
parser.add_argument('-c', '--celltype_col', dest='celltype_col', type=str,
                    default='celltype',
                    help='Column name for providing celltype annotations')
parser.add_argument('-n', '--n_counts_col', dest='n_counts_col', type=str,
                    default='total_counts',
                    help='Column name for providing the total counts per cell')
parser.add_argument('-s', '--scvi_file', dest='scvi', type=str,
                    default=None,
                    help='Path to csv file containing the SCVI embeddings')
parser.add_argument('-d', '--sccellfie_data_folder', dest='sccellfie_data_folder', type=str,
                    default='/nfs/team292/eg22/Metabolic-Tasks/Task-Data/homo_sapiens/',
                    help='Path to scCellFie data folder')
args = parser.parse_args()

# Directories and files
input_adata = args.input_adata
basename = os.path.splitext(os.path.basename(input_adata))[0]
sccellfie_data_folder = args.sccellfie_data_folder
output_directory = args.output_directory
os.makedirs(output_directory, exist_ok=True)
scvi = args.scvi

# Read Data
adata = read(input_adata, backed="r")
rxn_info = pd.read_json(os.path.join(sccellfie_data_folder, 'Rxn-Info-Recon2-2.json'))
task_info = pd.read_csv(os.path.join(sccellfie_data_folder, 'Task-Info.csv'))
task_by_rxn = pd.read_csv(os.path.join(sccellfie_data_folder, 'Task_by_Rxn.csv'), index_col='Task')
task_by_gene = pd.read_csv(os.path.join(sccellfie_data_folder, 'Task_by_Gene.csv'), index_col='Task')
rxn_by_gene = pd.read_csv(os.path.join(sccellfie_data_folder, 'Rxn_by_Gene.csv'), index_col='Reaction')
thresholds = pd.read_csv(os.path.join(sccellfie_data_folder, 'Thresholds.csv'), index_col='symbol')


if (scvi is not None):
    if os.path.isfile(scvi):
        scvi = pd.read_csv(scvi, index_col=0)

# Analysis setups
n_counts_col = args.n_counts_col  # Normally 'n_counts' column in adata.obs,
celltype_col = args.celltype_col
use_raw = False  # True if counts are in adata.raw.X, if they are in adata.X, set to False
chunk_size = 10_000
target_sum = 10_000  # Preprocessing for using threshold
n_neighbors = 10
smoothing_alpha = 0.33

# Rename gene names
ensembl_ids = False
if all([g.startswith('ENSG') for g in adata.var_names]):  # Check whether var_names are ensembl IDs
    gene_dict = thresholds.reset_index().set_index('ensembl_id')['symbol'].to_dict()
    ensembl_ids = True

# Analysis by chunks for cells
filter_pass_loc = np.array([i for i in range(adata.shape[0])])
save_idx = 0
cell_type_mapping = {}
for celltype, df in tqdm(adata.obs.groupby(celltype_col), desc='Running scCellFie in cells grouped by celltype'):
    adata_tmp = adata[list(df.index), :]
    adata_tmp = adata_tmp.to_memory()
    
    if adata_tmp.shape[0] == 0:
        continue
    
    if (scvi is not None):
        cells = [c for c in adata_tmp.obs.index if c in scvi.index]
        adata_tmp = adata_tmp[cells]
        adata_tmp.obsm['X_scVI'] = scvi.loc[cells, :]
        
    # Rename gene names
    if ensembl_ids:  # Check whether var_names are ensembl IDs
        var_names = [g for g in gene_dict.keys() if g in adata_tmp.var_names]
        adata_tmp = adata_tmp[:, var_names]
        adata_tmp.var_names = [gene_dict[g] for g in adata_tmp.var_names]

    # Organize data
    obs_names = adata_tmp.obs_names
    if 'counts' in adata_tmp.layers.keys():
        adata_tmp.X = adata_tmp.layers['counts'].copy()
    elif adata_tmp.raw is not None:
        adata_tmp.raw = adata_tmp.raw[obs_names, adata_tmp.var_names].to_adata()
        if adata_tmp.X.shape == adata_tmp.raw.X.shape:
            adata_tmp.X = adata_tmp.raw.X.copy()
    else:
        print('Raw counts were not found in adata.layers["counts"] or adata.raw. Assumming that adata.X contains raw counts.')
        
    n_counts = adata_tmp.obs[n_counts_col].values[:, None]

    if use_raw:
        X_view = adata_tmp.raw.X
    else:
        X_view = adata_tmp.X
        
    X_norm = X_view / n_counts * target_sum
    
    if isinstance(X_norm, sp.coo_matrix):
        X_norm = X_norm.tocsr()

    if use_raw:
        adata_tmp.raw = adata_tmp.raw.to_adata()
        adata_tmp.raw.X = X_norm
        adata_tmp.raw = adata_raw
    else:
        adata_tmp.X = X_norm  
        
    if adata_tmp.shape[0] > 2:       
        adata_knn = adata_tmp.raw.copy() if use_raw else adata_tmp.copy()
        if 'X_scVI' in adata_knn.obsm.keys():
            use_rep = 'X_scVI'
        elif 'X_scvi' in adata_knn.obsm.keys():
            use_rep = 'X_scvi'
        elif 'X_pca' in adata_knn.obsm.keys():
            use_rep = 'X_pca'
        else:
            sc.pp.log1p(adata_knn)
            sc.pp.highly_variable_genes(adata_knn, min_mean=0.0125, max_mean=3, min_disp=0.5)
            sc.tl.pca(adata_knn, svd_solver="arpack")
            use_rep = 'X_pca'
        sc.pp.neighbors(adata_knn, n_neighbors=n_neighbors, use_rep=use_rep)

        adata_tmp.obsp['distances'] = adata_knn.obsp['distances']
        adata_tmp.obsp['connectivities'] = adata_knn.obsp['connectivities']
        adata_tmp.uns['neighbors'] = adata_knn.uns['neighbors']
        
        del adata_knn

    if save_idx == 0:
        # Obtain info & inputs once
        adata2, gpr_rules, task_by_gene, rxn_by_gene, task_by_rxn = sccellfie.preprocessing.preprocess_inputs(adata_tmp,
                                                                                                              gpr_info=rxn_info,
                                                                                                              task_by_gene=task_by_gene,
                                                                                                              rxn_by_gene=rxn_by_gene,
                                                                                                              task_by_rxn=task_by_rxn,
                                                                                                              verbose=False
                                                                                                              )
        # Generate thresholds from GeneFormer
        thresholds = thresholds.loc[adata2.var_names, :]
        met_genes = list(adata2.var_names)
    else:
        correction_dict = CORRECT_GENES['human']
        correction_dict = {k : v for k, v in correction_dict.items() if v in met_genes}
        adata_tmp.var.index = [correction_dict[g] if g in correction_dict.keys() else g for g in adata_tmp.var.index]
        adata2 = adata_tmp[:, met_genes]
        if (use_raw) & (adata2.raw is not None):
            adata2.raw.var.index = adata2.var.index
            adata2.raw = adata2.raw[:, met_genes].to_adata()
    
    if adata_tmp.shape[0] > 2: 
        # Smooth gene expression
        sccellfie.smoothing.smooth_expression_knn(adata2,
                                                  mode='adjacency', 
                                                  alpha=smoothing_alpha,
                                                  use_raw=use_raw,
                                                  chunk_size=chunk_size if adata_tmp.shape[0] > 30000 else None,
                                                  disable_pbar=True)

        if use_raw:
            adata2.raw.X = adata2.layers['smoothed_X']
        else:
            adata2.X = adata2.layers['smoothed_X']

    # Compute gene scores, RAL & metabolic task scores
    sccellfie.gene_score.compute_gene_scores(adata2, thresholds[['sccellfie_threshold']], use_raw=use_raw)
    sccellfie.reaction_activity.compute_reaction_activity(adata2, gpr_rules, disable_pbar=True)
    sccellfie.metabolic_task.compute_mt_score(adata2, task_by_rxn, verbose=False)

    # Generate output dataframes
    gene_df = pd.DataFrame(adata2.layers['gene_scores'], index=adata2.obs_names,
                           columns=adata2.var_names)  # Gene scores
    rxn_df = adata2.reactions.to_df()  # Rxn activity
    mt_df = adata2.metabolic_tasks.to_df()  # Metabolic task score
    det_gene_df = adata2.reactions.uns['Rxn-Max-Genes']  # Determinant gene of each reaction per cell
    
    # Save dataframes
    gene_df.to_csv(os.path.join(output_directory, f'GS_dataframe_{save_idx}.csv'))
    rxn_df.to_csv(os.path.join(output_directory, f'RXN_dataframe_{save_idx}.csv'))
    mt_df.to_csv(os.path.join(output_directory, f'MT_dataframe_{save_idx}.csv'))
    det_gene_df.to_csv(os.path.join(output_directory, f'DetGene_dataframe_{save_idx}.csv'))
    
    cell_type_mapping[celltype] = save_idx
    
    # Delete variables to free memory
    del gene_df, rxn_df, mt_df, det_gene_df
    del adata_tmp, adata2
    save_idx += 1
    
mapping_file = os.path.join(output_directory, 'cell_type_mapping.json')
with open(mapping_file, 'w') as jsonfile:
    json.dump(cell_type_mapping, jsonfile)

print(f"Cell type mapping saved to {mapping_file}")