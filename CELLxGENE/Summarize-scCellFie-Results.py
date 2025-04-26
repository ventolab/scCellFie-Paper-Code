from tqdm import tqdm
import pandas as pd
import numpy as np
import sccellfie
import os
import json

def process_csv_files(input_directory, prefix, min_ncells=1, excluded_tissue=None, rename_tissue_dict=None):
    """
    Process CSV files from scCellFie output and calculate metrics.
    
    Args:
        input_directory (str): Directory containing scCellFie output
        prefix (str): Prefix of the CSV files to process (MT for Metabolic Tasks, RXN for Reactions)
        
    Returns:
        tuple: (results, min_max_df, features_seen)
            - results: List of tuples (col_name, trimean_df, variance_df, std_df, threshold_cells, nonzero_cells, n_cells)
            - min_max_df: DataFrame with min/max values for each feature
            - features_seen: Set of all unique features seen during processing
    """
    subfolders = [dir_ for dir_ in os.listdir(input_directory) 
                  if os.path.isdir(os.path.join(input_directory, dir_))]
    
    excluded = []
    subfolders = [subfolder for subfolder in subfolders if subfolder not in excluded]
    
    results = []
    
    # Initialize dictionaries to track min/max values
    single_cell_min = {}
    single_cell_max = {}
    cell_type_min = {}
    cell_type_max = {}
    
    # Track all seen features
    features_seen = set()
    
    for subfolder in subfolders:
        tissue = subfolder
        subfolder_path = os.path.join(input_directory, subfolder)
        
        # Load cell type mapping from JSON file
        mapping_file = os.path.join(subfolder_path, 'cell_type_mapping.json')
        if not os.path.exists(mapping_file):
            print(f"Warning: {mapping_file} not found. Skipping {subfolder}.")
            continue
            
        try:
            with open(mapping_file, 'r') as f:
                cell_type_to_index = json.load(f)
            
            # Reverse the mapping: index to cell type
            index_to_cell_type = {str(v): k for k, v in cell_type_to_index.items()}
        except Exception as e:
            print(f"Error loading mapping file {mapping_file}: {e}. Skipping {subfolder}.")
            continue
        
        # Get all CSV files for the specified prefix
        csv_files = [file for file in os.listdir(subfolder_path) 
                    if (file.endswith('.csv')) and (file.startswith(f'{prefix}_dataframe_'))]
        
        if excluded_tissue is not None:
            if tissue in excluded_tissue:
                print(f"{tissue} found in excluded list. Skipping {subfolder}.")
                continue
        
        if rename_tissue_dict is not None:
            if tissue in rename_tissue_dict.keys():
                tissue = rename_tissue_dict[tissue]
        
        for csv_file in tqdm(csv_files, desc=f'Processing {prefix} files for {tissue}', position=0, leave=True):
            # Extract index from filename - assuming format is PREFIX_dataframe_INDEX.csv
            index = csv_file.replace(f'{prefix}_dataframe_', '').replace('.csv', '')
            
            # Skip if index not in mapping
            if index not in index_to_cell_type:
                print(f"Warning: Index {index} not found in mapping for {subfolder}. Skipping file.")
                continue
            
            cell_type = index_to_cell_type[index]
            csv_path = os.path.join(subfolder_path, csv_file)
            
            try:
                df = pd.read_csv(csv_path, index_col=0)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}. Skipping file.")
                continue
            
            if df.empty:
                print(f"Warning: Empty dataframe in {csv_path}. Skipping file.")
                continue
                
            n_cells = df.shape[0]
            
            if n_cells < min_ncells:
                print(f"{cell_type} doesn't have enough cells. Total {n_cells}. Skipping file.")
                continue
            
            col_name = f'{tissue} / {cell_type}'
            
            # Calculate trimean
            trimean_values = sccellfie.expression.aggregation.AGG_FUNC['trimean'](df, axis=0)
            trimean_df = pd.DataFrame(
                trimean_values,
                index=df.columns,
                columns=[col_name]
            )
            
            # Calculate variance
            variance_values = df.var(axis=0)
            variance_df = pd.DataFrame(
                variance_values,
                index=df.columns,
                columns=[col_name]
            )
            
            # Calculate standard deviation
            std_values = df.std(axis=0)
            std_df = pd.DataFrame(
                std_values,
                index=df.columns,
                columns=[col_name]
            )
            
            # Count cells passing expression threshold
            threshold_cells = (df.ge(5*np.log(2)).astype(int).sum()).to_frame()
            threshold_cells.columns = [col_name]
            
            # Count cells with non-zero values
            nonzero_cells = (df.gt(0).astype(int).sum()).to_frame()
            nonzero_cells.columns = [col_name]
            
            # Store results
            results.append((col_name, trimean_df, variance_df, std_df, threshold_cells, nonzero_cells, n_cells))
            
            # Update features seen
            features_seen.update(df.columns)
            
            # Get min/max for all features in current dataframe using pandas operations
            curr_min_series = df.min()
            curr_max_series = df.max()
            curr_trimean_series = pd.Series(trimean_values, index=df.columns)
            
            # Initialize dictionaries if they're empty
            if not single_cell_min:
                single_cell_min = curr_min_series.to_dict()
            if not single_cell_max:
                single_cell_max = curr_max_series.to_dict()
            if not cell_type_min:
                cell_type_min = curr_trimean_series.to_dict()
            if not cell_type_max:
                cell_type_max = curr_trimean_series.to_dict()
            else:
                # Update global min/max values using vectorized operations
                for feature in df.columns:
                    single_cell_min[feature] = min(single_cell_min.get(feature, float('inf')), curr_min_series[feature])
                    single_cell_max[feature] = max(single_cell_max.get(feature, float('-inf')), curr_max_series[feature])
                    cell_type_min[feature] = min(cell_type_min.get(feature, float('inf')), curr_trimean_series[feature])
                    cell_type_max[feature] = max(cell_type_max.get(feature, float('-inf')), curr_trimean_series[feature])
    
    # Create min/max DataFrame with features as columns
    min_max_rows = ['single_cell_min', 'single_cell_max', 'cell_type_min', 'cell_type_max']
    min_max_data = {}
    
    for feature in sorted(features_seen):
        min_max_data[feature] = [
            single_cell_min.get(feature, 0),  # single_cell_min
            single_cell_max.get(feature, 0),  # single_cell_max
            cell_type_min.get(feature, 0),    # cell_type_min
            cell_type_max.get(feature, 0)     # cell_type_max
        ]
    
    min_max_df = pd.DataFrame(min_max_data, index=min_max_rows)
    
    return results, min_max_df, features_seen

def melt_cellfie_results(trimeans, variance, std, threshold_cells, nonzero_cells, cell_counts_df):
    """
    Melts CellFie results into a long format dataframe.
    
    Parameters
    ----------
    trimeans : pd.DataFrame
        DataFrame containing trimean values
    variance : pd.DataFrame
        DataFrame containing variance values
    std : pd.DataFrame
        DataFrame containing standard deviation values
    threshold_cells : pd.DataFrame
        DataFrame containing threshold cells counts
    nonzero_cells : pd.DataFrame
        DataFrame containing nonzero cells counts
    cell_counts_df : pd.DataFrame
        DataFrame containing cell counts by tissue and cell type
        
    Returns
    -------
    pd.DataFrame
        Long format dataframe containing all metrics per cell type, tissue and metabolic pathway
    """
    # Create a dictionary to map tissue/cell_type to total cells
    cell_counts_df['tissue_celltype'] = cell_counts_df['tissue'] + ' / ' + cell_counts_df['cell_type']
    total_cells_dict = dict(zip(cell_counts_df['tissue_celltype'], cell_counts_df['total_cells']))
    
    # Initialize lists to store the melted data
    data = []
    
    # Get features (index of any of the metric dataframes)
    features = trimeans.index
    
    # Iterate through the columns (tissue / cell type combinations)
    for col in trimeans.columns:
        tissue, cell_type = col.split(' / ')
        total_cells = total_cells_dict.get(col, 0)
        
        # Iterate through features
        for feature in features:
            data.append({
                'feature': feature,
                'tissue': tissue,
                'cell_type': cell_type,
                'trimean': trimeans.loc[feature, col],
                'variance': variance.loc[feature, col],
                'std': std.loc[feature, col],
                'n_cells_threshold': threshold_cells.loc[feature, col],
                'n_cells_nonzero': nonzero_cells.loc[feature, col],
                'total_cells': total_cells
            })
    
    # Create dataframe
    df_melted = pd.DataFrame(data)
    
    # Sort the dataframe
    df_melted = df_melted.sort_values(['tissue', 'cell_type', 'feature']).reset_index(drop=True)
    
    return df_melted

def process_prefix_data(input_directory, output_directory, prefix, min_ncells=1, excluded_tissue=None, rename_tissue_dict=None):
    """
    Process data for a specific prefix (MT or RXN)
    
    Args:
        input_directory (str): Directory containing input data
        output_directory (str): Directory to save results
        prefix (str): Prefix to process (MT or RXN)
    """
    # Set output name based on prefix
    if prefix == "MT":
        output_name = "Metabolic-Tasks"
    elif prefix == "RXN":
        output_name = "Reactions"
    else:
        output_name = prefix
    
    print(f"Processing {output_name} data...")
    
    # Process files and collect results
    results, min_max_df, features_seen = process_csv_files(input_directory, prefix, min_ncells, excluded_tissue, rename_tissue_dict)
    
    if not results:
        print(f"No {output_name} data found. Skipping.")
        return
    
    # Extract components from results
    col_names = []
    trimean_dfs = []
    variance_dfs = []
    std_dfs = []
    threshold_cells_dfs = []
    nonzero_cells_dfs = []
    cell_counts = {}
    
    for col_name, trimean_df, variance_df, std_df, threshold_cells_df, nonzero_cells_df, n_cells in results:
        col_names.append(col_name)
        trimean_dfs.append(trimean_df)
        variance_dfs.append(variance_df)
        std_dfs.append(std_df)
        threshold_cells_dfs.append(threshold_cells_df)
        nonzero_cells_dfs.append(nonzero_cells_df)
        cell_counts[col_name] = n_cells
    
    # Concatenate the DataFrames
    trimeans = pd.concat(trimean_dfs, axis=1).fillna(0)
    variance = pd.concat(variance_dfs, axis=1).fillna(0)
    std = pd.concat(std_dfs, axis=1).fillna(0)
    threshold_cells = pd.concat(threshold_cells_dfs, axis=1).fillna(0)
    nonzero_cells = pd.concat(nonzero_cells_dfs, axis=1).fillna(0)
    
    # Create cell counts DataFrame
    cell_counts_records = [
        (tissue, cell_type, n_cells) 
        for (tissue, cell_type), n_cells in zip(
            [name.split(' / ') for name in cell_counts.keys()],
            cell_counts.values()
        )
    ]
    
    cell_counts_df = pd.DataFrame.from_records(
        cell_counts_records, 
        columns=['tissue', 'cell_type', 'total_cells']
    )
    
    # Create melted results
    melted_results = melt_cellfie_results(trimeans, variance, std, threshold_cells, nonzero_cells, cell_counts_df)
    
    # Rename the 'feature' column based on the prefix
    if prefix == "MT":
        melted_results.rename(columns={'feature': 'metabolic_task'}, inplace=True)
    elif prefix == "RXN":
        melted_results.rename(columns={'feature': 'reaction'}, inplace=True)
    
    # Save results
    trimeans.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-Trimeans-MainTissue.csv'))
    variance.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-Variance-MainTissue.csv'))
    std.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-StdDev-MainTissue.csv'))
    threshold_cells.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-ThresholdCells-MainTissue.csv'))
    nonzero_cells.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-NonzeroCells-MainTissue.csv'))
    melted_results.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-MeltedResults.csv'), index=False)
    cell_counts_df.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-CellCounts-ByTissueAndCellType.csv'), index=False)
    
    # Save min/max values
    min_max_df.to_csv(os.path.join(output_directory, f'CELLxGENE-{output_name}-MinMax.csv'))
    
    print(f"{output_name} processing complete!")

def main():
    # Locations - separate input and output directories
    input_directory = '/lustre/scratch126/cellgen/team292/eg22/CELLxGENE/Run-sccellfie-V0_4_2-snapshot-Apr24/'
    output_directory = '/nfs/team292/eg22/Metabolic-Tasks/CELLxGENE/Processed-Results-V042/'
    
    os.makedirs(output_directory, exist_ok=True)
    
    # Cells filter
    min_ncells = 50
    tissue_exclude = ['digestive_system', 'respiratory_system']
    tissue_replace = {'immune_system' : 'tonsil', 'central_nervous_system' : 'cerebellum', 'reproductive_system' : 'early_gonad', 'skeletal_system' : 'bones', 'bladder_organ' : 'bladder'}
        
    # Process Metabolic Tasks (MT)
    process_prefix_data(input_directory, output_directory, "MT", min_ncells, tissue_exclude, tissue_replace)
    
    # Process Reactions (RXN)
    process_prefix_data(input_directory, output_directory, "RXN", min_ncells, tissue_exclude, tissue_replace)
    
    print("All processing complete!")

if __name__ == "__main__":
    main()