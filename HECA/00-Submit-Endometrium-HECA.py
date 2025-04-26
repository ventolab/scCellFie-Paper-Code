import os
import logging
import textwrap
import tempfile
import subprocess

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")

# Set the datasets source folder
datasets_by_tissue = "/nfs/team292/eg22/Datasets/Endometrium/"
# Set the output directory
output_directory="/nfs/team292/eg22/Metabolic-Tasks/Endometrium/HECA/MT-2025-02-18/"
os.makedirs(output_directory, exist_ok=True)

manual_files = ['endometriumAtlasV2_cells_with_counts.h5ad'] 

# Loop through each h5ad file in the directory
for h5ad in [f for f in os.listdir(datasets_by_tissue) if (any([mf in f for mf in manual_files]))]: # if f.endswith(".h5ad")]: (any([mf in f for mf in manual_files]))
    logging.info(f"File '{h5ad}'")
    # Build full path to file
    h5ad_path = os.path.join(datasets_by_tissue, h5ad)
    # Get file stats (size)
    h5ad_stats = os.stat(h5ad_path)
    size_gb = h5ad_stats.st_size / (1024 ** 3)

    memory_gb = int(64)

    logging.info(f"File size is {size_gb:.02f}G will use {memory_gb}G of memory")

    tissue = 'Endo-HECA'
    queue = 'long'
    
    logging.info(f"Building job script")
    # Build job submission script
    job = f"""
    #BSUB -J scCellFie_{tissue}
    #BSUB -G team292
    #BSUB -q {queue}
    #BSUB -M {memory_gb}G
    #BSUB -R "select[mem>{memory_gb}G] rusage[mem={memory_gb}G] span[hosts=1]"
    #BSUB -n 1
    #BSUB -o "{os.path.join(output_directory, tissue)}.%J.out"
    #BSUB -e "{os.path.join(output_directory, tissue)}.%J.err"

    # assuming you've got a conda env named single_cell
    # module load cellgen/conda 
    # conda activate single_cell
   
    # run python script
    /lustre/scratch126/cellgen/team292/eg22/miniforge3/envs/single_cell/bin/python3.10 /nfs/team292/eg22/FARM-Bsubs/scCellFie/run_sccellfie_endometrium_celltype.py -i "{h5ad_path}" -o "{output_directory}" -n n_counts -c celltype
    """

    # Remove leading spaces 
    job = textwrap.dedent(job)
    
    # Dump as a temp file
    with tempfile.TemporaryFile() as tmp:
        tmp.write(job.encode("utf-8"))
        tmp.seek(0)

        # Submit the job using bsub
        result = subprocess.run("bsub", shell=True, stdin=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(result.stdout)!=0:
            logging.info(result.stdout.decode())
        if len(result.stderr)!=0:
            logging.error(result.stderr.decode()) 