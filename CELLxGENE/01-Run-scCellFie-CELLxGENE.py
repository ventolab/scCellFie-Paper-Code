import os
import logging
import textwrap
import tempfile
import subprocess

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")

# Set the datasets source folder
datasets_by_tissue = "/lustre/scratch127/cellgen/cellgeni/projects/cellxgene_census_snapshot/2024-04-01/datasets_by_tissue/"
# Set the output directory
output_directory="/lustre/scratch126/cellgen/team292/eg22/CELLxGENE/Run-sccellfie-V0_4_2-snapshot-Apr24/"
os.makedirs(output_directory, exist_ok=True)

manual_files = []

# Loop through each h5ad file in the directory
for h5ad in [f for f in os.listdir(datasets_by_tissue) if f.endswith(".h5ad")]: # if f.endswith(".h5ad")]: # if (any([mf in f for mf in manual_files]))]
    logging.info(f"File '{h5ad}'")
    # Build full path to file
    h5ad_path = os.path.join(datasets_by_tissue, h5ad)
    # Get file stats (size)
    h5ad_stats = os.stat(h5ad_path)
    size_gb = h5ad_stats.st_size / (1024 ** 3)
    # Estimate 150% of the file size
    memory_gb = size_gb * 1.5
    
    if size_gb < 24:
        min_mem = 96 #64
    elif size_gb < 48:
        min_mem = 128
    else:
        min_mem = 256
    # Set the minimum memory size to 32GB
    memory_gb = max(min_mem, memory_gb) # 32
    # Set the maximum memory size to 650GB
    memory_gb = min(650, memory_gb) # 650
    # Make sure memory is an integer before submission
    memory_gb = int(memory_gb)

    tissue = h5ad.replace(".h5ad","")
    queue = 'long'
    
    if h5ad in ['blood.h5ad', 'lung.h5ad']:
        queue = 'week'
    if h5ad in ['adrenal_gland.h5ad']:
        memory_gb = 128
    elif h5ad in ['brain.h5ad']:
        queue = 'basement'
        
    logging.info(f"File size is {size_gb:.02f}G will use {memory_gb}G of memory")
    
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
    /lustre/scratch126/cellgen/team292/eg22/miniforge3/envs/single_cell/bin/python3.10 /nfs/team292/eg22/FARM-Bsubs/scCellFie/run_sccellfie_celltype_chunks.py -i "{h5ad_path}" -o "{output_directory}"
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