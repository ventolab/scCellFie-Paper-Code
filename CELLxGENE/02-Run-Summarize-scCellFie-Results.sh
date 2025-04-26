#BSUB -J CELLxGENE
#BSUB -G team292
#BSUB -q long
#BSUB -M 256G
#BSUB -R "select[mem>256G] rusage[mem=256G] span[hosts=1]"
#BSUB -n 1
#BSUB -o /nfs/team292/eg22/Metabolic-Tasks/CELLxGENE/MT-summary.%J.out
#BSUB -e /nfs/team292/eg22/Metabolic-Tasks/CELLxGENE/MT-summary.%J.err

set -eo pipefail

/lustre/scratch126/cellgen/team292/eg22/miniforge3/envs/single_cell/bin/python3.10 /nfs/team292/eg22/FARM-Bsubs/CELLxGENE/Summarize-scCellFie-Results.py
