#!/bin/bash
#SBATCH --job-name=hprepmed

#SBATCH -D ./
#SBATCH -o hprepmed.%j.%N.out
#SBATCH -e hprepmed.%j.%N.err

#SBATCH --get-user-env
#SBATCH --export=NONE

#SBATCH --clusters=serial
#SBATCH --partition=serial_std

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

python benchmarking/scripts/01_python/prepare_data.py \
    --root "$HRHSA_DATA_ROOT" \
    --size 32768 \
    --bands 6 \
    --storage-chunk 1024 \
    --points 5000000 \
    --workers "$SLURM_CPUS_PER_TASK"
