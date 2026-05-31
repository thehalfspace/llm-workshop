#!/bin/bash

# Set apptainer paths
export APPTAINER_BINDPATH="/oscar/home/,/oscar/scratch/,/oscar/data"
export APPTAINER_CACHEDIR="tmp"

# 2. Direct Triton to the internal driver path inside the NGC container
# export APPTAINERENV_LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real/lib:$LD_LIBRARY_PATH"
