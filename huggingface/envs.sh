#!/bin/bash

# Set apptainer paths
export APPTAINER_BINDPATH="/oscar/home/,/oscar/scratch/,/oscar/data"
export APPTAINER_CACHEDIR="tmp"

# 2. Force host GPU drivers to bind to a dedicated path inside the container
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,/usr/lib64/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1"

# 3. Inject the destination path directly into the container's execution environment
export APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
