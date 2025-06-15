#!/bin/bash

set -ex # Exit immediately if a command exits with a non-zero status, and print commands.

# Update package lists and install git if not present (it usually is)
# apt-get update && apt-get install -y git # Might not be needed if base image has it

# Install NVIDIA Apex from source
# This is often necessary because pip wheels for apex can be outdated or not available for all CUDA/Python versions.
# Using --no-cache-dir to ensure it rebuilds if needed.
# The --global-option flags are for older pip versions with apex, might not be needed for newer pip/apex but kept for safety.
echo "Installing NVIDIA Apex..."
pip install git+https://github.com/NVIDIA/apex.git --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"

# Check if apex is installed and usable by Python
python -c "import apex; print('Apex imported successfully')"

echo "Apex installation attempted."
echo "RunPod setup script completed."

# The handler.py will then be called by the RunPod environment.
