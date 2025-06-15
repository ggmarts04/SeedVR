#!/bin/bash

set -ex # Exit immediately if a command exits with a non-zero status, and print commands.

echo "Installing NVIDIA Apex from source..."
# Using --no-cache-dir to ensure it rebuilds if needed.
# The --global-option flags are for older pip versions with apex, might not be needed for newer pip/apex but kept for safety.
# Ensure that build essentials (like g++, make) and the correct CUDA toolkit are available
# in the Docker base image for this to succeed. The chosen -devel image should suffice.
pip install git+https://github.com/NVIDIA/apex.git --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"

echo "Verifying Apex installation..."
python -c "import apex; print('Apex imported successfully and can be found by Python.')"

echo "NVIDIA Apex installation script (part of Docker build) completed."
