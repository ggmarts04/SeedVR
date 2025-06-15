#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.
# Removed 'x' for now to make output less verbose unless an error actually occurs.

echo "Installing NVIDIA Apex from source..."
echo "Attempting to install Apex. This step may take a while and produce a lot of output."
echo "Full output from pip install command will be shown."

# Try to install Apex and capture more detailed output on failure
if pip install git+https://github.com/NVIDIA/apex.git --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"; then
    echo "Apex pip install command completed."
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Apex pip install command failed."
    echo "The error likely occurred in the output above this message."
    echo "Please check for compiler errors, missing CUDA components, or other issues."
    echo "Ensure the base Docker image has a compatible CUDA toolkit and all dev libraries."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1 # Ensure script exits with error code if pip install fails
fi

echo "Verifying Apex installation..."
if python -c "import apex; print('Apex imported successfully and can be found by Python.')"; then
    echo "Apex verification successful."
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Apex was installed but cannot be imported by Python."
    echo "This might indicate an issue with the installation or Python environment."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1 # Ensure script exits with error code if Python verification fails
fi

echo "NVIDIA Apex installation script (part of Docker build) completed successfully."
