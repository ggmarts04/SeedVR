# Base image from RunPod - choose one with PyTorch, Python 3.10, and CUDA 12.1
# Example: runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# Using a more generic reference that RunPod might resolve or a specific one:
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8     LC_ALL=C.UTF-8     PYTHONUNBUFFERED=1     DEBIAN_FRONTEND=noninteractive     RUNPOD_PROJECT_ROOT=/app

# Install git and build-essential
# build-essential provides compilers (g++, etc.) and tools like make, which are needed for compiling Apex.
RUN apt-get update && apt-get install -y --no-install-recommends     git     build-essential     && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR ${RUNPOD_PROJECT_ROOT}

# Copy the entire project repository content
COPY . .

# Make runpod.sh executable and run it to install Apex
# This bakes Apex into the Docker image.
RUN chmod +x ./runpod.sh && ./runpod.sh

# Install Python dependencies
# Ensure pip is up-to-date first
RUN python -m pip install --upgrade pip     && python -m pip install --no-cache-dir -r requirements.txt

# Expose the port RunPod expects for the handler (if applicable, often not needed for serverless)
# EXPOSE 8000 

# Set the default command.
# For RunPod serverless, the environment often looks for handler.py.
# The base RunPod images usually have an entrypoint that handles invoking the handler.
# If not, a CMD like ["python", "-u", "handler.py"] might be needed,
# or a specific RunPod entrypoint script.
# For now, relying on RunPod's standard Python worker behavior.
# If a specific CMD is required by RunPod for custom images, this might need adjustment.
# Often, the base image's ENTRYPOINT handles this.
# No specific CMD needed here if handler.py is automatically picked up.
# A simple CMD to keep the container running if necessary:
# CMD ["sleep", "infinity"]
# However, usually, the base image's entrypoint handles the worker lifecycle.

# Verify handler.py exists (optional sanity check for logs)
RUN if [ ! -f "/app/handler.py" ]; then echo "handler.py not found in /app!" && exit 1; fi

# The actual invocation of handler.py is typically managed by RunPod's infrastructure
# for serverless functions, using their standard Python worker entrypoint.
