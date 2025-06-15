import subprocess
import os
import random
import string
import requests
import logging
import sys
# At the top with other imports
from huggingface_hub import snapshot_download, HfFolder # Added HfFolder
import glob # To check for model files (no longer needed with flag file, but good to have if strategy changes)
# Removed import for ensure_color_fix_file as color_fix.py is now part of the repo


# Configure basic logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Determine paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "ckpts")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
SEEDVR_SCRIPT_PATH = os.path.join(CURRENT_DIR, "projects", "inference_seedvr2_3b.py") # Default to 3B model
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "ByteDance-Seed/SeedVR2-3B") # Default model, allow override by env var
MODEL_DOWNLOAD_COMPLETE_FLAG = os.path.join(MODEL_DIR, ".model_download_complete")


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Ensure MODEL_DIR exists at the script level for the flag file and cache
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_models_downloaded():
    """Downloads models from Hugging Face Hub if not already present."""
    if os.path.exists(MODEL_DOWNLOAD_COMPLETE_FLAG):
        logging.info(f"Model download flag found at {MODEL_DOWNLOAD_COMPLETE_FLAG}. Skipping download.")
        return

    logging.info(f"Models not found or download flag not present in {MODEL_DIR}. Starting download for {MODEL_REPO_ID}...")

    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        try:
            HfFolder.save_token(hf_token)
            logging.info("Hugging Face token found and saved.")
        except Exception as e:
            logging.warning(f"Failed to save Hugging Face token: {e}. Proceeding with anonymous download if possible.")
    else:
        logging.info("Hugging Face token not found. Proceeding with anonymous download if possible.")

    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=MODEL_DIR,
            cache_dir=os.path.join(MODEL_DIR, ".cache"), # Store cache within the model dir, hidden
            local_dir_use_symlinks=False, # Important for some environments
            resume_download=True,
            allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"], # As per README
        )
        logging.info(f"Successfully downloaded models from {MODEL_REPO_ID} to {MODEL_DIR}")
        # Create a flag file to indicate successful download
        with open(MODEL_DOWNLOAD_COMPLETE_FLAG, 'w') as f:
            f.write(f"Download complete for {MODEL_REPO_ID}")
        logging.info(f"Created model download flag at {MODEL_DOWNLOAD_COMPLETE_FLAG}")

    except Exception as e:
        logging.error(f"Failed to download models from {MODEL_REPO_ID}: {e}", exc_info=True)
        # If download fails, we should probably raise an error or exit
        # as the worker won't be able to function.
        raise # Reraise the exception to signal a critical failure

def download_video(url, save_path):
    """Downloads a video from a URL."""
    logging.info(f"Downloading video from {url} to {save_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Video downloaded successfully to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        return None

def run_seedvr_job(job):
    """
    Main handler function for RunPod Serverless.
    Takes a job input, downloads the video, runs SeedVR inference,
    and returns the path to the processed video.
    """
    input_data = job.get('input', {})
    video_url = input_data.get('video_url')

    if not video_url:
        return {"error": "Missing video_url in input"}

    # Generate a unique filename for the downloaded video
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    input_filename = f"input_video_{random_suffix}.mp4" # Assuming mp4, might need to be more robust
    input_video_path = os.path.join(OUTPUT_DIR, input_filename)

    # Download the video
    downloaded_video_path = download_video(video_url, input_video_path)
    if not downloaded_video_path:
        return {"error": f"Failed to download video from {video_url}"}

    # --- Model download and color_fix.py logic will be added here in later steps ---
    # For now, assume models are present and color_fix.py is in place.

    # Prepare for SeedVR inference
    # The inference script (inference_seedvr2_3b.py) will be modified
    # in a later step to accept --video_path directly and handle output dir.

    # For now, we'll construct a command.
    # This will need refinement based on how inference_seedvr2_3b.py is modified.
    # We also need to determine how to pass output resolution and other parameters.
    # These could also come from the job input.

    output_height = input_data.get('res_h', 720) # Default to 720p
    output_width = input_data.get('res_w', 1280) # Default to 1280p
    seed = input_data.get('seed', random.randint(0, 2**32 - 1))

    # The output of the script will be in a subfolder of OUTPUT_DIR,
    # named after the input video. We need to determine this exact path.
    # For now, let's assume the modified script saves it as processed_video.mp4 in OUTPUT_DIR
    processed_video_filename = f"processed_{random_suffix}.mp4"
    processed_video_path = os.path.join(OUTPUT_DIR, processed_video_filename)

    # Placeholder for where the script will save its output.
    # The actual inference script will need to be told where to save its output,
    # or we need to know its convention.
    # For now, the inference script is expected to save to OUTPUT_DIR/some_subfolder/video.mp4
    # This part will be firmed up when we modify the inference script.

    # Construct the command for torchrun
    # NUM_GPUS should ideally be determined by RunPod environment or set to 1 for simplicity if applicable
    num_gpus = input_data.get('num_gpus', 1) # Default to 1 GPU

    # The inference script projects/inference_seedvr2_3b.py has been modified to accept
    # a file path for --video_path and a specific file path for --output_video_path.
    # We provide the downloaded_video_path and the predetermined processed_video_path.

    command = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        SEEDVR_SCRIPT_PATH,
        "--video_path", downloaded_video_path, # This is a file path, as expected by modified script
        "--output_video_path", processed_video_path, # Provide specific output file path
        "--seed", str(seed),
        "--res_h", str(output_height),
        "--res_w", str(output_width),
        # "--sp_size", "NUM_SP" # sp_size might be needed depending on model and GPU, default for now
    ]

    logging.info(f"Running SeedVR inference with command: {' '.join(command)}")

    try:
        # Note: The original script uses torchrun. For a serverless worker,
        # it might be simpler to import and call a Python function directly from
        # the inference script if possible, after refactoring it.
        # For now, we stick to subprocess as per the original README's inference style.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=CURRENT_DIR)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logging.error(f"SeedVR inference failed. Return code: {process.returncode}")
            logging.error(f"Stdout: {stdout.decode()}")
            logging.error(f"Stderr: {stderr.decode()}")
            return {"error": "SeedVR inference failed", "stdout": stdout.decode(), "stderr": stderr.decode()}

        logging.info("SeedVR inference completed.")
        logging.info(f"Stdout: {stdout.decode()}")

        # --- Output file handling ---
        # The inference script (inference_seedvr2_3b.py) has been modified to save
        # its output directly to the path specified by --output_video_path.
        # In our case, this is `processed_video_path`.
        # We just need to check if this file was created.

        if os.path.exists(processed_video_path):
            logging.info(f"Processed video successfully saved at: {processed_video_path}")
            # TODO: Upload to a hosting service and return URL, or handle as per RunPod's output mechanism.
            # For now, just returning the path within the worker.
            return {"processed_video_path": processed_video_path}
        else:
            logging.error(f"Processed video not found at the expected path: {processed_video_path}")
            # Include stderr in the error if the file is missing, as it might contain clues.
            return {"error": "Processed video file not found after inference.", "stderr": stderr.decode()}

    except Exception as e:
        logging.error(f"An unexpected error occurred during SeedVR job: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

# This is the entry point for RunPod serverless.
# The function name must be 'handler'.
def handler(job):
    # Ensure models are downloaded (runs once per worker cold start ideally)
    # If this fails, it will raise an exception and the job won't proceed.
    try:
        ensure_models_downloaded()
    except Exception as e:
        logging.error(f"Model download/check failed critically: {e}", exc_info=True)
        # Return an error structure that RunPod can understand if initialization fails.
        return {"error": f"Critical error: Model setup failed: {str(e)}", "status": "failed_initialization"}

    # color_fix.py is now part of the repository in projects/video_diffusion_sr/
    # No download needed. The inference script will import it directly.
    # If it's missing, the inference script's import will fail, which is an error state.
    logging.info("Assuming color_fix.py is present in projects/video_diffusion_sr/")

    return run_seedvr_job(job)

if __name__ == "__main__":
    # Example usage for local testing (not part of RunPod execution)
    # Create a dummy job input
    sample_job = {
        "input": {
            "video_url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4", # A real sample URL
            "res_h": 360, # smaller for faster local test if possible
            "res_w": 640
        }
    }

    # To test locally, you would need:
    # 1. A running environment with all dependencies.
    # 2. Models downloaded to ./ckpts
    # 3. inference_seedvr2_3b.py to be runnable and configured.
    # 4. (Potentially) A dummy color_fix.py in place if the script tries to import it.

    # Create dummy color_fix.py for local test if it's imported by inference script
    color_fix_dir = os.path.join(CURRENT_DIR, "projects", "video_diffusion_sr")
    os.makedirs(color_fix_dir, exist_ok=True)
    if not os.path.exists(os.path.join(color_fix_dir, "color_fix.py")):
        with open(os.path.join(color_fix_dir, "color_fix.py"), "w") as f:
            f.write("# Dummy color_fix.py for testing
")
            f.write("class ColorFix:
")
            f.write("    def __init__(self, *args, **kwargs):
")
            f.write("        pass
")
            f.write("    def __call__(self, *args, **kwargs):
")
            f.write("        return args[0]
") # return the first arg (image)


    logging.info("Starting local test of handler.")
    # Before running local test, ensure models are available or the script handles their absence gracefully.
    # The handler currently assumes the inference script `inference_seedvr2_3b.py`
    # can be called and will find its models.
    # Also, the inference script expects --video_path to be a FOLDER.
    # The inference script `inference_seedvr2_3b.py` has now been modified
    # to accept a file path for --video_path and --output_video_path.
    # Local testing would still require models and dependencies.

    # result = handler(sample_job)
    # logging.info(f"Local test result: {result}")
    logging.info("Local test setup for handler.py is complex and relies on model availability and full environment.")
    logging.info("The handler.py structure is updated for modified inference script. Further steps involve dependency and model handling.")
