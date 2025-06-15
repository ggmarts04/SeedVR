import os
import requests
import logging
import sys

# Configure basic logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# --- Configuration ---
COLOR_FIX_URL = "https://raw.githubusercontent.com/pkuliyi2015/sd-webui-stablesr/master/srmodule/colorfix.py"
# Note: Changed URL to raw.githubusercontent.com for direct file access.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DESTINATION_DIR = os.path.join(CURRENT_DIR, "projects", "video_diffusion_sr")
DESTINATION_PATH = os.path.join(DESTINATION_DIR, "color_fix.py")

def ensure_color_fix_file():
    """
    Downloads the color_fix.py file if it doesn't already exist.
    Returns True if the file exists or was successfully downloaded, False otherwise.
    """
    if os.path.exists(DESTINATION_PATH):
        logging.info(f"'color_fix.py' already exists at {DESTINATION_PATH}. Skipping download.")
        return True

    logging.info(f"'color_fix.py' not found at {DESTINATION_PATH}. Attempting to download from {COLOR_FIX_URL}...")

    try:
        # Ensure the destination directory exists
        os.makedirs(DESTINATION_DIR, exist_ok=True)
        logging.info(f"Ensured destination directory exists: {DESTINATION_DIR}")

        response = requests.get(COLOR_FIX_URL, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(DESTINATION_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Successfully downloaded 'color_fix.py' to {DESTINATION_PATH}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download 'color_fix.py': {e}", exc_info=True)
        return False
    except OSError as e:
        logging.error(f"Failed to save 'color_fix.py' to {DESTINATION_PATH}: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during color_fix.py download: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logging.info("Running download_color_fix.py directly to ensure its presence.")
    if ensure_color_fix_file():
        logging.info("color_fix.py setup complete.")
    else:
        logging.error("color_fix.py setup failed.")
        sys.exit(1) # Exit with error if standalone setup fails
