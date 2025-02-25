"""
SAE-Vis Setup Instructions for Google Colab

This file contains instructions for properly setting up the SAE-Vis visualization 
in a Google Colab environment.
"""

# 1. First, clone the repositories if not already done
'''
!git clone https://github.com/path/to/sae_vis-crosscoder-vis.git
!git clone https://github.com/path/to/your/crosscoder-repo.git  # if needed
'''

# 2. Install required dependencies
'''
!pip install einops numpy torch transformers
'''

# 3. Setup - Add this code to your notebook

import os
import sys
import importlib

# Check the actual path to your sae_vis directory
# The correct path should be the parent directory containing the "sae_vis" folder
# In your case, it appears to be: /content/r1crossencode/sae_vis-crosscoder-vis

def setup_sae_vis(sae_vis_path):
    """Set up the Python path and verify the sae_vis location"""
    print(f"Setting up sae_vis from path: {sae_vis_path}")
    
    # First ensure the parent path is in sys.path
    parent_path = os.path.dirname(sae_vis_path)
    if parent_path not in sys.path:
        sys.path.append(parent_path)
    
    # Also add the path itself, which might be needed
    if sae_vis_path not in sys.path:
        sys.path.append(sae_vis_path)
    
    # Debug: Show what's in the directory
    print(f"Contents of {sae_vis_path}:")
    try:
        print(os.listdir(sae_vis_path))
    except Exception as e:
        print(f"Error accessing {sae_vis_path}: {e}")
    
    # Check if there's a sae_vis subdirectory
    sae_vis_subdir = os.path.join(sae_vis_path, "sae_vis")
    if os.path.exists(sae_vis_subdir) and os.path.isdir(sae_vis_subdir):
        print(f"Found sae_vis subdirectory. Contents:")
        print(os.listdir(sae_vis_subdir))
        
        # This is the key fix - make the parent directory (containing sae_vis/) 
        # available as a module source
        if sae_vis_path not in sys.path:
            sys.path.append(sae_vis_path)
    else:
        # If there's no subdirectory, maybe this path itself is the sae_vis module
        if os.path.exists(os.path.join(sae_vis_path, "model_fns.py")):
            print("Found model_fns.py directly in the specified path.")
            print("Will use this directory as the module.")
            # Make sure the directory containing the modules is in sys.path
            if sae_vis_path not in sys.path:
                sys.path.append(sae_vis_path)

# Use this function before importing sae_vis modules
setup_sae_vis("/content/r1crossencode/sae_vis-crosscoder-vis")

# Then proceed with your adapter imports
# Make sure your import_sae_vis_modules function correctly handles the path

# 4. Test importing the modules directly (for diagnostics)
'''
try:
    import sae_vis.model_fns
    print("Success importing sae_vis.model_fns directly!")
except Exception as e:
    print(f"Error importing directly: {e}")
'''

# 5. If the structure is different than expected, you might need to adapt
# your import_sae_vis_modules function or manually import the modules
'''
# Option 1: If sae_vis is in sae_vis-crosscoder-vis/sae_vis
from sae_vis_crosscoder_vis.sae_vis import model_fns

# Option 2: If Python modules are directly in sae_vis-crosscoder-vis/
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_fns", 
    "/content/r1crossencode/sae_vis-crosscoder-vis/model_fns.py"
)
model_fns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_fns)
''' 