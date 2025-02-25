"""
Fix for SAE-Vis Import Issues in Google Colab

This file contains code to fix the import issues with sae_vis in Google Colab.
Copy and paste this code at the beginning of your Colab notebook before 
attempting to import sae_vis modules.
"""

# Add this to the top of your Colab notebook
import os
import sys
import importlib

def fix_sae_vis_imports():
    """Fix the sae_vis imports by properly setting up the Python path"""
    # The path where sae_vis is located
    sae_vis_path = "/content/r1crossencode/sae_vis-crosscoder-vis"
    
    # Check if the path exists
    if not os.path.exists(sae_vis_path):
        print(f"Path {sae_vis_path} does not exist. Please check the path.")
        return False
    
    # Let's see what's in this directory
    print(f"Contents of {sae_vis_path}:")
    print(os.listdir(sae_vis_path))
    
    # Check if there's a sae_vis subdirectory
    sae_vis_subdir = os.path.join(sae_vis_path, "sae_vis")
    
    if os.path.exists(sae_vis_subdir) and os.path.isdir(sae_vis_subdir):
        print(f"Found sae_vis subdirectory. Contents:")
        print(os.listdir(sae_vis_subdir))
        
        # Add the parent directory to sys.path to make "sae_vis" importable
        sys.path.insert(0, sae_vis_path)
        
        # For debugging, let's try importing directly
        try:
            # You should be able to import like this after adding to sys.path
            import sae_vis.model_fns
            print("✅ Successfully imported sae_vis.model_fns!")
            return True
        except ImportError as e:
            print(f"⚠️ Still can't import sae_vis.model_fns: {e}")
            # Alternative approach - create a symbolic link in the current directory
            print("Trying alternative approach with symbolic link...")
            try:
                if not os.path.exists("/content/sae_vis"):
                    # Create a symbolic link to make imports easier
                    os.symlink(sae_vis_subdir, "/content/sae_vis")
                    print("Created symbolic link: /content/sae_vis -> " + sae_vis_subdir)
                # Add current directory to path
                if "/content" not in sys.path:
                    sys.path.insert(0, "/content")
                # Try import again
                import sae_vis.model_fns
                print("✅ Successfully imported sae_vis.model_fns via symlink!")
                return True
            except Exception as e2:
                print(f"❌ Failed with symbolic link approach: {e2}")
                return False
    else:
        print("⚠️ No sae_vis subdirectory found.")
        # Check if Python files exist directly in the given path
        if os.path.exists(os.path.join(sae_vis_path, "model_fns.py")):
            print("Found model_fns.py directly in specified path.")
            
            # If the modules are directly in this path, add it to sys.path
            sys.path.insert(0, sae_vis_path)
            
            # Create a package structure - create __init__.py if it doesn't exist
            init_file = os.path.join(sae_vis_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write("# Auto-generated __init__.py for sae_vis package\n")
                print(f"Created {init_file}")
            
            # Try importing the module directly
            try:
                # This should work if the modules are directly in the specified path
                model_fns = importlib.import_module("model_fns")
                print("✅ Successfully imported model_fns as a direct module!")
                
                # Provide a wrapper to access these modules with the expected structure
                class SaeVisWrapper:
                    def __init__(self):
                        self.model_fns = importlib.import_module("model_fns")
                        self.data_config_classes = importlib.import_module("data_config_classes")
                        self.data_storing_fns = importlib.import_module("data_storing_fns")
                        self.data_fetching_fns = importlib.import_module("data_fetching_fns")
                
                # Expose this wrapper for the adapter to use
                import builtins
                builtins.sae_vis_wrapper = SaeVisWrapper()
                print("Created sae_vis_wrapper in builtins for adapter to use")
                
                return True
            except ImportError as e:
                print(f"❌ Failed to import model_fns directly: {e}")
                return False
        else:
            print("❌ Neither sae_vis subdirectory nor model_fns.py found. Check your directory structure.")
            return False

def modify_import_adapter():
    """Modify your import_sae_vis_modules function to work with either import approach"""
    sae_vis_compat_path = "/content/r1crossencode/sae_vis_adapter/sae_vis_compat.py"
    
    if not os.path.exists(sae_vis_compat_path):
        print(f"⚠️ Could not find {sae_vis_compat_path}")
        return
    
    # Read the current file
    with open(sae_vis_compat_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add a fallback mechanism to check for builtins.sae_vis_wrapper
    modified_content = content.replace(
        "sae_vis_model_fns = importlib.import_module(\"sae_vis.model_fns\")",
        """# Try standard import first
try:
    sae_vis_model_fns = importlib.import_module("sae_vis.model_fns")
except ImportError:
    # Fall back to builtins wrapper if available
    import builtins
    if hasattr(builtins, 'sae_vis_wrapper'):
        print("Using sae_vis_wrapper from builtins")
        sae_vis_model_fns = builtins.sae_vis_wrapper.model_fns
    else:
        raise ImportError("Could not import sae_vis.model_fns and no wrapper available")"""
    )
    
    # Write the modified file
    with open(sae_vis_compat_path, "w", encoding="utf-8") as f:
        f.write(modified_content)
    
    print("✅ Updated import_sae_vis_modules function with fallback mechanism")

# Run the fix
print("Running sae_vis import fix...")
if fix_sae_vis_imports():
    print("✅ Fixed sae_vis imports! You should now be able to import sae_vis modules.")
    
    # Optionally modify the adapter
    print("Do you want to modify the adapter code for more robust imports? (y/n)")
    print("If running this in a notebook, answer with:")
    print("modify_import_adapter() # to modify")
    print("print('Skipping adapter modification') # to skip")
else:
    print("❌ Could not fix sae_vis imports automatically.")
    print("Please check your directory structure and file paths.")
    print("Manual steps to try:")
    print("1. Ensure sae_vis-crosscoder-vis is correctly cloned from github")
    print("2. Check that model_fns.py and other modules exist in the expected location")
    print("3. Try setting up a symbolic link: !ln -s /path/to/sae_vis /content/sae_vis") 

