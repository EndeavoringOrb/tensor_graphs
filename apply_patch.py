import subprocess
import os
import sys

def apply_patch(patch_file):
    if not os.path.exists(patch_file):
        print(f"Error: Patch file '{patch_file}' not found.")
        return

    try:
        # We use the -p1 flag, which is standard for patches generated 
        # from the root of the repository.
        print(f"Applying patch from {patch_file}...")
        
        # subprocess.run will call the system 'patch' command
        result = subprocess.run(
            ["patch", "-p1", "-i", patch_file],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("Patch applied successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("Error: Failed to apply patch.")
        print(f"Return code: {e.returncode}")
        print(f"Error output:\n{e.stderr}")
    except FileNotFoundError:
        print("Error: The 'patch' command was not found. Please ensure it is installed and in your PATH.")

if __name__ == "__main__":
    # If the file is named patch.json (as per your prompt), 
    # rename it to patch.diff or .patch for clarity, 
    # or just pass the filename here.
    apply_patch("patch.json")