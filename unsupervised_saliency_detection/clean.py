import shutil
import os

# List of directories to delete
directories_to_delete = ["masks", "results", "Test"]

# Current directory
current_directory = os.getcwd()

for dir_name in directories_to_delete:
    dir_path = os.path.join(current_directory, dir_name)

    # Check if directory exists
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # Remove the directory
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_name}' deleted successfully.")
    else:
        print(f"Directory '{dir_name}' does not exist in the current directory.")