import os
import shutil
import random

# Path to the directory containing all PNG files
source_dir = '/home/stud1/Desktop/ipr1_210809/IPR_project/MetaF2N-main/datasets/ffhq256'

# Path to the directory where the subset of files will be copied
destination_dir = '/home/stud1/Desktop/ipr1_210809/IPR_project/MetaF2N-main/datasets/ffhq_sample'

# Number of files you want in the subset
num_files_to_copy = 10000 # Change this number to your desired subset size

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all PNG files in the source directory
png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

# Randomly select a subset of PNG files
subset_files = random.sample(png_files, num_files_to_copy)

# Copy the selected PNG files to the destination directory
for filename in subset_files:
    src_file = os.path.join(source_dir, filename)
    dest_file = os.path.join(destination_dir, filename)
    
    # Copy the file from source to destination
    shutil.copy(src_file, dest_file)
    print(f"Copied {filename} to {destination_dir}")

print(f"Subset of {num_files_to_copy} PNG files copied to {destination_dir}.")
