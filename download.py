import os
import shutil
from glob import glob

import kagglehub

# Download dataset
path = kagglehub.dataset_download("ggtejas/aptos-diffmic-ben")

print("Downloaded to:", path)

# Source: cropped PNGs
src_dir = os.path.join(path, "cropped")

# Destination
dst_dir = "./dataset/aptos/train/"
os.makedirs(dst_dir, exist_ok=True)

# Move all PNG files
for file in glob(os.path.join(src_dir, "*.png")):
    shutil.move(file, os.path.join(dst_dir, os.path.basename(file)))

print("All images moved to", dst_dir)
