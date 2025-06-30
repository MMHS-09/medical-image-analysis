from pathlib import Path
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define input and output directories (adjust these paths as needed)
input_root = Path('/home/mhs/thesis/fastMRI_breast_IDS_150_300_DCM')
output_root = Path('/home/mhs/thesis/img_files/fastMRI_breast_IDS_150_300_DCM')

img_size = (256, 256)  # Desired output image size

# Ensure the output root directory exists
output_root.mkdir(parents=True, exist_ok=True)

# Find all .dcm files recursively in the input directory
dcm_files = list(input_root.rglob('*.dcm'))

# Process each DICOM file with a progress bar
for dcm_file in tqdm(dcm_files, desc="Converting DICOM to JPEG"):
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dcm_file)
        pixel_array = ds.pixel_array
        
        # Ensure it's a single-frame image (2D array)
        if pixel_array.ndim == 2:
            # Normalize pixel values to 0-255 for JPEG
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:
                normalized = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
            else:
                # If min and max are the same, create a blank image
                normalized = np.zeros_like(pixel_array)
            normalized = normalized.astype(np.uint8)
            
            # Convert to Pillow Image and resize to 128x128
            img = Image.fromarray(normalized)
            resized_img = img.resize(img_size, Image.LANCZOS)
            
            # Compute the output path, maintaining the folder structure
            rel_path = dcm_file.relative_to(input_root)
            jpg_rel_path = rel_path.with_suffix('.png')
            output_path = output_root / jpg_rel_path
            
            # Create the output directory if it doesn’t exist
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            resized_img.save(output_path, 'PNG')
        else:
            print(f"Skipping {dcm_file}: Multi-frame image detected.")
    except Exception as e:
        print(f"Error processing {dcm_file}: {e}")