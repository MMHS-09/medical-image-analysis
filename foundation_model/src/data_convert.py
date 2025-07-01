import os
import shutil
from pathlib import Path
import datetime
from PIL import Image
import glob

def reorganize_brain_stroke_dataset():
    """
    Reorganize the brain_stroke dataset by moving images and masks to the root directory
    and renaming them with a consistent naming convention.
    
    Original structure:
    - brain_stroke/PNG/[original_filenames].png
    - brain_stroke/MASKS/[original_filenames].png
    
    New structure:
    - brain_stroke/img1.png, img2.png, ... (original images)
    - brain_stroke/img1_mask.png, img2_mask.png, ... (corresponding masks)
    """
    
    # Define paths
    dataset_root = Path("/home/jobayer/research/mhs/medical-image-analysis/foundation_model/data/segmentation/brain_stroke")
    png_folder = dataset_root / "PNG"
    masks_folder = dataset_root / "MASKS"
    
    # Check if folders exist
    if not png_folder.exists():
        print(f"Error: PNG folder not found at {png_folder}")
        return
    
    if not masks_folder.exists():
        print(f"Error: MASKS folder not found at {masks_folder}")
        return
    
    # Get list of image files from PNG folder
    png_files = sorted([f for f in png_folder.glob("*.png")])
    mask_files = sorted([f for f in masks_folder.glob("*.png")])
    
    print(f"Found {len(png_files)} PNG files and {len(mask_files)} mask files")
    
    # Verify that we have matching numbers of images and masks
    if len(png_files) != len(mask_files):
        print("Error: Number of PNG files and mask files don't match!")
        return
    
    # Verify that filenames match between PNG and MASKS folders
    png_names = [f.name for f in png_files]
    mask_names = [f.name for f in mask_files]
    
    if png_names != mask_names:
        print("Error: PNG and mask filenames don't match!")
        print("PNG files that don't have corresponding masks:")
        for name in set(png_names) - set(mask_names):
            print(f"  - {name}")
        print("Mask files that don't have corresponding PNG files:")
        for name in set(mask_names) - set(png_names):
            print(f"  - {name}")
        return
    
    print("All checks passed. Starting reorganization...")
    
    # Create a mapping of original filename to new filename
    reorganization_log = []
    
    # Move and rename files
    for i, (png_file, mask_file) in enumerate(zip(png_files, mask_files), 1):
        # New filenames
        new_img_name = f"img{i}.png"
        new_mask_name = f"img{i}_mask.png"
        
        # New paths
        new_img_path = dataset_root / new_img_name
        new_mask_path = dataset_root / new_mask_name
        
        # Move and rename the image
        shutil.move(str(png_file), str(new_img_path))
        print(f"Moved {png_file.name} -> {new_img_name}")
        
        # Move and rename the mask
        shutil.move(str(mask_file), str(new_mask_path))
        print(f"Moved {mask_file.name} -> {new_mask_name}")
        
        # Log the mapping for reference
        reorganization_log.append({
            'original_name': png_file.name,
            'new_img_name': new_img_name,
            'new_mask_name': new_mask_name
        })
    
    # Remove empty folders
    try:
        png_folder.rmdir()
        print(f"Removed empty PNG folder: {png_folder}")
    except OSError as e:
        print(f"Could not remove PNG folder: {e}")
    
    try:
        masks_folder.rmdir()
        print(f"Removed empty MASKS folder: {masks_folder}")
    except OSError as e:
        print(f"Could not remove MASKS folder: {e}")
    
    # Save reorganization log
    log_file = dataset_root / "reorganization_log.txt"
    with open(log_file, 'w') as f:
        f.write("Brain Stroke Dataset Reorganization Log\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total files processed: {len(reorganization_log)}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Filename Mapping:\n")
        f.write("-" * 20 + "\n")
        
        for entry in reorganization_log:
            f.write(f"Original: {entry['original_name']}\n")
            f.write(f"  -> Image: {entry['new_img_name']}\n")
            f.write(f"  -> Mask:  {entry['new_mask_name']}\n\n")
    
    print(f"\nReorganization complete!")
    print(f"Processed {len(reorganization_log)} image-mask pairs")
    print(f"Reorganization log saved to: {log_file}")
    print(f"\nFinal structure:")
    print(f"  - {len(reorganization_log)} images: img1.png, img2.png, ..., img{len(reorganization_log)}.png")
    print(f"  - {len(reorganization_log)} masks: img1_mask.png, img2_mask.png, ..., img{len(reorganization_log)}_mask.png")

def verify_reorganization():
    """
    Verify that the reorganization was successful by checking the final structure.
    """
    dataset_root = Path("/home/jobayer/research/mhs/medical-image-analysis/foundation_model/data/segmentation/brain_stroke")
    
    # Count images and masks
    img_files = sorted(dataset_root.glob("img*.png"))
    mask_files = sorted(dataset_root.glob("img*_mask.png"))
    
    # Filter out mask files from img_files to get only main images
    main_img_files = [f for f in img_files if not f.name.endswith("_mask.png")]
    
    print(f"\nVerification Results:")
    print(f"Main images found: {len(main_img_files)}")
    print(f"Mask images found: {len(mask_files)}")
    
    # Check if we have matching pairs
    missing_masks = []
    for img_file in main_img_files:
        img_num = img_file.name.replace("img", "").replace(".png", "")
        expected_mask = dataset_root / f"img{img_num}_mask.png"
        if not expected_mask.exists():
            missing_masks.append(f"img{img_num}_mask.png")
    
    if missing_masks:
        print(f"Warning: Missing masks for {len(missing_masks)} images:")
        for mask in missing_masks:
            print(f"  - {mask}")
    else:
        print("✓ All images have corresponding masks")
    
    # Check if old folders still exist
    old_png_folder = dataset_root / "PNG"
    old_masks_folder = dataset_root / "MASKS"
    
    if old_png_folder.exists():
        print(f"Warning: Old PNG folder still exists at {old_png_folder}")
    else:
        print("✓ Old PNG folder successfully removed")
    
    if old_masks_folder.exists():
        print(f"Warning: Old MASKS folder still exists at {old_masks_folder}")
    else:
        print("✓ Old MASKS folder successfully removed")

def reorganize_pmram_classification_dataset():
    """
    Reorganize the PMRAM classification dataset by converting non-PNG images to PNG
    and renaming them with a consistent naming convention.
    
    Structure:
    - pmram/class1/[various image files] -> pmram/class1/img1.png, img2.png, ...
    - pmram/class2/[various image files] -> pmram/class2/img1.png, img2.png, ...
    - etc.
    """
    
    # Define paths
    dataset_root = Path("/home/jobayer/research/mhs/medical-image-analysis/foundation_model/data/classification/pmram")
    
    # Check if dataset exists
    if not dataset_root.exists():
        print(f"Error: Dataset folder not found at {dataset_root}")
        return
    
    print(f"Processing PMRAM classification dataset at: {dataset_root}")
    
    # Get all class directories
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("Error: No class directories found in the dataset")
        return
    
    print(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    reorganization_log = []
    total_processed = 0
    
    for class_dir in sorted(class_dirs):
        print(f"\nProcessing class: {class_dir.name}")
        
        # Get all image files in this class directory
        image_files = []
        for ext in supported_extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        print(f"  Found {len(image_files)} image files")
        
        if not image_files:
            print(f"  No image files found in {class_dir.name}")
            continue
        
        class_log = {
            'class_name': class_dir.name,
            'files_processed': 0,
            'files_converted': 0,
            'file_mappings': []
        }
        
        # Process each image file
        for i, img_file in enumerate(image_files, 1):
            try:
                new_filename = f"img{i}.png"
                new_path = class_dir / new_filename
                
                # If file is already PNG and has the correct name, skip
                if img_file.suffix.lower() == '.png' and img_file.name == new_filename:
                    print(f"    {img_file.name} already in correct format")
                    class_log['file_mappings'].append({
                        'original': img_file.name,
                        'new': new_filename,
                        'converted': False
                    })
                    continue
                
                # Load and convert image
                with Image.open(img_file) as img:
                    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create white background for transparent images
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as PNG
                    img.save(new_path, 'PNG')
                
                # Remove original file if it's different from the new one
                if img_file != new_path:
                    img_file.unlink()
                    converted = img_file.suffix.lower() != '.png'
                    print(f"    {img_file.name} -> {new_filename}" + 
                          (" (converted)" if converted else " (renamed)"))
                    class_log['files_converted'] += 1 if converted else 0
                else:
                    converted = False
                    print(f"    {img_file.name} -> {new_filename} (processed)")
                
                class_log['file_mappings'].append({
                    'original': img_file.name,
                    'new': new_filename,
                    'converted': converted
                })
                class_log['files_processed'] += 1
                total_processed += 1
                
            except Exception as e:
                print(f"    Error processing {img_file.name}: {e}")
        
        reorganization_log.append(class_log)
        print(f"  Completed: {class_log['files_processed']} files processed, {class_log['files_converted']} converted")
    
    # Save reorganization log
    log_file = dataset_root / "reorganization_log.txt"
    with open(log_file, 'w') as f:
        f.write("PMRAM Classification Dataset Reorganization Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {total_processed}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for class_log in reorganization_log:
            f.write(f"Class: {class_log['class_name']}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Files processed: {class_log['files_processed']}\n")
            f.write(f"Files converted: {class_log['files_converted']}\n\n")
            
            f.write("File mappings:\n")
            for mapping in class_log['file_mappings']:
                status = " (converted)" if mapping['converted'] else " (renamed)" if mapping['original'] != mapping['new'] else ""
                f.write(f"  {mapping['original']} -> {mapping['new']}{status}\n")
            f.write("\n")
    
    print(f"\nReorganization complete!")
    print(f"Total files processed: {total_processed}")
    print(f"Reorganization log saved to: {log_file}")

def verify_pmram_reorganization():
    """
    Verify that the PMRAM classification dataset reorganization was successful.
    """
    dataset_root = Path("/home/jobayer/research/mhs/medical-image-analysis/foundation_model/data/classification/pmram")
    
    if not dataset_root.exists():
        print(f"Error: Dataset folder not found at {dataset_root}")
        return
    
    print(f"\nVerifying PMRAM classification dataset at: {dataset_root}")
    
    # Get all class directories
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("Error: No class directories found")
        return
    
    print(f"\nVerification Results:")
    print(f"Class directories found: {len(class_dirs)}")
    
    total_images = 0
    for class_dir in sorted(class_dirs):
        # Count PNG files with img*.png pattern
        png_files = list(class_dir.glob("img*.png"))
        # Count any non-PNG files that might remain
        non_png_files = []
        for ext in ['.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp']:
            non_png_files.extend(class_dir.glob(f"*{ext}"))
            non_png_files.extend(class_dir.glob(f"*{ext.upper()}"))
        
        print(f"  {class_dir.name}: {len(png_files)} PNG files", end="")
        if non_png_files:
            print(f", {len(non_png_files)} non-PNG files remaining")
            for f in non_png_files:
                print(f"    Warning: {f.name} not converted")
        else:
            print(" ✓")
        
        total_images += len(png_files)
    
    print(f"\nTotal images: {total_images}")
    print("Verification complete!")

if __name__ == "__main__":
    print("Dataset Reorganization Script")
    print("=" * 40)
    
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1]
        
        if task == "brain_stroke":
            print("Brain Stroke Dataset Reorganization")
            print("-" * 35)
            
            # First, let's see the current state
            dataset_root = Path("/home/jobayer/research/mhs/medical-image-analysis/foundation_model/data/segmentation/brain_stroke")
            print(f"Dataset location: {dataset_root}")
            
            # Check if reorganization has already been done
            if not (dataset_root / "PNG").exists() and not (dataset_root / "MASKS").exists():
                print("It appears the dataset has already been reorganized.")
                verify_reorganization()
            else:
                # Run the reorganization
                reorganize_brain_stroke_dataset()
                
                # Verify the results
                verify_reorganization()
        
        elif task == "pmram":
            print("PMRAM Classification Dataset Reorganization")
            print("-" * 42)
            
            # Run the reorganization
            reorganize_pmram_classification_dataset()
            
            # Verify the results
            verify_pmram_reorganization()
        
        else:
            print(f"Unknown task: {task}")
            print("Available tasks: brain_stroke, pmram")
    
    else:
        print("Usage: python data_convert.py <task>")
        print("Available tasks:")
        print("  brain_stroke - Reorganize brain stroke segmentation dataset")
        print("  pmram        - Reorganize PMRAM classification dataset")
    
    # PMRAM classification dataset reorganization
    pmram_dataset_root = Path("/home/jobayer/research/mhs/medical-image-analysis/foundation_model/data/classification/pmram")
    if pmram_dataset_root.exists():
        print("\nPMRAM Classification Dataset Detected")
        print("=" * 45)
        
        # Check if reorganization has already been done
        if all((d.glob("img*.png") for d in pmram_dataset_root.iterdir() if d.is_dir())):
            print("It appears the PMRAM dataset has already been reorganized.")
            verify_pmram_reorganization()
        else:
            # Run the reorganization
            reorganize_pmram_classification_dataset()
            
            # Verify the results
            verify_pmram_reorganization()
    else:
        print("No PMRAM Classification Dataset found")