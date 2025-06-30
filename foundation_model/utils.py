#!/usr/bin/env python3
"""
Dataset preparation and validation utilities
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from collections import Counter
import json

def validate_classification_dataset(dataset_path: str, dataset_name: str, expected_classes: list):
    """Validate classification dataset structure"""
    print(f"\nValidating classification dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    issues = []
    stats = {}
    
    # Check each class directory
    for class_name in expected_classes:
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            issues.append(f"Missing class directory: {class_name}")
            stats[class_name] = 0
            continue
        
        # Count images in class directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_count = 0
        
        for file in os.listdir(class_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
        
        stats[class_name] = image_count
        
        if image_count == 0:
            issues.append(f"No images found in class: {class_name}")
    
    # Print statistics
    print(f"📊 Class distribution:")
    total_images = sum(stats.values())
    for class_name, count in stats.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    print(f"📈 Total images: {total_images}")
    
    # Check for class imbalance
    if total_images > 0:
        min_count = min(stats.values())
        max_count = max(stats.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 5:
            issues.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
        elif imbalance_ratio > 2:
            print(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f})")
    
    # Report issues
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✅ Dataset validation passed!")
        return True


def validate_segmentation_dataset(dataset_path: str, dataset_name: str):
    """Validate segmentation dataset structure"""
    print(f"\nValidating segmentation dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    issues = []
    
    # Get all files
    files = os.listdir(dataset_path)
    
    # Separate images and masks
    image_files = []
    mask_files = []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in files:
        if file.endswith(':Zone.Identifier'):
            continue  # Skip zone identifier files
        
        if any(file.lower().endswith(ext) for ext in image_extensions):
            if 'mask' in file.lower():
                mask_files.append(file)
            else:
                image_files.append(file)
    
    print(f"📊 Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Check for matching pairs
    matched_pairs = 0
    unmatched_images = []
    unmatched_masks = []
    
    for img_file in image_files:
        img_base = os.path.splitext(img_file)[0]
        
        # Look for corresponding mask
        mask_found = False
        for mask_file in mask_files:
            mask_base = os.path.splitext(mask_file)[0]
            if img_base in mask_base or mask_base.replace('_mask', '') == img_base:
                matched_pairs += 1
                mask_found = True
                break
        
        if not mask_found:
            unmatched_images.append(img_file)
    
    # Check for unmatched masks
    for mask_file in mask_files:
        mask_base = os.path.splitext(mask_file)[0].replace('_mask', '')
        
        img_found = False
        for img_file in image_files:
            img_base = os.path.splitext(img_file)[0]
            if mask_base == img_base or mask_base in img_base:
                img_found = True
                break
        
        if not img_found:
            unmatched_masks.append(mask_file)
    
    print(f"📈 Matched pairs: {matched_pairs}")
    
    if unmatched_images:
        issues.append(f"Images without masks: {len(unmatched_images)}")
        if len(unmatched_images) <= 5:
            print(f"  Unmatched images: {unmatched_images}")
    
    if unmatched_masks:
        issues.append(f"Masks without images: {len(unmatched_masks)}")
        if len(unmatched_masks) <= 5:
            print(f"  Unmatched masks: {unmatched_masks}")
    
    # Report issues
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✅ Dataset validation passed!")
        return True


def validate_all_datasets(config_path: str):
    """Validate all datasets specified in config"""
    print("🔍 Validating all datasets...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = config['paths']['data_root']
    
    all_valid = True
    validation_results = {}
    
    # Validate classification datasets
    print("\n" + "="*50)
    print("CLASSIFICATION DATASETS")
    print("="*50)
    
    for dataset_config in config.get('classification_datasets', []):
        dataset_name = dataset_config['name']
        classes = dataset_config['classes']
        dataset_path = os.path.join(data_root, 'classification', dataset_name)
        
        is_valid = validate_classification_dataset(dataset_path, dataset_name, classes)
        validation_results[f"classification_{dataset_name}"] = is_valid
        all_valid = all_valid and is_valid
    
    # Validate segmentation datasets
    print("\n" + "="*50)
    print("SEGMENTATION DATASETS")
    print("="*50)
    
    for dataset_config in config.get('segmentation_datasets', []):
        dataset_name = dataset_config['name']
        dataset_path = os.path.join(data_root, 'segmentation', dataset_name)
        
        is_valid = validate_segmentation_dataset(dataset_path, dataset_name)
        validation_results[f"segmentation_{dataset_name}"] = is_valid
        all_valid = all_valid and is_valid
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    for dataset_name, is_valid in validation_results.items():
        status = "✅ PASSED" if is_valid else "❌ FAILED"
        print(f"{dataset_name}: {status}")
    
    print(f"\nOverall validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
    
    return all_valid, validation_results


def create_dataset_info(config_path: str, output_path: str = "dataset_info.json"):
    """Create detailed dataset information"""
    print(f"📝 Creating dataset information file: {output_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = config['paths']['data_root']
    dataset_info = {
        'creation_date': str(Path(__file__).stat().st_mtime),
        'data_root': data_root,
        'classification_datasets': {},
        'segmentation_datasets': {}
    }
    
    # Process classification datasets
    for dataset_config in config.get('classification_datasets', []):
        dataset_name = dataset_config['name']
        classes = dataset_config['classes']
        dataset_path = os.path.join(data_root, 'classification', dataset_name)
        
        if os.path.exists(dataset_path):
            class_stats = {}
            total_images = 0
            
            for class_name in classes:
                class_path = os.path.join(dataset_path, class_name)
                if os.path.exists(class_path):
                    image_count = len([f for f in os.listdir(class_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                    class_stats[class_name] = image_count
                    total_images += image_count
                else:
                    class_stats[class_name] = 0
            
            dataset_info['classification_datasets'][dataset_name] = {
                'classes': classes,
                'class_distribution': class_stats,
                'total_images': total_images,
                'num_classes': len(classes)
            }
    
    # Process segmentation datasets
    for dataset_config in config.get('segmentation_datasets', []):
        dataset_name = dataset_config['name']
        num_classes = dataset_config['num_classes']
        dataset_path = os.path.join(data_root, 'segmentation', dataset_name)
        
        if os.path.exists(dataset_path):
            files = os.listdir(dataset_path)
            
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                          and 'mask' not in f.lower() and not f.endswith(':Zone.Identifier')]
            
            mask_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                         and 'mask' in f.lower() and not f.endswith(':Zone.Identifier')]
            
            dataset_info['segmentation_datasets'][dataset_name] = {
                'num_classes': num_classes,
                'num_images': len(image_files),
                'num_masks': len(mask_files),
                'image_mask_pairs': min(len(image_files), len(mask_files))
            }
    
    # Save dataset info
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Dataset information saved to: {output_path}")
    return dataset_info


def setup_directories(config_path: str):
    """Setup required directories based on config"""
    print("📁 Setting up directories...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    directories = [
        config['paths']['data_root'],
        config['paths']['models_dir'],
        config['paths']['logs_dir'],
        config['paths']['results_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")
    
    print("✅ Directory setup complete!")


def main():
    parser = argparse.ArgumentParser(description='Dataset utilities for Medical Image Analysis')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['validate', 'info', 'setup'],
                       help='Action to perform')
    parser.add_argument('--output', type=str, help='Output file path (for info action)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    if args.action == 'validate':
        is_valid, results = validate_all_datasets(args.config)
        sys.exit(0 if is_valid else 1)
    
    elif args.action == 'info':
        output_path = args.output or 'dataset_info.json'
        create_dataset_info(args.config, output_path)
    
    elif args.action == 'setup':
        setup_directories(args.config)


if __name__ == "__main__":
    main()
