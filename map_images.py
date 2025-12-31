#!/usr/bin/env python3
"""
Script to process data3.out and create a CSV mapping of images.

This script reads image paths from data3.out file and extracts:
- Dataset name (top-level folder)
- File path (complete path)
- Class (subfolder containing the image, representing medical diagnosis)

Output format: dataset,file_path,class
"""

import csv
import os
import re
import pandas as pd
from pathlib import Path

duplicates = set({})

def is_valid_image_path(line):
    """Check if the line is a valid image path (not file listing output)."""
    # Filter out lines that start with file listing patterns
    if re.match(r'^-[rwx-]+\s+\d+', line):  # Unix file listing format
        return False
    if re.match(r'^\s*$', line):  # Empty lines
        return False
    if not re.search(r'\.(jpg|jpeg|png|tiff|tif)$', line, re.IGNORECASE):  # Not an image
        return False
    if line.startswith('total '):  # Directory totals
        return False
    
    return True


def extract_dataset_and_class(file_path, image_class_df=None):
    """
    Extract dataset name, subclass and class from file path.
    
    Expected format: dataset/[subdirs...]/subclass/filename.ext (class is derrived from subclass)
    
    Args:
        file_path (str): Full path to the image file
        
    Returns:
        dict: Contains dataset, subclass, class, and additional fields, or None if can't parse
    """
    parts = file_path.split('/')
    
    if len(parts) < 3:
        return None
    
    dataset = parts[0]
    modality = None
    modality_subtype = None
    plane = None

    if file_path in duplicates:
        print(f"Duplicate file path found: {file_path}")
        return None
    duplicates.add(file_path)

    if dataset in ('stroke'):
        subclass_name = parts[-3]  # Folder containing the image file
    else:
        subclass_name = parts[-2]  # Folder containing the image file

    original_subclass_name = subclass_name  # Keep original for later use

    # for Br35H
    if dataset == 'Br35H':
        modality = 'MRI'
        plane = 'axial'
        if subclass_name == 'yes':
            class_name = 'tumor'
        else:
            class_name = 'normal'
        
        subclass_name = ''
        original_subclass_name = ''

    # for Br35H
    elif dataset == 'figshare':
        modality = 'MRI'
        modality_subtype = 'T1C+'
        subclass_name = parts[-2]  # Folder containing the image file
        if subclass_name == '1':
            subclass_name = 'meningioma'
        elif subclass_name == '2':
            subclass_name = 'glioma'
        elif subclass_name == '3':
            subclass_name = 'pituitary tumor'
        else: 
            subclass_name = ''
        
        original_subclass_name = subclass_name
        class_name = 'tumor'

    # process brain-tumor-mri-datasets
    elif dataset == 'brain-tumor-mri-dataset': # skip this for now
        return None
        # modality = 'MRI'
        # plane = 'axial'
        # if subclass_name in ('meningioma', 'glioma', 'pituitary'):
        #     class_name = 'tumor'
        # else:
        #     class_name = 'normal'

    # process images 17
    elif dataset == 'images-17':
        modality = 'MRI'
        plane = 'axial'
        folder = parts[1] # e.g .Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+
        if 'T1C+' in folder:
            modality_subtype = 'T1C+'
        elif 'T1' in folder:
            modality_subtype = 'T1'
        elif 'T2' in folder:
            modality_subtype = 'T2' 
        else:
            modality_subtype = 'other'

        subclass_name = subclass_name.lower()
        if 'normal' in subclass_name:
            class_name = 'normal'
            subclass_name = ''
            original_subclass_name = ''
            
        elif 'outros' in subclass_name:
            class_name = 'Other Types of Abnormalities'
        else:
            class_name = 'tumor'

        subclass_name = subclass_name.split(' ')[0]
        subclass_name = '' if subclass_name == 'outros' or subclass_name == 'other'  else subclass_name
        original_subclass_name = subclass_name if subclass_name == '' else original_subclass_name


    # process images 44c
    elif dataset == 'images-44c':
        modality = 'MRI'

        folder = parts[1]  # e.g. images-44c/Tuberculoma T1C+/45bc34000e8c5846936e4dc2821a18_big_gallery.jpeg

        if 'T1C' in folder:
            modality_subtype = 'T1C+'
        elif 'T1' in folder:
            modality_subtype = 'T1'
        elif 'T2' in folder:
            modality_subtype = 'T2' 
        else:
            modality_subtype = 'other'

        subclass_name = subclass_name.lower()

        if 'normal' in subclass_name:
            class_name = 'normal'
            subclass_name = ''
        else:
            class_name = 'tumor'

        subclass_name = subclass_name.split(' ')[0]
        subclass_name = '' if subclass_name == 'outros' or subclass_name == 'other' else subclass_name

        original_subclass_name = subclass_name  # Keep original for later use

        if subclass_name in ('astrocitoma', 'ependimoma', 'ganglioglioma', 'glioblastoma', 'oligodendroglioma'):
            if subclass_name == 'astrocitoma':
                original_subclass_name = 'astrocytoma'
            elif subclass_name == 'ependimoma':
                original_subclass_name = 'ependymoma'

            subclass_name = 'glioma'

    # process multimodal
    elif dataset == 'multimodal':
        return None  # Skip multimodal dataset as it is not needed
        folder =  parts[2]  # multimodal/Dataset/Brain Tumor MRI images/Healthy/mri_healthy (348).jpg

        if "MRI" in folder:
            modality = 'MRI'
        else:
            modality = 'CT'


        if 'healthy' in subclass_name.lower():
            class_name = 'normal'
        else:
            class_name = 'tumor'
        subclass_name = subclass_name.lower()

    # process sclerosis
    elif dataset == 'sclerosis':
        modality = 'MRI'
        folder = parts[2]  # e.g. sclerosis/MS/MS Axial_crop/A1 (259).png

        if 'Axial' in folder:
            plane = 'axial'
        elif 'Saggital' in folder:
            plane = 'sagittal'
        else:
            plane = 'other'
        
        modality = 'MRI'
        modality_subtype = 'FLAIR'
        

        subclass_name = subclass_name.lower()
        
        if 'control' in subclass_name:
            class_name = 'normal'
        else:
            class_name = 'multiple_sclerosis'
        
        original_subclass_name = ''
        subclass_name = ''
    
    # process stroke
    elif dataset == 'stroke':
        subclass_name = subclass_name.lower()
        modality = 'CT'
        plane = 'axial'

        if subclass_name == 'external_test':
            # return None  # Skip external test set
            file_name = int(parts[-1].split('.')[0])
            subclass_name = '' if image_class_df[image_class_df['image_id'] == file_name].iloc[0]['Stroke'] == 1 else 'normal'

        subclass_name = 'ischemic' if subclass_name == 'ischemia' else subclass_name
        original_subclass_name = 'Ischemic' if original_subclass_name == 'Ischemia' else original_subclass_name
        
        subclass_name = 'hemorrhagic' if subclass_name == 'bleeding' else subclass_name
        original_subclass_name = 'Hemorrhagic' if original_subclass_name == 'Bleeding' else original_subclass_name
        

        if 'normal' in subclass_name:
            class_name = 'normal'
            subclass_name = ''
            original_subclass_name = ''
        else:
            class_name = 'stroke'
    

    result = {
        'dataset': dataset,
        'file_path': file_path,
        'subclass': subclass_name,
        'original_class': original_subclass_name,
        'class': class_name,
        'modality': modality,
        'modality_subtype': modality_subtype,
        'plane': plane
    }

    return result


def process_data_file(input_file, output_file, aisd_file, test_file=None):
    """
    Process the data3.out file and create a CSV mapping.
    
    Args:
        input_file (str): Path to data3.out file
        output_file (str): Path to output CSV file
    """
    processed_paths = []
    skipped_lines = 0
    image_class_df = {}

    if test_file:
        print(f"Reading from: {test_file}")

        image_class_df = pd.read_csv(test_file, encoding='utf-8')
    
    print(f"Reading from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not is_valid_image_path(line):
                print(f"Skipping invalid line {line_num}: {line}")
                skipped_lines += 1
                continue
            
            result = extract_dataset_and_class(line, image_class_df)

            if result:
                processed_paths.append(result)
            else:
                print(f"Warning: Could not parse line {line_num}: {line}")
    
    print(f"Processed {len(processed_paths)} valid image paths")
    print(f"Skipped {skipped_lines} invalid lines")
    
    # Write to CSV
    print(f"Writing to: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['dataset', 'file_path', 'class', 'original_class', 'subclass', 'modality', 'modality_subtype', 'plane']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in processed_paths:
            writer.writerow(entry)
        
        
        csvfile.writelines([','.join(row) + '\n' for row in csv.reader(open(str(aisd_file)))])
        
        
    
    print(f"CSV file created with {len(processed_paths)} entries")
    
    # Print some statistics
    print("\n--- Statistics ---")
    
    # Count by dataset
    dataset_counts = {}
    for entry in processed_paths:
        dataset = entry['dataset']
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print("Images per dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")
    
    # Count by class
    class_counts = {}
    for entry in processed_paths:
        class_name = entry['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nUnique classes found: {len(class_counts)}")
    print("Top 10 classes by image count:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes[:10]:
        print(f"  {class_name}: {count}")


def main():
    """Main function to run the script."""
    # Define file paths
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    input_file = parent_dir / 'chatmed' / 'data5.out'
    output_file = parent_dir  / 'chatmed' / 'image_mapping.csv'
    test_file = parent_dir / 'chatmed' /  'brain_stroke_ct_dataset_labels.csv'
    aisd_file = parent_dir  / 'chatmed' / 'aisd.csv'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    try:
        process_data_file(str(input_file), str(output_file), str(aisd_file), str(test_file))
        print(f"\nSuccess! Output saved to: {output_file}")
        
        #open(str(output_file), 'a', newline='').writelines([','.join(row) + '\n' for row in csv.reader(open(str(aisd_file)))])
        print(f"\nSuccess! Output from aisd dataset saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
