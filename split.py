import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_stratified_subsets(csv_path, output_dir=None, n_subsets=5, subset_size=0.1):
    """
    Create 5 non-overlapping subsets with 10% of images each, maintaining distribution
    of class, original_class, subclass, modality, modality_subtype, and plane.
    
    Args:
        csv_path (str): Path to imag_mapping.csv
        output_dir (str): Directory to save subset files (optional)
        n_subsets (int): Number of subsets to create (default: 5)
        subset_size (float): Proportion of data in each subset (default: 0.1)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Total images: {len(df)}")
    
    # Create a combined stratification column
    stratify_cols = ['class', 'original_class', 'subclass', 'modality', 'modality_subtype', 'plane']
    
    # Handle missing values by filling with 'Unknown'
    df_stratify = df[stratify_cols].fillna('Unknown')
    
    # Create combined stratification key
    df['stratify_key'] = df_stratify.apply(lambda x: '|'.join(x.astype(str)), axis=1)
    
    # Check distribution of stratification keys
    key_counts = df['stratify_key'].value_counts()
    print(f"Number of unique combinations: {len(key_counts)}")
    print(f"Minimum group size: {key_counts.min()}")
    
    # Filter out groups that are too small to split
    min_group_size = n_subsets  # Need at least n_subsets samples per group
    valid_keys = key_counts[key_counts >= min_group_size].index
    
    if len(valid_keys) < len(key_counts):
        small_groups = len(key_counts) - len(valid_keys)
        print(f"Warning: {small_groups} groups have fewer than {min_group_size} samples and will be distributed randomly")
    
    # Separate data into stratifiable and non-stratifiable
    df_stratifiable = df[df['stratify_key'].isin(valid_keys)].copy()
    df_small_groups = df[~df['stratify_key'].isin(valid_keys)].copy()
    print(f"Stratifiable groups: {len(df_stratifiable)} ")
    print(f"Non-stratifiable groups: {len(df_small_groups)}")

    # Initialize subsets
    subsets = [pd.DataFrame() for _ in range(n_subsets)]
    
    # Process stratifiable data
    if len(df_stratifiable) > 0:
        remaining_data = df_stratifiable.copy()
        
        for i in range(n_subsets):
            # Calculate the proportion for this split
            # We want each subset to have subset_size of the original data
            remaining_proportion = len(remaining_data) / len(df_stratifiable)
            current_split_size = subset_size / remaining_proportion
            
            if current_split_size >= 1.0:
                # Take all remaining data for this subset
                subsets[i] = pd.concat([subsets[i], remaining_data])
                remaining_data = pd.DataFrame()
                break
            
            # Stratified split
            try:
                subset_data, remaining_data = train_test_split(
                    remaining_data,
                    test_size=1-current_split_size,
                    stratify=remaining_data['stratify_key'],
                    random_state=42 + i
                )
                subsets[i] = pd.concat([subsets[i], subset_data])
            except ValueError as e:
                print(f"Stratification failed for subset {i}: {e}")
                # Fall back to random split
                subset_data, remaining_data = train_test_split(
                    remaining_data,
                    test_size=1-current_split_size,
                    random_state=42 + i
                )
                subsets[i] = pd.concat([subsets[i], subset_data])
        
        # Add remaining data to last subset
        if len(remaining_data) > 0:
            subsets[n_subsets - 1] = pd.concat([subsets[n_subsets - 1], remaining_data])
    
    # Distribute small groups randomly across subsets
    if len(df_small_groups) > 0:
        np.random.seed(42)
        subset_assignments = np.random.choice(n_subsets, size=len(df_small_groups))
        
        for i in range(n_subsets):
            mask = subset_assignments == i
            if mask.sum() > 0:
                subsets[i] = pd.concat([subsets[i], df_small_groups[mask]])
    
    # Remove the stratify_key column from final subsets
    for i in range(n_subsets):
        if 'stratify_key' in subsets[i].columns:
            subsets[i] = subsets[i].drop('stratify_key', axis=1)
    
    # Print subset statistics
    print("\nSubset Statistics:")
    for i, subset in enumerate(subsets):
        print(f"Subset {i+1}: {len(subset)} images ({len(subset)/len(df)*100:.1f}%)")
    
    # Save subsets to files
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    for i, subset in enumerate(subsets):
        filename = f"subset_{i+1}.csv"
        if output_dir:
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
        
        subset.to_csv(filepath, index=False)
        print(f"Saved {filename} with {len(subset)} images")
    
    # Verify no overlaps
    all_indices = set()
    for subset in subsets:
        subset_indices = set(subset.index)
        overlap = all_indices.intersection(subset_indices)
        if overlap:
            print(f"Warning: Found overlapping indices: {len(overlap)}")
        all_indices.update(subset_indices)
    
    print(f"\nTotal images in subsets: {len(all_indices)}")
    print(f"Original total: {len(df)}")
    print("✓ No overlaps detected" if len(all_indices) == len(df) else "⚠ Possible data loss or overlap")
    
    return subsets

def print_distribution_comparison(original_df, subsets, columns):
    """Print distribution comparison between original data and subsets"""
    print("\nDistribution Comparison:")
    
    for col in columns:
        if col in original_df.columns:
            print(f"\n{col.upper()}:")
            orig_dist = original_df[col].value_counts(normalize=True).sort_index()
            
            print(f"{'Original':<15}", end="")
            for i in range(len(subsets)):
                print(f"Subset {i+1:<15}", end="")
            print()
            
            for value in orig_dist.index:
                print(f"{str(value):<15}{orig_dist[value]:<15.3f}", end="")
                for subset in subsets:
                    if col in subset.columns and len(subset) > 0:
                        subset_dist = subset[col].value_counts(normalize=True)
                        if value in subset_dist.index:
                            print(f"{subset_dist[value]:<15.3f}", end="")
                        else:
                            print(f"{'0.000':<15}", end="")
                    else:
                        print(f"{'N/A':<15}", end="")
                print()

if __name__ == "__main__":
    # Path to your image_mapping.csv file
    csv_path = "subsets/subset_6.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please ensure the file is in the current directory or provide the correct path.")
    else:
        # Create subsets
        subsets = create_stratified_subsets(csv_path=csv_path, output_dir="subsets_phase2", n_subsets=2, subset_size=0.5)
        
        # Read original data for comparison
        original_df = pd.read_csv(csv_path)
        
        # Compare distributions
        comparison_cols = ['class', 'original_class', 'subclass', 'modality', 'modality_subtype', 'plane']
        print_distribution_comparison(original_df, subsets, comparison_cols)