import pandas as pd
import numpy as np
import os
from datetime import datetime

def quantize_value(value):
    """
    Quantize continuous values to match user input format.
    Maps to: 0.0, 0.25, 0.5, 0.75, 1.0
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        # If conversion fails, return 0.5 as default
        return 0.5
    
    if value < 0.125:
        return 0.0
    elif value < 0.375:
        return 0.25
    elif value < 0.625:
        return 0.5
    elif value < 0.875:
        return 0.75
    else:
        return 1.0

def quantize_dataset(input_csv='data/animalfulldata.csv', output_csv=None, backup=True):
    """
    Quantize all numeric columns in the dataset (except animal_name).
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, overwrites input)
        backup: If True, creates a backup of the original file
    """
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"❌ Error: File '{input_csv}' not found!")
        return
    
    print(f"📂 Loading dataset from '{input_csv}'...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    print(f"✅ Loaded {len(df)} animals with {len(df.columns)} columns")
    
    # Create backup if requested
    if backup and output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = input_csv.replace('.csv', f'_backup_{timestamp}.csv')
        try:
            df.to_csv(backup_name, index=False)
            print(f"💾 Backup created: '{backup_name}'")
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")
    
    # Identify columns to quantize (all except 'animal_name')
    columns_to_quantize = [col for col in df.columns if col != 'animal_name']
    
    print(f"\n🔄 Quantizing {len(columns_to_quantize)} feature columns...")
    
    # Track statistics
    total_values = 0
    changed_values = 0
    
    # Quantize each column
    for col in columns_to_quantize:
        original = df[col].copy()
        df[col] = df[col].apply(quantize_value)
        
        # Count changes
        differences = (original != df[col])
        changed_values += differences.sum()
        total_values += len(df[col])
    
    # Calculate statistics
    change_percentage = (changed_values / total_values * 100) if total_values > 0 else 0
    
    print(f"\n📊 Quantization Statistics:")
    print(f"   Total values processed: {total_values:,}")
    print(f"   Values changed: {changed_values:,}")
    print(f"   Percentage changed: {change_percentage:.2f}%")
    
    # Show distribution of values after quantization
    print(f"\n📈 Value Distribution After Quantization:")
    all_values = df[columns_to_quantize].values.flatten()
    unique, counts = np.unique(all_values, return_counts=True)
    for val, count in zip(unique, counts):
        percentage = (count / len(all_values)) * 100
        print(f"   {val:.2f}: {count:>8,} ({percentage:>5.2f}%)")
    
    # Save the quantized dataset
    output_file = output_csv if output_csv else input_csv
    
    try:
        df.to_csv(output_file, index=False)
        print(f"\n✅ Quantized dataset saved to '{output_file}'")
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")
        return
    
    print(f"\n🎉 Quantization complete!")
    
    # Show sample of changes
    print(f"\n🔍 Sample of first 5 animals:")
    print(df.head()[['animal_name'] + columns_to_quantize[:5]])

def main():
    """Main function with user interaction."""
    print("=" * 60)
    print("🧮 Dataset Quantization Tool")
    print("=" * 60)
    print("\nThis tool will quantize all feature values to:")
    print("  0.0 (No), 0.25 (Probably Not), 0.5 (Maybe),")
    print("  0.75 (Probably), 1.0 (Yes)")
    print()
    
    # Get input file
    input_file = input("Enter input CSV filename [animalfulldata.csv]: ").strip()
    if not input_file:
        input_file = "animalfulldata.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ File '{input_file}' not found!")
        return
    
    # Ask about overwriting
    print(f"\n⚠️  This will modify '{input_file}'")
    overwrite = input("Do you want to overwrite the original file? (y/n) [y]: ").strip().lower()
    
    if overwrite == 'n':
        output_file = input("Enter output filename: ").strip()
        if not output_file:
            print("❌ Output filename cannot be empty!")
            return
        backup = False
    else:
        output_file = None
        backup = True
    
    print()
    quantize_dataset(input_file, output_file, backup)

if __name__ == "__main__":
    main()