import os
import pandas as pd

def combine_csvs(directory, output_file):
    """
    Combines all CSV files in the specified directory into one CSV file.

    Args:
        directory (str): Path to the directory containing CSV files.
        output_file (str): Path for the output combined CSV file.

    Returns:
        None
    """
    # List to hold dataframes
    dataframes = []

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        # Check if the file is a CSV
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            print(f"Reading {file_path}")
            # Read the CSV and append to the list
            dataframes.append(pd.read_csv(file_path))

    # Combine all dataframes into one
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        # Write the combined dataframe to the output file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined CSV saved to {output_file}")
    else:
        print("No CSV files found in the directory.")

# Example usage
directory_path = 'anomaly detector'  # Replace with your directory containing CSV files
output_csv = 'combined_output1.csv'  # Replace with your desired output CSV file name
combine_csvs(directory_path, output_csv)
