import h3
import s2cell
import pandas as pd
import argparse

if __name__ == "__main__":
    """Script for translating H3 grid IDs to S2 grid IDs and saving the result to a parquet file.

    The script reads a parquet file containing H3 grid IDs, converts each H3 ID to its corresponding
    latitude and longitude, translates these coordinates to an S2 cell ID, and saves the result to a 
    specified target parquet file.

    Args:
        --path (str): The path to the input parquet file containing H3 grid IDs.
        --target_path (str): The path to save the output parquet file with H3 and S2 grid IDs.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='h3 to S2 grid tranlsation')
    parser.add_argument('--path', type=str, required=True, help='Path to the input parquet file')
    parser.add_argument('--target_path', type=str, required=True, help='Path to save the output parquet file')
    args = parser.parse_args()
    # Read data from the input parquet file
    data = pd.read_parquet(args.path)
    # Get unique H3 grid IDs from the data
    h3_data = data['h3_id'].unique()
    h3_s2_list = []
    # Convert each H3 ID to its corresponding S2 cell ID
    for h3_id in h3_data:
        latitude, longitude = h3.h3_to_geo(h3_id)
        s2_cell = s2cell.lat_lon_to_cell_id(lat=latitude, lon=longitude, level=13)
        h3_s2_list.append(tuple((h3_id, s2_cell)))
    # Create a DataFrame from the list of H3 to S2 mappings
    h3_s2_df = pd.DataFrame(h3_s2_list, columns=['h3ID', 's2CellID'])
    # Save the DataFrame to the specified target parquet file
    h3_s2_df.to_parquet(args.target_path)