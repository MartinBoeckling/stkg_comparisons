import pandas as pd
import re
import argparse

def hybrid_dataset_creation(base_datapath: str, vector_datapath: str, output_hybrid_path: str):
    """Creates a hybrid dataset by merging the base dataset with a vector dataset.
    Args:
        base_datapath (str): The file path to the base dataset in Parquet format.
        vector_datapath (str): The file path to the vector dataset in Parquet format.
        output_hybrid_path (str): The file path to save the resulting hybrid dataset in Parquet format.
    """
    base_dataset = pd.read_parquet(base_datapath)
    base_dataset = base_dataset.rename({'h3ID': 'ID', 'date': 'DATE'}, axis=1)
    vector_dataset = pd.read_parquet(vector_datapath)
    if re.findall("worldkg", vector_datapath):
        base_dataset_date = pd.unique(base_dataset['DATE'])
        vector_dataset['DATE'] = [base_dataset_date for i in vector_dataset.index]
        vector_dataset = vector_dataset.explode('DATE', ignore_index=True)
        hybrid_dataset = pd.merge(base_dataset, vector_dataset, on=['ID', 'DATE'], how='inner')
    else:
        hybrid_dataset = base_dataset.merge(vector_dataset, left_on=['ID', 'DATE'], right_on=["ID", "DATE"], how='inner')
    hybrid_dataset = hybrid_dataset.dropna()
    hybrid_dataset.to_parquet(output_hybrid_path, engine='pyarrow')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for the overall vector merge")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    hybrid_dataset_creation(args.base_path, args.vector_path, args.output_path)