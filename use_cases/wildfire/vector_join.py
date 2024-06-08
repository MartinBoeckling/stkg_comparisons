import pandas as pd
import re
import argparse

def hybrid_dataset_creation(base_datapath: str, vector_datapath: str, output_hybrid_path: str) -> None:
    base_dataset = pd.read_parquet(base_datapath)
    base_dataset = base_dataset.rename({'h3ID': 'ID', 'date': 'DATE'}, axis=1)
    vector_dataset = pd.read_parquet(vector_datapath)
    if re.findall('KnowWhereGraph/vector', vector_datapath):
        base_dataset['DATE'] = pd.to_datetime(base_dataset['DATE'])
        base_dataset['YEAR'] = base_dataset['DATE'].dt.strftime("%Y-01-01")
        vector_dataset = vector_dataset.rename({'ID': 's2CellID'}, axis=1)
        gridCellID = pd.read_parquet('data/wildfire/base_data/h3_s2_data.parquet')
        gridCellID['s2CellID'] = 'http://stko-kwg.geog.ucsb.edu/lod/resource/s2.level13.' + gridCellID['s2CellID'].astype(str)
        vector_dataset = gridCellID.merge(vector_dataset, how='left', on='s2CellID')
        hybrid_dataset = base_dataset.merge(vector_dataset, left_on=['ID', 'YEAR'], right_on=['h3ID', 'DATE'], how='inner')
        hybrid_dataset = hybrid_dataset.drop(['h3ID', 's2CellID', 'DATE_y', 'YEAR'], axis=1)
        hybrid_dataset = hybrid_dataset.rename({'DATE_x': 'DATE'}, axis=1)
    elif re.findall("worldkg", vector_datapath):
        base_dataset_date = pd.unique(base_dataset['DATE'])
        vector_dataset['DATE'] = [base_dataset_date for i in vector_dataset.index]
        vector_dataset = vector_dataset.explode('DATE', ignore_index=True)
        hybrid_dataset = pd.merge(base_dataset, vector_dataset, on=['ID', 'DATE'], how='inner')
    else:
        base_dataset['DATE'] = pd.to_datetime(base_dataset['DATE'])
        base_dataset['YEAR'] = base_dataset['DATE'].dt.strftime("%Y-01-01")
        hybrid_dataset = base_dataset.merge(vector_dataset, left_on=['ID', 'YEAR'], right_on=["ID", "DATE"], how='inner')
        hybrid_dataset = hybrid_dataset.drop(['DATE_y', 'YEAR'], axis=1)
        hybrid_dataset = hybrid_dataset.rename({'DATE_x': 'DATE'}, axis=1)
    hybrid_dataset = hybrid_dataset.dropna()
    hybrid_dataset.to_parquet(output_hybrid_path, engine='pyarrow')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for the overall vector merge")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    hybrid_dataset_creation(args.base_path, args.vector_path, args.output_path)