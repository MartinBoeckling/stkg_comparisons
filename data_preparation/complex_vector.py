import torch
import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from tqdm import tqdm
import geopandas as gpd
import argparse

def pca_compression(data: pd.DataFrame) -> list:
    """Compresses the input DataFrame using Principal Component Analysis (PCA) and returns a list of the compressed data.

    Args:
        data (pd.DataFrame): A DataFrame containing the data to be compressed. The DataFrame must have an "ID" column which will be removed before compression.

    Returns:
        list: A list containing the compressed data. If the DataFrame has only one row, it returns the row as a list. Otherwise, it returns the PCA-transformed data as a list.
    """
    id_value = pd.unique(data.pop("ID"))[0]
    if len(data) == 1:
        data_list = data.values.tolist()[0]
        return data_list
    else:
        vector_data = data.to_numpy().T
        pca = PCA(n_components=1, random_state=12)
        transformed_data = pca.fit_transform(vector_data)
        transformed_data = transformed_data.flatten()
        transformed_data = transformed_data.tolist()
        return transformed_data

def vector_extraction(folder_path_str: str, output_path: str, grid_id_path: str, mapper_path = "") -> None:
    """Extracts and compresses vector representations from entity embeddings stored in the specified folder and saves the result to the output path.
    The script differentiates between WorldKG, KnowWhereGraph as well as OSMH3KG.

    Args:
        folder_path_str (str): The path to the folder containing the model and data files.
        output_path (str): The path where the output parquet file with compressed vectors will be saved.
        grid_id_path (str): The path to the parquet file containing grid ID mappings.
        mapper_path (str, optional): The path to the parquet file containing the mapper data. Defaults to "".

    Returns:
        Writes data to parquet file
    """
    folder_path = Path(folder_path_str)
    if re.findall('worldkg', str(folder_path)):
        model_path = folder_path.glob("**/*.pkl")
        model_path = list(model_path)[0]
        folder_path = model_path.parent
        kge_model = torch.load(model_path)
        entity_path = str(folder_path) + '/training_triples/entity_to_id.tsv.gz'
        entity_to_id = pd.read_csv(entity_path, sep='\t', compression='gzip')
        grid_id_df = pd.read_parquet(grid_id_path)
        world_kg_mapper = pd.read_parquet(mapper_path)
        osm_h3_merge = pd.merge(world_kg_mapper, grid_id_df, how="inner", on="h3_id")
        osm_h3_merge = osm_h3_merge[['h3_id', 'osm_id']]
        osm_h3_merge = osm_h3_merge.dropna()
        osm_h3_merge['osm_id'] = 'http://www.worldkg.org/resource/' + osm_h3_merge['osm_id']
        osm_id = pd.unique(osm_h3_merge['osm_id'])
        entity_to_id = entity_to_id[entity_to_id['label'].isin(osm_id)]
        entity_id_list = entity_to_id['id'].to_list()
        entity_id = torch.tensor(entity_id_list)
        entity_embedding_tensor = kge_model.entity_representations[0](indices=entity_id).detach().cpu().numpy()
        entity_embedding_df = pd.DataFrame(entity_embedding_tensor)
        entity_embedding_df = entity_embedding_df.astype(float)
        entity_embedding_df = entity_embedding_df.add_prefix('vector', axis=1)
        entity_embedding_df['id'] = entity_id_list
        entity_embedding_df = entity_embedding_df.merge(entity_to_id, how="left", on="id")
        entity_embedding_df = entity_embedding_df.merge(osm_h3_merge, how="left", left_on="label", right_on="osm_id")
        entity_embedding_df = entity_embedding_df.drop(["osm_id", "id", "label"], axis=1)
        entity_embedding_df = entity_embedding_df.rename({"h3_id": "ID"}, axis=1)
        entity_compressed_embedding = entity_embedding_df.groupby("ID").apply(pca_compression)
        entity_compressed_embedding = entity_compressed_embedding.reset_index(name="vector")
        entity_compressed_embedding_list = entity_compressed_embedding['vector'].values.tolist()
        entity_compressed_embedding_df = pd.DataFrame(data=entity_compressed_embedding_list)
        entity_compressed_embedding_df = entity_compressed_embedding_df.add_prefix('vector')
        entity_compressed_embedding_df['ID'] = entity_compressed_embedding['ID']
        entity_compressed_embedding_df.to_parquet(output_path)
    elif re.findall('KnowWhereGraph', str(folder_path)):
        folder_list = sorted(list(folder_path.glob('*')))
        iteration = 0
        list_df = []
        mapper_data = pd.read_parquet(mapper_path)
        mapper_data['s2CellID'] = 'http://stko-kwg.geog.ucsb.edu/lod/resource/s2.level13.' + mapper_data['s2CellID'].astype(str)
        for folder in tqdm(folder_list):
            model_path = str(folder) + '/trained_model.pkl'
            kge_model = torch.load(model_path)
            entity_path = str(folder) + '/training_triples/entity_to_id.tsv.gz'
            entity_to_id = pd.read_csv(entity_path, sep='\t', compression='gzip')
            grid_id = pd.unique(mapper_data['s2CellID'])
            entity_to_id = entity_to_id[entity_to_id['label'].isin(grid_id)]
            entity_id_list = entity_to_id['id'].to_list()
            entity_id = torch.tensor(entity_id_list)
            entity_embedding_tensor = kge_model.entity_representations[0](indices=entity_id).detach().cpu().numpy()
            if iteration == 0:
                previous_array = entity_embedding_tensor
                pass
            else:
                previous_array, entity_embedding_tensor, _ = procrustes(previous_array, entity_embedding_tensor)
            entity_df = pd.DataFrame(data = entity_embedding_tensor)
            entity_df = entity_df.astype(float)
            entity_df = entity_df.add_prefix('vector')
            entity_df['id'] = entity_id_list
            entity_df['DATE'] = folder.stem
            entity_df = entity_df.merge(entity_to_id, how='left', on='id')
            entity_df = entity_df.merge(mapper_data, how="inner", left_on="label", right_on="s2CellID")
            entity_df = entity_df.drop(["label", "s2CellID"], axis=1)
            entity_df = entity_df.rename({"h3ID": "label"}, axis=1)
            list_df.append(entity_df)
            iteration += 1
        
        df_entities = pd.concat(list_df, axis=0)
        df_entities = df_entities.drop('id', axis=1)
        df_entities = df_entities.rename({'label': 'ID'}, axis=1)
        df_entities.to_parquet(output_path)
    else:
        folder_list = sorted(list(folder_path.glob('*')))
        iteration = 0
        list_df = []
        for folder in tqdm(folder_list):
            model_path = str(folder) + '/trained_model.pkl'
            kge_model = torch.load(model_path)
            entity_path = str(folder) + '/training_triples/entity_to_id.tsv.gz'
            entity_to_id = pd.read_csv(entity_path, sep='\t', compression='gzip')
            grid_id_df = gpd.read_parquet(grid_id_path)
            grid_id = pd.unique(grid_id_df['h3_id'])
            entity_to_id = entity_to_id[entity_to_id['label'].isin(grid_id)]
            entity_id_list = entity_to_id['id'].to_list()
            entity_id = torch.tensor(entity_id_list)
            entity_embedding_tensor = kge_model.entity_representations[0](indices=entity_id).detach().cpu().numpy()
            if iteration == 0:
                previous_array = entity_embedding_tensor
                pass
            else:
                previous_array, entity_embedding_tensor, _ = procrustes(previous_array, entity_embedding_tensor)

            entity_df = pd.DataFrame(data = entity_embedding_tensor)
            entity_df = entity_df.astype(float)
            entity_df = entity_df.add_prefix('vector')
            entity_df['id'] = entity_id_list
            entity_df['DATE'] = folder.stem
            entity_df = entity_df.merge(entity_to_id, how='left', on='id')
            list_df.append(entity_df)
            iteration += 1
            
        df_entities = pd.concat(list_df, axis=0)
        df_entities = df_entities.drop('id', axis=1)
        df_entities = df_entities.rename({'label': 'ID'}, axis=1)
        df_entities.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vector Transformation argument parser')
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--grid_id_path", type=str, required=True)
    parser.add_argument("--mapper_path", type=str, required=False)
    args = parser.parse_args()
    vector_extraction(folder_path_str=args.folder_path, output_path=args.output_path, grid_id_path= args.grid_id_path, mapper_path=args.mapper_path)