import pandas as pd
from sklearn.decomposition import PCA
import argparse

def pca_compression(data: pd.DataFrame) -> list:
    """Compresses the input DataFrame using Principal Component Analysis (PCA) and returns a list of the compressed data.

    Args:
        data (pd.DataFrame): A DataFrame containing the data to be compressed. The DataFrame must have an "ID" column which will be removed before compression.

    Returns:
        list: A list containing the compressed data. If the DataFrame has only one row, it returns the row as a list. Otherwise, it returns the PCA-transformed data as a list.
    """
    id_value = pd.unique(data.pop("h3_id"))[0]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vector preparation scipt for RDF2Vec worldkg')
    parser.add_argument('--mapper_path', type=str, required=True, help='Mapper path to the h3-s2 mapping parquet file')
    parser.add_argument('--grid_path', type=str, required=True, help='grid path to the h3 grid parquet file')
    parser.add_argument('--vector_path', type=str, required=True, help='data path to the RDF2Vec vector dataset')
    parser.add_argument('--target_path', type=str, required=True, help='Path to save the output triples from SPARQL the query')
    args = parser.parse_args()
    world_kg_mapper = pd.read_parquet(args.mapper_path)
    grid_id_df = pd.read_parquet(args.grid_path)
    osm_h3_merge = pd.merge(world_kg_mapper, grid_id_df, how="inner", on="h3_id")
    osm_h3_merge = osm_h3_merge[['h3_id', 'osm_id']]
    osm_h3_merge = osm_h3_merge.dropna()
    osm_h3_merge['osm_id'] = 'http://www.worldkg.org/resource/' + osm_h3_merge['osm_id']
    osm_h3_merge = osm_h3_merge.rename({"osm_id": "ID"}, axis=1)
    vector_dataset = pd.read_parquet(args.vector_datapath)
    vector_dataset = vector_dataset.merge(osm_h3_merge, how="inner", on="ID")
    vector_dataset = vector_dataset.drop(["ID", "DATE"], axis=1)
    vector_dataset = vector_dataset.groupby("h3_id").apply(pca_compression)
    vector_dataset = vector_dataset.reset_index(name="vector")

    vector_dataset_list = vector_dataset['vector'].values.tolist()
    vector_dataset_df = pd.DataFrame(data=vector_dataset_list)
    vector_dataset_df = vector_dataset_df.add_prefix('vector')
    vector_dataset_df['h3_id'] = vector_dataset['h3_id']

    vector_dataset_df = vector_dataset_df.rename({"h3_id": "ID"}, axis=1)
    vector_dataset_df.to_parquet(args.target_path, index=False)