"""
Title: Generate h3 grid with flexible configuration options
Description: 

- Input:
    - grid_level: Level of h3 grid cell as integer in range 0 to 15 (Default: 9) Background on grid level can be found here: https://h3geo.org/docs/core-library/restable
    - grid_compaction: Boolean value whether the selection of grid cells should be compacted or not. Compaction background can be found here: https://h3geo.org/docs/highlights/indexing/
"""
# import packages
import h3
import geopandas as gpd
import json
import argparse
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from pyspark.sql.functions import col
from shapely.geometry import Polygon
from shapely.geometry import box
from sedona.spark import SedonaContext
from sedona.sql.st_functions import ST_GeoHash, ST_MakeValid
# from data_preparation.stkg.helper.constants import *
from OSMPythonTools.nominatim import Nominatim
from shapely.geometry import shape
from OSMPythonTools.api import Api


class h3_grid_creation:
    """_summary_
    """
    def __init__(self, area_grid_file: str, grid_clipping: bool, clipping_name: str, cpu_cores: int, spark_master: str,
                 spark_driver_memory: str, spark_executor_memory: str, geo_hash_level: int, grid_parquet_path: str,
                 grid_level: int, grid_compaction: bool, single_parquet: bool) -> None:
        """_summary_

        Args:
            area_grid_file (str): _description_
            grid_clipping (bool): _description_
            clipping_name (str): _description_
            cpu_cores (int): _description_
            spark_master (str): _description_
            spark_driver_memory (str): _description_
            spark_executor_memory (str): _description_
            geo_hash_level (int): _description_
            grid_parquet_path (str): _description_
            grid_level (int): _description_
            grid_compaction (bool): _description_
            single_parquet (bool): _description_
        """

        self.grid_level = grid_level
        self.grid_compaction = grid_compaction
        self.single_parquet = single_parquet
        
        self.create_h3_grid(area_grid_file, grid_clipping, clipping_name, cpu_cores, spark_master, spark_driver_memory, 
                            spark_executor_memory, geo_hash_level, grid_parquet_path)


    def extract_h3_grid(self, world_data_feature: dict) -> dict:
        """
        For the creation of our Knowledge Graph, we align the OpenStreetMap geometries using the h3 DGG
        created by Uber. Each individual grid cell is respresented as a regular hexagon (with the exception of 12
        grid cells per level which have a pentagon structure). To generate the grid cells we use a
        base of the world map

        Args:
            world_data_feature (dict): Dictionary that contains country properties as a nested
            dictionary together with the respective geometry of a country

        Returns:
            dict: Return of dictionary that contains feature specific keys:
                - country (str): 
                - continent (str):
                - subregion (str):
                - h3_id (list):
                - land_geometry:
        """
        properties = world_data_feature['properties']
        geometry = world_data_feature['geometry']
        h3_id = h3.polyfill_geojson(geometry, res=self.grid_level)
        if self.grid_compaction:
            h3_id = h3.compact(h3_id)
        row_dict = {'country': properties['admin'], 'continent': properties['continent'], 'subregion': properties['subregion'], 'h3_id': list(h3_id), 'land_geometry': str(geometry)}
        return row_dict


    def create_h3_grid(self, area_grid_file: str, grid_clipping: bool, clipping_name: str, cpu_cores: int, spark_master: str,
                       spark_driver_memory: str, spark_executor_memory: str, geo_hash_level: int, grid_parquet_path: str) -> None:
        """_summary_

        Args:
            area_grid_file (str): _description_
            grid_clipping (bool): _description_
            clipping_name (str): _description_
            cpu_cores (int): _description_
            spark_master (str): _description_
            spark_driver_memory (str): _description_
            spark_executor_memory (str): _description_
            geo_hash_level (int): _description_
            grid_parquet_path (str): _description_
        """
        world_data = gpd.read_parquet(area_grid_file)
        if grid_clipping:
            nominatim = Nominatim()
            area_parameter = nominatim.query(clipping_name).toJSON()[0]
            area_parameter_type = area_parameter.get("osm_type")
            area_parameter_id = area_parameter.get("osm_id")
            api = Api()
            api_data = api.query(f"{area_parameter_type}/{area_parameter_id}")
            geometry = shape(api_data.geometry())
            world_data = gpd.clip(world_data, mask=geometry)
        world_data = world_data.explode()
        world_data = json.loads(world_data.to_json())
        world_data = world_data['features']
        with Pool(cpu_cores) as pool:
            row_list = list(tqdm(pool.imap_unordered(self.extract_h3_grid, world_data), total=len(world_data)))
        data = pd.DataFrame.from_records(row_list)
        data = data.explode(column='h3_id')
        data = data.drop('land_geometry', axis=1)
        data = data.dropna()
        data['geometry'] = data['h3_id'].apply(lambda h3Id: h3.h3_to_geo_boundary(h=h3Id, geo_json=True))
        data['geometry'] = data['geometry'].apply(Polygon)
        gdf_data = gpd.GeoDataFrame(data, geometry="geometry")
        gdf_data = gdf_data.reset_index(drop=True)
        config = SedonaContext.builder().\
            master(spark_master).\
            appName('h3generation').\
            config('spark.driver.memory', spark_driver_memory).\
            config('spark.executor.memory', spark_executor_memory).\
            config('spark.jars.packages',
                'org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.5.1,'
                'org.datasyslab:geotools-wrapper:1.5.1-28.2').\
            getOrCreate()
        sedona = SedonaContext.create(config)
        sedona_df = sedona.createDataFrame(gdf_data)
        sedona_df = sedona_df.withColumn('geohash', ST_GeoHash('geometry', geo_hash_level)).orderBy('geohash')
        sedona_df =  sedona_df.withColumn("geometry", ST_MakeValid(col("geometry")))
        if self.single_parquet:
            sedona_df.coalesce(1).write.mode('overwrite').format('geoparquet').save(grid_parquet_path)
        else:
            sedona_df.write.mode('overwrite').format('geoparquet').save(grid_parquet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid creation argument parser')
    parser.add_argument("--area_grid_file", type=str, required=True)
    parser.add_argument("--grid_clipping", default=False, action="store_true")
    parser.add_argument("--clipping_name", type=str, required=True)
    parser.add_argument("--cpu_cores", type=int, required=True)
    parser.add_argument("--spark_master", type=str, required=True)
    parser.add_argument("--spark_driver_memory", type=str, required=True)
    parser.add_argument("--spark_executor_memory", type=str, required=True)
    parser.add_argument("--geo_hash_level", type=int, required=True)
    parser.add_argument("--grid_parquet_path", type=str, required=True)
    parser.add_argument("--grid_level", type=int, required=True)
    parser.add_argument("--grid_compaction", default=False, action="store_true")
    parser.add_argument("--single_parquet", default=False, action="store_true")

    args = parser.parse_args()
    h3_grid_creation(area_grid_file=args.area_grid_file, grid_clipping=args.grid_clipping, clipping_name=args.clipping_name, cpu_cores = args.cpu_cores, spark_master = args.spark_master,
                   spark_driver_memory = args.spark_driver_memory, spark_executor_memory = args.spark_executor_memory, geo_hash_level = args.geo_hash_level, grid_parquet_path = args.grid_parquet_path,
                   grid_level = args.grid_level, grid_compaction = args.grid_compaction, single_parquet=args.single_parquet)