"""
File containing the different constants relevant for this repository
"""
# data gathering variables
helper_path_directory = "/ceph/mboeckli/stkg/helper"
osm_data_path = "/ceph/mboeckli/stkg_comparison_data/wildfire_data/base_data/openstreetmap/california"
osm_parquet_path = "/ceph/mboeckli/stkg_comparison_data/wildfire_data/base_data/openstreetmap/california/parquet"
osm_start_date = "2010-01-01"
osm_end_date = "2022-12-31"
osm_area = "California, United States"
osm_clipping = False
ogr_temporary = "/ceph/mboeckli/stkg/ogrTemp"

# data preparation variables
cpu_cores = 27
spark_master = f"local[{cpu_cores}]"
spark_temp_directory = "/ceph/mboeckli/sparkTmp"
area_grid_file = "data_preparation/stkg/helper/worlddata/world_data.parquet"
spark_driver_memory = "150G"
spark_executor_memory = "400G"
kg_output_path = "/ceph/mboeckli/stkg_comparison_data/wildfire_data/knowledge_graph/ownstkg/knowledge_graph"
grid_clipping = True
grid_level = 7
grid_compaction = False
grid_parquet_path = "/ceph/mboeckli/stkg_comparison_data/wildfire_data/base_data/grid//h3_grid.parquet"
geo_hash_level = 4
sedona_packages = ['org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.5.1,'
                   'org.datasyslab:geotools-wrapper:1.5.1-28.2']
geometry_file_path = "/ceph/mboeckli/stkg_comparison_data/wildfire_data/base_data/openstreetmap/california/osm_geometry"