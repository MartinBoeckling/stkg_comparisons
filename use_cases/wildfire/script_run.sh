python data_preparation/stkg/data_preparation/h3_generation.py\
    --area_grid_file data_preparation/stkg/helper/worlddata/world_data.parquet\
    --grid_clipping \
    --clipping_name "Boston Massachusetts, USA"\
    --cpu_cores 25\
    --spark_master "local[25]"\
    --spark_driver_memory "150G"\
    --spark_executor_memory "150G"\
    --geo_hash_level 8 \
    --grid_parquet_path data/wildfire/grid \
    --grid_level 7 \
    --single_parquet

# Run R File use_cases/wildfire/datapreparation.R

# Run R File use_cases/wildfire/datasetcreation.R

# download openstreetmap file
wget https://download.geofabrik.de/north-america/us/california-140101.osm.pbf -O data/wildfire/base_data/california/california-140101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-140101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-150101.osm.pbf -O data/wildfire/base_data/california/california-150101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-150101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-160101.osm.pbf -O data/wildfire/base_data/california/california-160101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-160101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-170101.osm.pbf -O data/wildfire/base_data/california/california-170101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-170101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-180101.osm.pbf -O data/wildfire/base_data/california/california-180101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-180101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-190101.osm.pbf -O data/wildfire/base_data/california/california-190101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-190101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-200101.osm.pbf -O data/wildfire/base_data/california/california-200101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-200101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

wget https://download.geofabrik.de/north-america/us/california-210101.osm.pbf -O data/wildfire/base_data/california/california-210101.osm.pbf
python data_preparation/stkg/data_gathering/osm_parquet_transform.py \
    --osm_data_path data/wildfire/base_data/california/california-210101.osm.pbf \
    --osm_parquet_path data/wildfire/base_data/california/parquet \
    --osm_area California, USA

# OSMh3KG 
python data_preparation/stkg/data_preparation/geoparquet_sedona.py \
    --spark_master "local[25]"\
    --spark_driver_memory "150G"\
    --spark_executor_memory "150G"\
    --geo_hash_level 6 \
    --spark_temp_directory "spark_temp_directory"\
    --osm_parquet_path "data/wildfire/base_data/california/parquet"\
    --geometry_file_path "data/wildfire/base_data/california/sedona_parquet"

python data_preparation/stkg/data_preparation/osm_kg_data_preparation.py \
    --spark_master "local[25]" \
    --spark_driver_memory "150G" \
    --spark_executor_memory "150G" \
    --spark_temp_directory "spark_temp_directory" \
    --geometry_file_path "data/wildfire/base_data/california/sedona_parquet" \
    --grid_parquet_path "data/wildfire/grid/h3_grid.parquet" \
    --grid_level 7 \
    --kg_output_path "data/wildfire/knowledge_graph/ownstkg/knowledgegraph"

python data_preparation/stkg/data_preparation/osm_kg_optimization.py \
    --spark_master "local[25]" \
    --spark_driver_memory "150G" \
    --spark_executor_memory "150G" \
    --spark_temp_directory "spark_temp_directory" \
    --kg_output_path "data/wildfire/knowledge_graph/ownstkg/knowledgegraph"


python data_preparation/rdf2vec.py \
    --path "data/wildfire/knowledge_graph/ownstkg/knowledgegraph" \
    --distance 4 \
    --walknumber 500 \
    --train \
    --chunksize 250 \
    --savepath "data/wildfire/knowledge_graph/ownstkg/vector" \
    --alignmentprojection \
    --grid_path "data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet"

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/ownstkg/vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_rdf2vec_ownstkg.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_rdf2vec_ownstkg.parquet \
    -d 2020-01-01

python data_preparation/complex.py \
    --kg_path "data/wildfire/knowledge_graph/ownstkg/knowledgegraph" \
    --log_dir "data/wildfire/knowledge_graph/ownstkg/transe_vector" \
    --model "transe" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

# extract transe embedding vector
python data_preparation/complex_vector.py \
    --folder_path "data/wildfire/knowledge_graph/ownstkg/transe_vector" \
    --output_path "data/wildfire/knowledge_graph/ownstkg/transe_vector/vectorDf.parquet" \
    --grid_id_path "data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet"

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/ownstkg/transe_vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_transe_ownstkg.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_transe_ownstkg.parquet \
    -d 2020-01-01


python data_preparation/complex.py \
    --kg_path "data/wildfire/knowledge_graph/ownstkg/knowledgegraph" \
    --log_dir "data/wildfire/knowledge_graph/worldkg/complex_vector" \
    --model "complex" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path "data/wildfire/knowledge_graph/ownstkg/complex_vector" \
    --output_path "data/wildfire/knowledge_graph/ownstkg/complex_vector/vectorDf.parquet" \
    --grid_id_path "data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet"

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/ownstkg/transe_vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_complex_ownstkg.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_complex_ownstkg.parquet \
    -d 2020-01-01

# WorldKG
python data_preparation/worldkg/CreateTriples.py \
    --grid_res 7 \
    --pbf_file data/wildfire/base_data/california/california-140101.osm.pbf \
    --kg_path data/wildfire/knowledge_graph/worldkg/knowledgegraph \
    --grid_map_path data/wildfire/base_data/worldkg_mapper.parquet \
    --date 2014-01-01

python data_preparation/rdf2vec.py \
    --path "data/wildfire/knowledge_graph/worldkg/knowledgegraph"\
    --distance 4 \
    --walknumber "500" \
    --train \
    --chunksize 100 \
    --savepath "data/wildfire/knowledge_graph/worldkg/vector" \
    --alignmentprojection \
    --grid_path "data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet"

python data_preparation/worldkg/rdf2vec_vector_prep.py \
    --mapper_path "data/wildfire/base_data/worldkg_mapper.parquet"\
    --grid_path "data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet"\
    --vector_path "data/wildfire/knowledge_graph/worldkg/vector/vectorDf.parquet"\
    --target_path "data/wildfire/knowledge_graph/worldkg/vector/vectorDf.parquet"

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/worldkg/vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_rdf2vec_worldkg.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_rdf2vec_worldkg.parquet \
    -d 2020-01-01


python data_preparation/complex.py \
    --kg_path "data/wildfire/knowledge_graph/worldkg/knowledgegraph" \
    --log_dir "data/wildfire/knowledge_graph/worldkg/transe_vector" \
    --model "transe" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/wildfire/knowledge_graph/worldkg/transe_vector \
    --output_path data/wildfire/knowledge_graph/worldkg/transe_vector/vectorDf.parquet \
    --grid_id_path data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet \
    --mapper_path data/wildfire/base_data/worldkg_mapper.parquet

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/worldkg/transe_vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_transe_worldkg.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_transe_worldkg.parquet \
    -d 2020-01-01


python data_preparation/complex.py \
    --kg_path "data/wildfire/knowledge_graph/worldkg/knowledgegraph" \
    --log_dir "data/wildfire/knowledge_graph/worldkg/complex_vector" \
    --model "complex" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/wildfire/knowledge_graph/worldkg/complex_vector \
    --output_path data/wildfire/knowledge_graph/worldkg/complex_vector/vectorDf.parquet \
    --grid_id_path data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet \
    --mapper_path data/wildfire/base_data/worldkg_mapper.parquet

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/worldkg/complex_vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_complex_worldkg.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_complex_worldkg.parquet \
    -d 2020-01-01


# KnowWhereGraph preparation
python data_preparation/KWG/h3_to_s2.py \
    --path data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet \
    --target_path data/wildfire/base_data/h3_s2_data.parquet

python data_preparation/KWG/SPARQLKWG.py \
    --mapper_path data/wildfire/base_data/h3_s2_data.parquet \
    --target_path data/wildfire/knowledge_graph/KnowWhereGraph/triple_data.parquet

python data_preparation/KWG/kwgPreparation.py \
    --path data/wildfire/knowledge_graph/KnowWhereGraph/triple_data.parquet \
    --target_path data/wildfire/knowledge_graph/KnowWhereGraph/knowledgegraph \
    --start_date 2016-09-01 \
    --end_date 2017-09-01 \
    --time_frequency MS

python data_preparation/rdf2vec.py \
    --path "data/wildfire/knowledge_graph/KnowWhereGraph/knowledgegraph"\
    --distance 4 \
    --walknumber "500" \
    --train \
    --chunksize 100 \
    --savepath "data/wildfire/knowledge_graph/KnowWhereGraph/vector" \
    --alignmentprojection \
    --grid_path "data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet"

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/KnowWhereGraph/vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_rdf2vec_KnowWhereGraph.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_rdf2vec_KnowWhereGraph.parquet \
    -d 2020-01-01

python data_preparation/complex.py \
    --kg_path "data/wildfire/knowledge_graph/KnowWhereGraph/knowledgegraph" \
    --log_dir "data/wildfire/knowledge_graph/KnowWhereGraph/transe_vector" \
    --model "transe" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/wildfire/knowledge_graph/KnowWhereGraph/transe_vector \
    --output_path data/wildfire/knowledge_graph/KnowWhereGraph/transe_vector/vectorDf.parquet \
    --grid_id_path data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet \
    --mapper_path data/wildfire/base_data/h3_s2_data.parquet

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/KnowWhereGraph/transe_vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_transe_KnowWhereGraph.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_transe_KnowWhereGraph.parquet \
    -d 2020-01-01

python data_preparation/complex.py \
    --kg_path "data/wildfire/knowledge_graph/KnowWhereGraph/knowledgegraph" \
    --log_dir "data/wildfire/knowledge_graph/KnowWhereGraph/complex_vector" \
    --model "complex" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/wildfire/knowledge_graph/KnowWhereGraph/complex_vector \
    --output_path data/wildfire/knowledge_graph/KnowWhereGraph/complex_vector/vectorDf.parquet \
    --grid_id_path data/wildfire/grid/h3_grid.parquet/part-00000-0a7d70b2-545e-419b-b34f-8d8f59d76ac9-c000.snappy.parquet \
    --mapper_path data/wildfire/base_data/h3_s2_data.parquet

python use_cases/wildfire/vector_join.py \
    --base_path "data/wildfire/ml_data/wildfire_base.parquet" \
    --vector_path "data/wildfire/knowledge_graph/KnowWhereGraph/complex_vector/vectorDf.parquet" \
    --output_path "data/wildfire/ml_data/wildfire_hybrid_complex_KnowWhereGraph.parquet"

python modeling/xgboost_regression.py \
    -p data/wildfire/ml_data/wildfire_hybrid_complex_KnowWhereGraph.parquet \
    -d 2020-01-01