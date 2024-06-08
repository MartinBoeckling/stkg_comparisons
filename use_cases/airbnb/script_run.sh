# prepare AirBnB data
python use_cases/airbnb/airbnb_data_prep.py

# retrieve data from OpenStreetMap associated to Boston
python data_preparation/stkg/data_gathering/osm_api_retrieval.py \
    --osm_start_date "2015-06-01" \
    --osm_end_date "2018-09-01" \
    --osm_area "Boston Massachusetts, USA" \
    --output_path "data/airbnb/base_data/openstreetmap/parquet"

# generate h3 grid for Boston
python data_preparation/stkg/data_preparation/h3_generation.py\
    --area_grid_file data_preparation/stkg/helper/worlddata/world_data.parquet\
    --grid_clipping \
    --clipping_name "Boston Massachusetts, USA"\
    --cpu_cores 25\
    --spark_master "local[25]"\
    --spark_driver_memory "150G"\
    --spark_executor_memory "150G"\
    --geo_hash_level 6 \
    --grid_parquet_path data/airbnb/base_data/grid \
    --grid_level 9 \
    --single_parquet
# create sedona parquet files
python data_preparation/stkg/data_preparation/geoparquet_sedona.py \
    --spark_master "local[25]"\
    --spark_driver_memory "150G"\
    --spark_executor_memory "150G"\
    --geo_hash_level 6 \
    --spark_temp_directory "spark_temp_directory"\
    --osm_parquet_path "data/airbnb/base_data/openstreetmap/parquet"\
    --geometry_file_path "data/airbnb/base_data/openstreetmap/sedona_parquet"
# prepare OSMh3KG Knowledge Graph
python data_preparation/stkg/data_preparation/osm_kg_data_preparation.py \
    --spark_master "local[25]" \
    --spark_driver_memory "150G" \
    --spark_executor_memory "150G" \
    --spark_temp_directory "spark_temp_directory" \
    --geometry_file_path "data/airbnb/base_data/openstreetmap/sedona_parquet" \
    --grid_parquet_path "data/airbnb/base_data/grid" \
    --grid_level 9 \
    --kg_output_path "data/airbnb/knowledge_graph/ownstkg/knowledgegraph"
# create optimization of written files
python data_preparation/stkg/data_preparation/osm_kg_optimization.py \
    --spark_master "local[25]" \
    --spark_driver_memory "150G" \
    --spark_executor_memory "150G" \
    --spark_temp_directory "spark_temp_directory" \
    --kg_output_path "data/airbnb/knowledge_graph/ownstkg/knowledgegraph"


# create RDF2Vec vector for OSMh3KG
python data_preparation/rdf2vec.py \
    --path "data/airbnb/knowledge_graph/ownstkg/knowledgegraph"\
    --distance 4 \
    --walknumber "500" \
    --train \
    --chunksize 100 \
    --savepath "data/airbnb/knowledge_graph/ownstkg/vector" \
    --alignmentprojection \
    --grid_path "data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet"

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/ownstkg/vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_rdf2vec_ownstkg.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_rdf2vec_ownstkg.parquet \
    -d 2017-04-01

# create transe embedding model
python data_preparation/complex.py \
    --kg_path "data/airbnb/knowledge_graph/ownstkg/knowledgegraph" \
    --log_dir "data/airbnb/knowledge_graph/ownstkg/transe_vector" \
    --model "transe" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

# extract transe embedding vector
python data_preparation/complex_vector.py \
    --folder_path "data/airbnb/knowledge_graph/ownstkg/transe_vector" \
    --output_path "data/airbnb/knowledge_graph/ownstkg/transe_vector/vectorDf.parquet" \
    --grid_id_path "data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet"

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/ownstkg/transe_vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_transe_ownstkg.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_transe_ownstkg.parquet \
    -d 2017-04-01


python data_preparation/complex.py \
    --kg_path "data/airbnb/knowledge_graph/ownstkg/knowledgegraph" \
    --log_dir "data/airbnb/knowledge_graph/worldkg/complex_vector" \
    --model "complex" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path "data/airbnb/knowledge_graph/ownstkg/complex_vector" \
    --output_path "data/airbnb/knowledge_graph/ownstkg/complex_vector/vectorDf.parquet" \
    --grid_id_path "data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet"

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/ownstkg/transe_vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_complex_ownstkg.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_complex_ownstkg.parquet \
    -d 2017-04-01

# WorldKG preparation
osmium extract --bbox "-71.1912442,42.2279112,-70.8044881,42.3969775" \
    data/airbnb/base_data/openstreetmap/massachusetts-160101.osm.pbf \
    -o data/airbnb/base_data/openstreetmap/boston.pbf

python data_preparation/worldkg/CreateTriples.py \
    --grid_res 9 \
    --pbf_file data/airbnb/base_data/openstreetmap/boston.pbf \
    --kg_path data/airbnb/knowledge_graph/worldkg/knowledgegraph \
    --grid_map_path data/airbnb/base_data/worldkg_mapper.parquet \
    --date 2016-01-01

python data_preparation/rdf2vec.py \
    --path "data/airbnb/knowledge_graph/worldkg/knowledgegraph"\
    --distance 4 \
    --walknumber "500" \
    --train \
    --chunksize 100 \
    --savepath "data/airbnb/knowledge_graph/worldkg/vector" \
    --alignmentprojection \
    --grid_path "data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet"

python data_preparation/worldkg/rdf2vec_vector_prep.py \
    --mapper_path "data/airbnb/base_data/worldkg_mapper.parquet"\
    --grid_path "data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet"\
    --vector_path "data/airbnb/knowledge_graph/worldkg/vector/vectorDf.parquet"\
    --target_path "data/airbnb/knowledge_graph/worldkg/vector/vectorDf.parquet"

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/worldkg/vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_rdf2vec_worldkg.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_rdf2vec_worldkg.parquet \
    -d 2017-04-01

python data_preparation/complex.py \
    --kg_path "data/airbnb/knowledge_graph/worldkg/knowledgegraph" \
    --log_dir "data/airbnb/knowledge_graph/worldkg/transe_vector" \
    --model "transe" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/airbnb/knowledge_graph/worldkg/transe_vector \
    --output_path data/airbnb/knowledge_graph/worldkg/transe_vector/vectorDf.parquet \
    --grid_id_path data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet \
    --mapper_path data/airbnb/base_data/worldkg_mapper.parquet

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/worldkg/transe_vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_transe_worldkg.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_transe_worldkg.parquet \
    -d 2017-04-01

python data_preparation/complex.py \
    --kg_path "data/airbnb/knowledge_graph/worldkg/knowledgegraph" \
    --log_dir "data/airbnb/knowledge_graph/worldkg/complex_vector" \
    --model "complex" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/airbnb/knowledge_graph/worldkg/complex_vector \
    --output_path data/airbnb/knowledge_graph/worldkg/complex_vector/vectorDf.parquet \
    --grid_id_path data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet \
    --mapper_path data/airbnb/base_data/worldkg_mapper.parquet

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/worldkg/complex_vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_complex_worldkg.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_complex_worldkg.parquet \
    -d 2017-04-01

# KnowWhereGraph preparation
python data_preparation/KWG/h3_to_s2.py \
    --path data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet \
    --target_path data/airbnb/base_data/h3_s2_data.parquet

python data_preparation/KWG/SPARQLKWG.py \
    --mapper_path data/airbnb/base_data/h3_s2_data.parquet \
    --target_path data/airbnb/knowledge_graph/KnowWhereGraph/triple_data.parquet

python data_preparation/KWG/kwgPreparation.py \
    --path data/airbnb/knowledge_graph/KnowWhereGraph/triple_data.parquet \
    --target_path data/airbnb/knowledge_graph/KnowWhereGraph/knowledgegraph \
    --start_date 2016-09-01 \
    --end_date 2017-09-01 \
    --time_frequency MS

python data_preparation/rdf2vec.py \
    --path "data/airbnb/knowledge_graph/KnowWhereGraph/knowledgegraph"\
    --distance 4 \
    --walknumber "500" \
    --train \
    --chunksize 100 \
    --savepath "data/airbnb/knowledge_graph/KnowWhereGraph/vector" \
    --alignmentprojection \
    --grid_path "data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet"

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/KnowWhereGraph/vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_rdf2vec_KnowWhereGraph.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_rdf2vec_KnowWhereGraph.parquet \
    -d 2017-04-01

python data_preparation/complex.py \
    --kg_path "data/airbnb/knowledge_graph/KnowWhereGraph/knowledgegraph" \
    --log_dir "data/airbnb/knowledge_graph/KnowWhereGraph/transe_vector" \
    --model "transe" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/airbnb/knowledge_graph/KnowWhereGraph/transe_vector \
    --output_path data/airbnb/knowledge_graph/KnowWhereGraph/transe_vector/vectorDf.parquet \
    --grid_id_path data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet \
    --mapper_path data/airbnb/base_data/h3_s2_data.parquet

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/KnowWhereGraph/transe_vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_transe_KnowWhereGraph.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_transe_KnowWhereGraph.parquet \
    -d 2017-04-01

python data_preparation/complex.py \
    --kg_path "data/airbnb/knowledge_graph/KnowWhereGraph/knowledgegraph" \
    --log_dir "data/airbnb/knowledge_graph/KnowWhereGraph/complex_vector" \
    --model "complex" \
    --device "cuda:0" \
    --checkpoint "/ceph/mboeckli/stkg_comparison/checkpoints"

python data_preparation/complex_vector.py \
    --folder_path data/airbnb/knowledge_graph/KnowWhereGraph/complex_vector \
    --output_path data/airbnb/knowledge_graph/KnowWhereGraph/complex_vector/vectorDf.parquet \
    --grid_id_path data/airbnb/base_data/grid/part-00000-e1a0cb12-251b-4af2-975e-acc9fc46ad82-c000.snappy.parquet \
    --mapper_path data/airbnb/base_data/h3_s2_data.parquet

python use_cases/airbnb/vector_join.py \
    --base_path "data/airbnb/ml_data/airbnb_base.parquet" \
    --vector_path "data/airbnb/knowledge_graph/KnowWhereGraph/complex_vector/vectorDf.parquet" \
    --output_path "data/airbnb/ml_data/airbnb_hybrid_complex_KnowWhereGraph.parquet"

python modeling/xgboost_regression.py \
    -p data/airbnb/ml_data/airbnb_hybrid_complex_KnowWhereGraph.parquet \
    -d 2017-04-01