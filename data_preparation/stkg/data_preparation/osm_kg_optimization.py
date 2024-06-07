from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
import argparse

def optimize_parquet(spark_master: str, spark_driver_memory:str, spark_executor_memory: str,
					 spark_temp_directory: str, kg_output_path: str) -> None:

	builder = SparkSession.\
			builder.\
			master(spark_master).\
			appName('delta_optimization').\
			config('spark.driver.memory', spark_driver_memory).\
			config('spark.executor.memory', spark_executor_memory).\
			config("spark.local.dir", spark_temp_directory).\
			config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension").\
			config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog").\
			config("spark.databricks.delta.retentionDurationCheck.enabled", "false")

	spark = configure_spark_with_delta_pip(builder).getOrCreate()
	deltaTable = DeltaTable.forPath(spark, kg_output_path)

	deltaTable.optimize().executeCompaction()
	deltaTable.vacuum(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DeltaTable optimizer argument parser')
	parser.add_argument("--spark_master", type=str, required=True)
	parser.add_argument("--spark_driver_memory", type=str, required=True)
	parser.add_argument("--spark_executor_memory", type=str, required=True)
	parser.add_argument("--spark_temp_directory", type=str, required=True)
	parser.add_argument("--kg_output_path", type=str, required=True)
	args = parser.parse_args()
	optimize_parquet(spark_master=args.spark_master, spark_driver_memory=args.spark_driver_memory,
				  	spark_executor_memory=args.spark_executor_memory, spark_temp_directory=args.spark_temp_directory,
					kg_output_path=args.kg_output_path)
	