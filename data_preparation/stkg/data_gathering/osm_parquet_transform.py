import subprocess
from pathlib import Path
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.api import Api
from multiprocessing import Pool
from shapely.geometry import shape
import argparse

# def bash_execution(destination_path, pbf_file, area_clipping):
#     folder_path = f'{destination_path}/{pbf_file.stem.split(".")[0]}'
#     print(f"Transformation start for file: {pbf_file.stem.split('.')[0]}")
#     if area_clipping:
#         raise NotImplementedError
#     else:
#         subprocess.run(["bash", "data_gathering/osm_parquet_transform.sh", ogr_temporary,
#                     "CONFIG_FILE=helper/osmconf.ini", folder_path, pbf_file])

def transformation(folder_path: str, destination_path: str, area_information: str,
                   area_clipping: bool):
    """_summary_

    Args:
        folder_path (str): _description_
        destination_path (str): _description_
        area_information (str): _description_
        area_clipping (bool): _description_
    """
    folder_path = Path(folder_path)
    nominatim = Nominatim()
    area_parameter = nominatim.query(area_information).toJSON()[0]
    pbf_files = folder_path.glob("**/*osm.pbf")
    # pool_file = Pool(5)
    # pool_file.map_async(bash_execution, pbf_files)
    # pool_file.close()
    for pbf_file in pbf_files:
        folder_path = f'{destination_path}/{pbf_file.stem.split(".")[0]}'
        print(f"Transformation start for file: {pbf_file.stem.split('.')[0]}")
        if area_clipping:
            area_parameter_type = area_parameter.get("osm_type")
            area_parameter_id = area_parameter.get("osm_id")
            api = Api()
            api_data = api.query(f"{area_parameter_type}/{area_parameter_id}")
            osm_geometry = shape(api_data.geometry())
            osm_geometry = osm_geometry.wkt
            folder_path = folder_path + '_' +  ''.join(character for character in area_information if character.isalnum())
            print(folder_path)
            subprocess.run(["bash", "data_preparation/stkg/data_gathering/osm_parquet_transform.sh", "data_preparation/stkg/helper/osmconf.ini",
                        "CONFIG_FILE=data_preparation/stkg/helper/osmconf.ini", folder_path, pbf_file, osm_geometry])
        else:
            subprocess.run(["bash", "data_preparation/stkg/data_gathering/osm_parquet_transform.sh", "data_preparation/stkg/helper/osmconf.ini",
                        "CONFIG_FILE=data_preparation/stkg/helper/osmconf.ini", folder_path, pbf_file])

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='OSM parquet transformer argument parser')
    parser.add_argument("--osm_data_path", type=str, required=True)
    parser.add_argument("--osm_parquet_path", type=str, required=True)
    parser.add_argument("--osm_area", type=str, required=True)
    parser.add_argument("--osm_clipping", default=False, action="store_true")
    args = parser.parse_args()
    transformation(folder_path=args.osm_data_path, destination_path=args.osm_parquet_path, area_information=args.osm_area,
                   area_clipping=args.osm_clipping)