"""
Title:
    OSM API retrieval
Description:
    This script provides the wrapper method to download flexible areas over the world. As a backbone the Overpass API is used, which
    only allows the download of smaller regions on city/ small stae level. Overpass has a limit of 5 GB query result return
Input:
    - osm_start_date: Start date of API extraction
    - osm_end_date: End date of API extraction
    - date_frequency: Frequency to generate time sequence between osm_start_date and osm_end_date
    - city_name: Name of the respective city
    - geometry_tags: 
    - time_out:
Output:
    - Write Geoparquet file to defined location
"""
import argparse
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import logging

def swap_xy(geom: shape) -> str:
    if geom.is_empty:
        return geom

    if geom.has_z:
        def swap_xy_coords(coords):
            for x, y, z in coords:
                yield (y, x, z)
    else:
        def swap_xy_coords(coords):
            for x, y in coords:
                yield (y, x)
    # Process coordinates from each supported geometry type
    if geom.geom_type in ('Point', 'LineString', 'LinearRing'):
        return type(geom)(list(swap_xy_coords(geom.coords)))
    elif geom.geom_type == 'Polygon':
        ring = geom.exterior
        shell = type(ring)(list(swap_xy_coords(ring.coords)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = type(ring)(list(swap_xy_coords(ring.coords)))
        return type(geom)(shell, holes)
    elif geom.geom_type.startswith('Multi') or geom.geom_type == 'GeometryCollection':
        # Recursive call
        return type(geom)([swap_xy(part) for part in geom.geoms])
    else:
        raise ValueError('Type %r not recognized' % geom.geom_type)


def isvalid(geom: str) -> bool:
    try:
        shape(geom)
        return True
    except:
        return False


def crawlOpenStreetMapData(city_name: str, element_types: str | list, timeout_span: int, path_file: str, crawl_date: str) -> None:
    print(path_file)
    nominatim = Nominatim()
    areaId = nominatim.query(city_name).areaId()
    overpass = Overpass()
    query = overpassQueryBuilder(area=areaId, elementType=element_types, includeGeometry=True, out='body')
    result = overpass.query(query, timeout=timeout_span,
                            date=crawl_date)
    result_data = {}
    i = 0
    for element in result.elements():
        try:
            if element.tags() is not None:
                tags = element.tags()
                tags.pop("created_by", None)
                tags.pop("converted_by", None)
                tags.pop("source", None)
                tags.pop("time", None)
                tags.pop("ele", None)
                tags.pop("attribution", None)
                if tags:
                    osm_id = element.id()
                    geometry = element.geometry()
                    result_data[i] = ({'osm_id': osm_id, 'all_tags': str(tags), 'geometry' : geometry})
                    i += 1
                else:
                    continue
            else:
                continue
        except:
            pass
    
    resultDataFrame = pd.DataFrame.from_dict(result_data, orient="index")
    resultDataFrame['isValid'] = resultDataFrame['geometry'].apply(lambda x: isvalid(x))
    resultDataFrame = resultDataFrame[resultDataFrame['isValid']]
    resultDataFrame = resultDataFrame[resultDataFrame['isValid']]
    geoResultDataFrame = gpd.GeoDataFrame(resultDataFrame, geometry='geometry')
    geoResultDataFrame.geometry = geoResultDataFrame.geometry.map(swap_xy)
    geoResultDataFrame.to_parquet(path_file)

def handleRequestOSM(start_date: str, end_date: str, date_frequency: str, city_name: str, geometry_tags: list, time_out: int, output_path: str) -> None:
    assert pd.to_datetime(start_date) >= pd.to_datetime("2012-01-01"), "OpenStreetMap has only captured data starting from 2012"
    assert pd.to_datetime(end_date) <= pd.Timestamp.today(), "OpenStreetMap can not retrieve data from the future"
    dateRange = pd.date_range(start=start_date, end=end_date, freq=date_frequency).values
    for date in tqdm(dateRange):
        date = str(date)
        crawlOpenStreetMapData(city_name, geometry_tags, time_out, f'{output_path}/osm_data_{pd.to_datetime(date).date()}.parquet', date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OSM parquet transformer argument parser')
    parser.add_argument("--osm_start_date", type=str, required=True)
    parser.add_argument("--osm_end_date", type=str, required=True)
    parser.add_argument("--osm_area", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    logging.getLogger('OSMPythonTools').setLevel(logging.ERROR)
    handleRequestOSM(start_date=args.osm_start_date, end_date=args.osm_end_date, date_frequency='MS',
                    city_name=args.osm_area, geometry_tags=['node', 'way', 'relation'], time_out=200,
                    output_path=args.output_path)