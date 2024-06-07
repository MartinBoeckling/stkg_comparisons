from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import argparse


class KWGConstructor:
    """A class to construct the KnowWhereGraph knowledge graph from given input data.

    This class reads data from an input file, performs SPARQL queries to retrieve
    triples, and writes the triples to an output file.
    """
    def __init__(self, inputPath: str, outputPath: str) -> None:
        """Initializes the KWGConstructor with input and output paths.

        Args:
            inputPath (str): Path to the input file containing grid data.
            outputPath (str): Path to the output file where the resulting triples will be saved.
        """
        self.tripleList = []
        gridData = pd.read_parquet(inputPath)
        s2IDValues = gridData['s2CellID'].values
        tripleListNested = self.SPARQLRequest(s2IDValues)
        tripleList = [triple for tripleDoubleList in tripleListNested for triple in tripleDoubleList]
        tripelDf = pd.DataFrame(tripleList)
        tripelDf.to_parquet(outputPath, index=False)

        
    def SPARQLTripleGenerator(self, queryBody: dict) -> dict:
        """Generates a triple dictionary from a SPARQL query result.

        Args:
            queryBody (dict): A dictionary containing the SPARQL query result.

        Returns:
            dict: A dictionary representing the triple with 'subject', 'predicate', 'object', and 'ID' keys.
        """
        subject = queryBody.get('s').get('value')
        predicate = queryBody.get('p').get('value')
        object_string = queryBody.get('o').get('value')
        return {'subject': subject, 'predicate': predicate, 'object': object_string, 'ID': f'http://stko-kwg.geog.ucsb.edu/lod/resource/s2.level13.{self.ID}'}

    def SPARQLQuerierGrid(self, subject: str) -> list:
        """Performs a SPARQL query for a given subject and returns a list of triples.

        Args:
            subject (str): The subject for which to perform the SPARQL query.

        Returns:
            list: A list of dictionaries, each representing a triple.
        """        
        sparql = SPARQLWrapper("https://stko-kwg.geog.ucsb.edu/sparql")
        sparql.setReturnFormat(JSON)
        self.ID = subject
        query = f"""PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
        select ?s ?p ?o where {{ 
            ?s kwg-ont:hasID "{subject}".
            ?s ?p ?o .
        }}
        """
        sparql.setQuery(query)
        while True:
            try:
                query_result = sparql.queryAndConvert()
                queryBindings = query_result["results"]["bindings"]
                tripleList = list(map(self.SPARQLTripleGenerator, queryBindings))

            except Exception as e:
                print({'subject': subject, 'exception': e})
                continue
            break
        return tripleList

    def SPARQLRequest(self, s2IDValues: list) -> list:
        """Performs SPARQL queries for a list of s2CellID values and returns the resulting triples.

        Args:
            s2IDValues (list): A list of s2CellID values to query.

        Returns:
            list: A list of lists of triples, where each inner list contains triples for a specific s2CellID.
        """
        sparql = SPARQLWrapper("https://stko-kwg.geog.ucsb.edu/sparql")
        sparql.setReturnFormat(JSON)
        with Pool(processes=8) as pool:
            tripleList = list(pool.map(self.SPARQLQuerierGrid, s2IDValues))
        return tripleList

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='KnowWhereGraph SPARQL parameters')
    parser.add_argument('--mapper_path', type=str, required=True, help='Mapper path to the h3-s2 mapping parquet file')
    parser.add_argument('--target_path', type=str, required=True, help='Path to save the output triples from SPARQL the query')
    args = parser.parse_args()
    kwgClass = KWGConstructor(inputPath=args.mapper_path, outputPath=args.target_path)

