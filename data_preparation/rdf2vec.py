'''
Title:
RDF2Vec transformation script
Description:
This script transforms an edge dataframe containing the following columns into a vector
representation using the RDF2Vec algorithm:
    - from: ID of grid cell or ID of geometric object as starting node
    - to: ID of grid cell or ID of geometric object as target node
    - description: Description of edge relation between starting and target node
    - DATE: Date of geometric object (YEAR or monthly data)
    - ID: Associated ID of grid cell.
The script contains three main methods. The first method is the dataPreparation method
which splits the edge dataframe into a training datafram grouping the dataframe to the
column year and a transformation dataframe grouping to the columns year and id.
The second script is the kgTraining method in which the training dataframe is grouping
to the year 
Input:
    - dataPath: Path to folder with edge dataframes in format dir/.../dir
    - distance: Distance of node to other node
    - maxWalks: maximum number of walks per defined entity
    - train: Boolean value if RDF2Vec should be performed
    - clustering: Boolean value if vector representation should be clustered with KMeans
    - chunk: 
    - save:
    - 
Output:
    - Transformer model embedding in format of pickle file
    - Vector representation of grid cell IDs in format of pickle file
'''
# import packages
import argparse
import pandas as pd
import numpy as np
import pickle
import random
from gensim_word2vec_procrustes_align import smart_procrustes_align_gensim
from tqdm import tqdm
from pathlib import Path
from igraph import Graph
from deltalake import DeltaTable
from itertools import groupby
import multiprocessing as mp
import re
from gensim.models.word2vec import Word2Vec as W2V
import geopandas as gpd


class kgEmbedding:

    def __init__(self, data_path: str, distance: int, max_walks: str,
                 train: bool, clustering: bool, chunksize: int, save_path: str,
                 retrain: bool, alignment_projection: bool, grid_path: str):
        """Initializes the rdf2vec kgEmbedding class with the given parameters.

        Args:
            data_path (str): Path to the data files.
            distance (int): Maximum distance for walks.
            max_walks (str): Maximum number of walks.
            train (bool): Whether to train the model.
            clustering (bool): Whether to perform clustering.
            chunksize (int): Size of chunks for processing.
            save_path (str): Path to save the results.
            retrain (bool): Whether to retrain the model.
            alignment_projection (bool): Whether to use alignment projection.
            grid_path (str): Path to the grid data.

        Raises:
            IndexError: If no parquet file is found in the data path.
        """
        # transform string to Path structure
        self.data_path = Path(data_path)
        # assign distance variable to class variable
        self.distance = distance
        # assign maximum walks to class variable
        self.max_walks = max_walks
        # assign train to class variable
        self.train = train
        # assign clustering to class variable
        self.clustering = clustering
        # assign chunksize to class variable
        self.chunksize = chunksize
        # assign savepath to class variable
        self.save_path = Path(save_path)
        # assign retrain to class variable
        self.retrain = retrain
        # assign alignment to class variable
        self.alignment = alignment_projection
        self.cpu_count = int(50)
        self.grid_path = grid_path
        # create logging directory Path name based on file name
        logging_directory = self.save_path
        # create logging directory
        logging_directory.mkdir(parents=True, exist_ok=True)
        # extract all file paths from directory
        # create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        directory_files = list(sorted(self.data_path.glob('**/*.parquet')))
        if not directory_files:
            raise IndexError('No parquet file has been found')
        # assign empty method to get accessed
        self.model = W2V(min_count=0, workers=self.cpu_count, seed=15)
        # if training variable is true, extract vectors in RDF2Vec
        directories = [directory.parent.stem for directory in directory_files if directory.parent.stem != "_delta_log"]
        directories = sorted(set(directories))
        directories = directories[5:]
        if self.train:
            for file_path in tqdm(directories, position=0, leave=True):
                date = re.findall("\d{4}-\d{2}-\d{2}", file_path)[0]
                graph_data = self.data_preparation(date)
                self.kg_training(graph_data, logging_directory, date)
        # extract stored models from loggingDirectory
        logging_files = list(sorted(logging_directory.glob('*.pkl')))
        # extract dates from stored files name
        logging_files_date = [re.findall(
            r'\d+-\d+-\d+|\d+', logging_file.stem)[0] for logging_file in logging_files]

        # construct file dictionary with loggingFilesDate as key and file name as value
        file_dict = dict(zip(logging_files_date, logging_files))
        # extract result dictionary with ID and corresponding year
        result_dict = self.kg_transformer(file_dict)
        # create vector representation as dataframe
        self.vector_df(result_dict)

    def data_preparation(self, filter_date: str) -> pd.DataFrame:
        """Prepares the data for training by filtering and formatting it.

        Args:
            filter_date (str): The date to filter the data by.

        Returns:
            pd.DataFrame: The prepared data.
        """
        # check if variable dataPath is a directory or a file
        # Read a Delta Table containing the entities we want to classify
        graph_data = DeltaTable(self.data_path)
        graph_data = graph_data.to_pandas(partitions=[("date", "=", filter_date)])
        # change all columns to object type
        graph_data = graph_data.astype(str)
        # check if subject column is in table
        if 'subject' not in graph_data.columns:
            graph_data = graph_data.rename({'from': 'subject', 'description': 'predicate', 'to': 'object'}, axis=1)
        else:
            pass
        # order columns in determined order
        if 'ID' in graph_data.columns:
            graph_data = graph_data[['subject', 'object', 'predicate', 'ID']]
        else:
            graph_data = graph_data[['subject', 'object', 'predicate']]
        # return prepared edge dataframe
        return graph_data

    def predicate_generation(self, path_list: str) -> list:
        """Generates a sequence of predicates for a given path.

        Args:
            path_list (str): The path for which to generate predicates.

        Returns:
            list: A list of predicates for the given path.
        """
        # assign class graph to graph variable
        graph = self.graph
        # extract predicate of edge given edge id stored in numpy
        pred_values = [e.attributes()['predicate'] for e in graph.es(path_list)]
        # extract node sequences that are part of the edge path and flatten numpy array
        node_sequence = np.array([graph.vs().select(e.tuple).get_attribute_values(
            'name') for e in graph.es(path_list)]).flatten()
        # delete consecutive character values in numpy array based from prior matrix
        node_sequence = np.array([key for key, _group in groupby(node_sequence)]).tolist()
        # combine predicate values and node sequences to one single array
        if node_sequence:
            path_sequence = [node_sequence[0], pred_values[0], node_sequence[1]]
        else:
            path_sequence = []
        # return path sequence numpy array
        return path_sequence

    def walk_iteration(self, id_number: int) -> list:
        """Performs a walk iteration for a given node ID.

        Args:
            id_number (int): The node ID for which to perform the walk iteration.

        Returns:
            list: A list of walk sequences.
        """
        # assign class graph variable to local graph variable
        graph = self.graph
        # assign class maxWalks variable to local maxWalks variable
        max_walks = self.max_walks
        # extract index of graph node
        node_index = graph.vs.find(id_number).index
        # perform breadth-first search algorithm
        bfs_list = graph.bfsiter(node_index, 'out', advanced=True)
        # iterate over breadth-first search iterator object to filter those paths out
        # defined distance variable
        distance_list = [
            node_path for node_path in bfs_list if node_path[1] <= self.distance]
        # create vertex list from distance list extracting vertex element
        vertex_list = [vertex_element[0] for vertex_element in distance_list]
        # check if all paths should be extracted
        if max_walks == -1:
            pass
        else:
            # limit maximum walks to maximum length of walkSequence length
            vertex_list_len = len(vertex_list)
            if vertex_list_len < max_walks:
                max_walks = vertex_list_len
            # random sample defined maximumWalk from vertexList list
            random.seed(15)
            vertex_list = random.sample(vertex_list, max_walks)
        # compute shortest path from focused node index to extracted vertex list outputting edge ID
        shortest_path_list = graph.get_shortest_paths(
            v=node_index, to=vertex_list, output='epath')
        # extract walk sequences with edge id to generate predicates
        walk_sequence = list(map(self.predicate_generation, shortest_path_list))
        # return walkSequence list
        return walk_sequence

    def kg_vector_extraction(self, entity: str) -> dict:
        """Extracts the vector representation of a given entity.

        Args:
            entity (str): The entity for which to extract the vector representation.

        Returns:
            dict: The vector representation of the entity.
        """
        try:
            vector = self.model.wv.get_vector(entity)
        except:
            vector = {}
        return vector

    def kg_training(self, graph_data: pd.DataFrame, loggingPath: str, date: str) -> None:
        """Trains the knowledge graph model with the provided data.

        Args:
            graph_data (pd.DataFrame): The data to train the model on.
            loggingPath (str): The path to save the logging information.
            date (str): The date for the current training iteration.
        """
        if 'ID' in graph_data.columns:
            entities = pd.unique(graph_data.pop('ID'))
        elif not self.grid_path:
            entities = pd.unique(graph_data['subject'])
        else:
            grid_data = gpd.read_parquet(self.grid_path)
            entities = pd.unique(grid_data['h3_id'])

        # transform values of row values dataframe into list
        graph_values = graph_data.to_records(index=False)
        # initialize Knowledge Graph
        self.graph = Graph().TupleList(
            graph_values, directed=True, edge_attrs=['predicate'])
        print(self.graph.summary())
        # initialize multiprocessing pool with cpu number
        pool = mp.Pool(self.cpu_count)
        # extract walk predicates using the walkIteration method
        walkPredicateList = list(tqdm(pool.imap_unordered(self.walk_iteration, entities, chunksize=self.chunksize),
                                      desc=f'Walk Extraction {date}', total=len(entities), position=0, leave=True))
        # walkPredicateList = list(pool.imap_unordered(self.walkIteration, entities, chunksize=self.chunksize))
        # close multiprocessing pool
        pool.close()
        # build up corpus on extracted walks
        corpus = [
            walk for entity_walks in walkPredicateList for walk in entity_walks]
        # retrain routine
        # check if retraining is true
        if self.retrain:
            # retrieve class model states
            model = self.model
            # check if already word vector exists
            if len(self.model.wv) == 0:
                # pass corpus to build vocabolary for Word2Vec model
                model.build_vocab(corpus)
            else:
                # pass corpus to build vocabolary for Word2Vec model
                model.build_vocab(corpus, update=True)
            # train Word2Vec model on corpus
            model.train(corpus, total_examples=model.corpus_count, epochs=10)
            # assign model back to class variable to enable retraining
            self.model = model
        # check if alignment is used
        elif self.alignment:
            # retrieve class model states
            model = W2V(min_count=0, workers=self.cpu_count, seed=15)
            # build up vocabulary of current iteration
            model.build_vocab(corpus)
            # train Word2Vec model on corpus
            model.train(corpus, total_examples=model.corpus_count, epochs=10)
            # check if previous model has been trained
            if len(self.model.wv) == 0:
                # no action as then nothing needs to be done
                pass
            else:
                # assign previous model stored in class variable to variable called previousModel
                previousModel = self.model
                # perform procrustes based alignment of gensim models with previous model
                model = smart_procrustes_align_gensim(previousModel, model)
            self.model = model
        else:
            # initialize Word2Vec model
            model = W2V(min_count=0, workers=self.cpu_count, seed=15, )
            # pass corpus to build vocabolary for Word2Vec model
            model.build_vocab(corpus)
            # train Word2Vec model on corpus
            model.train(corpus, total_examples=model.corpus_count, epochs=10)
            self.model = model
        # save trained model
        modelPath = f'{self.save_path}/entityVector{date}.pkl'
        entityVector = list(map(self.kg_vector_extraction, entities))
        dictEntity = dict(zip(entities, entityVector))
        with open(modelPath, 'wb') as f:
            pickle.dump(dictEntity, f)
        # delete variables with large memory consumption
        del corpus, walkPredicateList, model

    def kg_transformer(self, fileDict: dict) -> dict:
        """Transforms the embeddings from the given file dictionary into a result dictionary.

        Args:
            fileDict (dict): A dictionary where the keys are dates and the values are paths to the corresponding model files.

        Returns:
            dict: A dictionary where the keys are dates and the values are the loaded entity vectors.
        """
        print('Transformation started')
        # initialize result dictionary
        resultDict = {}
        # iterate over given transform dataframe
        for transformKey in tqdm(fileDict, desc='Transformation iteration'):
            modelFilePath = fileDict[transformKey]
            # Open the model file and load the entity vectors
            with open(modelFilePath, 'rb') as file:
                entityVector = pickle.load(file)
            # Store the entity vectors in the result dictionary
            resultDict[transformKey] = entityVector
        return resultDict

    def vector_df(self, vectorDict: dict) -> None:
        """Creates a DataFrame from the given vector dictionary and saves it as a parquet file.

        Args:
            vectorDict (dict): A dictionary where the keys are dates and the values are dictionaries of entity vectors.
        """
        kgVectorDfList = []
        # Iterate over each date in the vector dictionary
        for dateKey in tqdm(vectorDict.keys(), desc='Pandas Df generation'):
            valueDict = vectorDict[dateKey]
            # Generate column names for the vector components
            columnNameList = [f'osmVector{number}' for number in range(0, 100)]
            # Create a DataFrame from the entity vectors
            kgVectorDf = pd.DataFrame.from_dict(
                valueDict, orient='index', columns=columnNameList)
            kgVectorDf['ID'] = kgVectorDf.index
            kgVectorDf['DATE'] = dateKey
            kgVectorDfList.append(kgVectorDf)
        # Concatenate all DataFrames into one
        kgVectorCompleteDf = pd.concat(kgVectorDfList, ignore_index=True)
        # Save the complete DataFrame as a parquet file
        kgVectorCompleteDf.to_parquet(f'{self.save_path}/vectorDf.parquet', index=False)

if __name__ == '__main__':
    # initialize the command line argparser
    parser = argparse.ArgumentParser(description='RDF2Vec argument parameters')
    # add train argument parser
    parser.add_argument('-t', '--train', default=False, action='store_true',
                        help="use parameter if Word2Vec training should be performed")
    # add clustering argument
    parser.add_argument('-c', '--clustering', default=False, action='store_true',
                        help="use parameter if clustering should be performed on extracted vectors")
    # add path argument parser
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='string value to data path')
    parser.add_argument('-grid', '--grid_path', type=str, required=True,
                        help='string value to grid data path')
    # add distance argument parser
    parser.add_argument('-d', '--distance', type=int, required=True,
                        help='walk distance from selected node')
    # add walk number argument parser
    parser.add_argument('-w', '--walknumber', type=int, required=True,
                        help='maximum walk number from selected node')
    # add chunksize argument
    parser.add_argument('-chunk', '--chunksize', type=int, required=True,
                        help="use parameter to determine chunksize for parallel processing")
    parser.add_argument('-save', '--savepath', type=str, required=True,
                        help="use parameter to save path for files")
    parser.add_argument('-r', '--retrain', default=False, action='store_true',
                        help="use parameter if Word2Vec model should be retrained to align vector spaces")
    parser.add_argument('-a', '--alignmentprojection', default=False, action='store_true',
                        help="use parameter if extracted vectors should be aligned")
    # store parser arguments in args variable
    args = parser.parse_args()
    kgEmbedding(data_path=args.path, distance=args.distance, max_walks=args.walknumber, train=args.train,
                clustering=args.clustering, chunksize=args.chunksize, save_path=args.savepath, retrain=args.retrain,
                alignment_projection=args.alignmentprojection, grid_path=args.grid_path)