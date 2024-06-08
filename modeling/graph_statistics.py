from deltalake import DeltaTable
from pathlib import Path
import re
from tqdm import tqdm
from igraph import Graph, mean
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to calculate graph data relation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to graph data parquet file')
    args = parser.parse_args()
    data_path = Path(args.data_path)
    directory_files = list(sorted(data_path.glob('**/*.parquet')))
    directories = [directory.parent.stem for directory in directory_files if directory.parent.stem != "_delta_log"]
    directories = sorted(set(directories))
    for file_path in tqdm(directories, position=0, leave=True):
        date = re.findall("\d{4}-\d{2}-\d{2}", file_path)[0]
        graph_data = DeltaTable(data_path)
        graph_data = graph_data.to_pandas(partitions=[("date", "=", date)])
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
        
        graph_values = graph_data.to_records(index=False)
        graph = Graph().TupleList(graph_values, directed=True, edge_attrs=['predicate'])
        print(date)
        print(graph.summary())
        print(round(mean(graph.degree(mode="out")),4))