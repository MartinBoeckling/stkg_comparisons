from pykeen.triples import TriplesFactory
from pykeen.models import ComplEx, TransE
from pykeen.pipeline import pipeline
import os
from pathlib import Path
from deltalake import DeltaTable
import re
import argparse

def pykeen_training(kg_path: str, logging_directory: str, model_name: str, device_name: str, checkpoint_path: str) -> None:
    """Trains a knowledge graph embedding model using PyKEEN and saves the results to a specified logging directory.

    Args:
        kg_path (str): The path to the knowledge graph data.
        logging_directory (str): The directory where logging information and model checkpoints will be saved.
        model_name (str): The name of the model to be trained ('complex', 'transe', 'compgcn').
        device_name (str): The name of the device to be used for training (e.g., 'cpu', 'cuda').
        checkpoint_path (str): The path to the directory where checkpoints will be saved.

    Raises:
        NotImplementedError: If the specified model_name is not implemented.
    """
    # Set the environment variable for PyKEEN to specify the checkpoint path
    os.environ["PYKEEN_HOME"] = checkpoint_path
    # Define the folder path and get all parquet files
    folder_path = Path(kg_path)
    directory_files = list(sorted(folder_path.glob('**/*.parquet')))
    # Get unique directories containing the parquet files, excluding '_delta_log'
    directories = [directory.parent.stem for directory in directory_files if directory.parent.stem != "_delta_log"]
    directories = sorted(set(directories))
    for directory in directories:
        # Extract the date from the directory name
        date = re.findall("\d{4}-\d{2}-\d{2}", directory)[0]
        # Create a logging directory for the specific date
        logging_directory = Path(f'{logging_directory}/{date}')
        logging_directory.mkdir(exist_ok=True, parents=True)
        # Load data from DeltaTable and convert to pandas DataFrame
        data = DeltaTable(folder_path)
        data = data.to_pandas(partitions=[("date", "=", date)])
        # Rename columns if 'subject' is not present
        if 'subject' not in data.columns:
            data = data.rename({'from': 'subject', 'to': 'object', 'description': 'predicate'}, axis=1)
        else:
            pass
        # Convert data to numpy array
        data_array = data[['subject', 'predicate', 'object']].values
        # Initialize the appropriate model based on the model_name
        if model_name == "complex":
            triples_factory = TriplesFactory.from_labeled_triples(
                triples=data_array
            )
            model = ComplEx(triples_factory=triples_factory, random_seed=14)
        elif model_name == "transe":
            triples_factory = TriplesFactory.from_labeled_triples(
                triples=data_array
            )
            model = TransE(triples_factory=triples_factory, random_seed=14)
        elif model_name == "compgcn":
            triples_factory = TriplesFactory.from_labeled_triples(
                triples=data_array, create_inverse_triples=True
            )
        else:
            # Raise an error if the model_name is not implemented
            raise NotImplementedError(f"The model '{model_name}' is not implemented. within this script")
        # Split the data into training, testing, and validation sets
        training_triples, testing_triples, validation_triples = triples_factory.split(ratios = [.8, .1, .1], random_state=14)
        pipeline_results = pipeline(
            training=training_triples,
            testing=testing_triples,
            validation=validation_triples,
            model=model,
            random_seed=14,
            epochs=100,
            stopper='early',
            result_tracker='wandb',
            result_tracker_kwargs={'project':'stkg_comparison_callback_1'},
            device=device_name
        )
        # Save the results to the logging directory
        pipeline_results.save_to_directory(logging_directory)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyKeen argument parser')
    parser.add_argument("--kg_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    pykeen_training(kg_path = args.kg_path, logging_directory = args.log_dir, model_name = args.model, device_name = args.device, checkpoint_path = args.checkpoint)