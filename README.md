# Comparing Spatial-Temporal Knowledge Graph on spatial downstream tasks
This repository contains code and data for comparing spatial-temporal knowledge graphs (STKG) on various spatial downstream tasks. The comparisons include use cases like AirBnB and Wildfire.
## Prerequisites
- Python: Ensure you have Python installed. Recommended version is 3.11.6, which have been used for the scripts.
- R: R is required for some parts of the data analysis. Scripts are tested with R 4.2.3


> If dependencies are not installed via Conda, make sure to fulfill the requirements for the different packages
## Installation
### Python
Set up the Python environment:

```
conda env create -f environment.yml
conda activate stkg_comparisons
```
### R
Set up the R environment:

```
conda env create -f r-environment.yml
conda activate r-stkg_comparisons
```

## Usage
### Data
### Use Cases
Overall we differentiate between two datasets, which we use for the overall evaluation: The AirBnB dataset as well as the Wildfire dataset. In the following, the necessary script is linked
#### AirBnB
In order to run the experiments, please have a look at the shell file containing the script to run the experiments:

[Shell Script](use_cases/airbnb/script_run.sh)
#### Wildfire
