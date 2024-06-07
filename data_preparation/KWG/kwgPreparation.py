import pandas as pd
import re
import datetime
from tqdm import tqdm
from deltalake import write_deltalake
import argparse

pd.options.mode.chained_assignment = None  # default='warn'
def dateParser(inputPredicate: str) -> datetime.datetime | None:
    """Parses a date from the given input string.

    Args:
        inputPredicate (str): The input string containing the date.

    Returns:
        datetime.datetime | None: The parsed date as a datetime object with the day set to the first of the month,
                                  or None if no date is found in the input string.
    """
    # Convert input to string
    inputPredicate = str(inputPredicate)
    # Search for a date pattern in the input string
    date = re.search(r"(\d{4}-\d{2}-\d{2})", inputPredicate)
    if date:
        # Extract the date string
        date = date.group(1)
        # Convert the date string to a datetime object and set the day to the first of the month
        date = datetime.datetime.strptime(date, '%Y-%m-%d').replace(day=1)
    return date

def prepare_kgw(input_path: str, output_path: str, min_date: str, max_date: str, frequency: str) -> None:
    """Prepares the KnowWhereGraph (KWG) data by filtering and partitioning based on date.


    Args:
        input_path (str): The path to the input parquet file.
        output_path (str): The path to save the processed data.
        min_date (str): The start date for filtering the data (format: 'YYYY-MM-DD').
        max_date (str): The end date for filtering the data (format: 'YYYY-MM-DD').
        frequency (str): The frequency for generating date ranges (e.g., 'MS' for month start).
    """
    # Read data from the input parquet file
    data = pd.read_parquet(input_path)
    # Parse dates from the 'object' column
    data['date'] = data['object'].apply(dateParser)
    # Generate a date range based on the specified frequency
    dateRange = pd.date_range(min_date, max_date, freq=frequency).date
    # Iterate over each date in the generated date range
    for date in tqdm(dateRange):
        # Select data where the date is null
        dataChangedDate = data[data.date.isnull()]
        strDate = str(date)
        # Fill null dates with the current date in the loop
        dataChangedDate.date = dataChangedDate.date.fillna(strDate)
        # Select data where the date matches the current date in the loop
        dataMonthEvent = data[data.date == date]
        # Concatenate the data with changed dates and the data with matching dates
        preparedData = pd.concat([dataChangedDate, dataMonthEvent], ignore_index=True)
        # Set the 'date' column to the current date as string
        preparedData["date"] = strDate
        # Write the prepared data to the output path, partitioning by 'date'
        write_deltalake(output_path, preparedData, partition_by="date", overwrite_schema=True, mode="append")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparation of KnowWhereGraph data to triple data')
    parser.add_argument('--path', type=str, required=True, help='Path to the input parquet file')
    parser.add_argument('--target_path', type=str, required=True, help='Path to save the output parquet file')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for temporal dimension')
    parser.add_argument('--end_date', type=str, required=True, help='End date for temporal dimension')
    parser.add_argument('--time_frequency', type=str, required=True, help='Frequency of time dimension')
    args = parser.parse_args()
    prepare_kgw(args.path,
                args.target_path,
                args.start_date, args.end_date, args.time_frequency)