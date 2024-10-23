# Description: Utility functions for REMA-SM
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_datetime64_any_dtype
import numpy as np
import warnings

lm_map = {
    "roberta": "roberta-base",
    "mpnet": "microsoft/mpnet-base",
    "arctic": "Snowflake/snowflake-arctic-embed-m-v1.5",
}


def get_dataset_paths(dataset):
    dataset_map = {
        "gdc": "GDC",
        "chembl": "ChEMBL",
        "opendata": "OpenData",
        "tpcdi": "TPC-DI",
        "wikidata": "Wikidata",
    }

    task_map = {
        "joinable": "Joinable",
        "semjoinable": "Semantically-Joinable",
        "unionable": "Unionable",
        "viewunion": "View-Unionable",
    }

    if "-" in dataset:
        task = dataset.split("-")[1]
        dataset = dataset.split("-")[0]
        data_dir = f"datasets/{dataset_map[dataset]}/{task_map[task]}"
    else:
        data_dir = f"datasets/{dataset_map[dataset]}"

    return (
        f"{data_dir}/source-tables",
        f"{data_dir}/target-tables",
        f"{data_dir}/matches.csv",
    )


def to_lowercase(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()
    return df


def process_tables(source_table, target_table):
    processed_source_table = to_lowercase(source_table)
    processed_target_table = to_lowercase(target_table)
    return processed_source_table, processed_target_table


def get_samples(values, n=15, random=False):
    unique_values = values.dropna().unique()
    if random:
        tokens = np.random.choice(
            unique_values, min(15, len(unique_values)), replace=False
        )
    else:
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n)
        tokens = most_frequent_values.index.tolist()
    return [str(token) for token in tokens]


def infer_column_dtype(column, datetime_threshold=0.9):
    # Try converting to numeric (int or float)
    temp_col = pd.to_numeric(column, errors="coerce")
    if not temp_col.isnull().all():
        if is_integer_dtype(temp_col.dtype):
            return "integer"
        elif is_float_dtype(temp_col.dtype):
            return "float"

    # Suppress warnings from datetime conversion to avoid user confusion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            temp_col = pd.to_datetime(column, errors="coerce")
            if not temp_col.isnull().all() and (
                temp_col.notna().sum() / len(temp_col) >= datetime_threshold
            ):
                return "datetime"
        except Exception:
            pass

    # Default to categorical if other conversions fail
    return "categorical"


def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
