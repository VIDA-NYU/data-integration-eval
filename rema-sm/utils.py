# Description: Utility functions for REMA-SM
import numpy as np


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


def get_samples(values, n=15):
    unique_values = values.dropna().unique()
    tokens = np.random.choice(unique_values, min(15, len(unique_values)), replace=False)
    return [str(token) for token in tokens]
