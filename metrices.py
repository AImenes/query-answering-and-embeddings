import pandas as pd

def compute_hits_at_k(df, k):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas dataframe.")

    required_columns = [
        "local_kg_hit_rewriting",
        "online_kg_hit_rewriting",
        "local_kg_hit_original",
        "online_kg_hit_original",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Dataframe must contain the column: {column}")

    hits = {}
    #print(df)
    for column in required_columns:
        numerator = (df[column].head(k).sum())
        hits[column] = numerator / k

    return hits

def compute_mrr(df):
    """
    Computes the Mean Reciprocal Rank (MRR) for the four specified boolean columns
    in a given result dataframe.

    Args:
        df (pd.DataFrame): A dataframe containing the boolean columns:
                           "local_kg_hit_rewriting", "online_kg_hit_rewriting",
                           "local_kg_hit_original", "online_kg_hit_original".

    Returns:
        dict: A dictionary containing the MRR results for the four columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas dataframe.")

    required_columns = [
        "local_kg_hit_rewriting",
        "online_kg_hit_rewriting",
        "local_kg_hit_original",
        "online_kg_hit_original",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Dataframe must contain the column: {column}")

    mrr = {}

    #Optimistic mrr
    for column in required_columns:
        first_hit_rank = (df[df[column] == True].index[0] + 1) if df[column].any() else 0
        mrr[column] = (1 / first_hit_rank) if first_hit_rank > 0 else 0

    return mrr

def average_metrics(metrics_list, n):
    """
    Calculate the average for every common key across the dictionaries in the input list.

    Args:
        metrics_list (list): A list of dictionaries, each containing 'hits@k' and 'mrr' keys with
                             corresponding nested dictionaries as values. The nested dictionaries
                             should have keys: "local_kg_hit_rewriting", "online_kg_hit_rewriting",
                             "local_kg_hit_original", "online_kg_hit_original".

    Returns:
        dict: A dictionary containing the average values for 'hits@k' and 'mrr' for each column type.
    """
    if not all(isinstance(d, dict) for d in metrics_list):
        raise ValueError("All elements in the input list must be dictionaries.")

    total_metrics = {f'hits@{n}': {"local_kg_hit_rewriting": 0, "online_kg_hit_rewriting": 0,
                                "local_kg_hit_original": 0, "online_kg_hit_original": 0},
                     'mrr': {"local_kg_hit_rewriting": 0, "online_kg_hit_rewriting": 0,
                             "local_kg_hit_original": 0, "online_kg_hit_original": 0}}

    for metrics in metrics_list:
        for metric_type, values in metrics.items():
            for key, value in values.items():
                total_metrics[metric_type][key] += value

    count = len(metrics_list)
    avg_metrics = {metric_type: {key: value / count for key, value in values.items()}
                   for metric_type, values in total_metrics.items()}

    return avg_metrics