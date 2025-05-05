"""
Dataset loading and processing utilities for the Info Value Toolkit.
"""
from typing import Tuple, Dict, Any, Final
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
from urllib.request import urlretrieve

github_data_url: Final[str] = "https://github.com/Guoziyang27/decision-infovalue/raw/main/data/"

def load_housing_data(with_human_data: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the housing price dataset.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    df = pd.read_csv(cache(github_data_url + "AmesHousing.csv"))

    if with_human_data:
        human_df = pd.read_csv(cache(github_data_url + "house_price_human.csv"))
        df = pd.merge(df, human_df, left_on="Order", right_on="Order", how="left")
    
    metadata = {
        "name": "Ames Iowa Housing Prices",
        "source": "Kaggle",
        "n_samples": len(df),
        "n_features": df.shape[1],
        "feature_names": list(df.columns),
        "target_name": "SalePrice",
        "description": "Ames Iowa Housing Prices dataset with various features"
    }
    
    return df, metadata


def load_recidivism_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the recidivism dataset.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    
    # Load the data
    human_df = pd.read_csv(cache("https://raw.githubusercontent.com/stanford-policylab/recidivism-predictions/refs/heads/master/data/public/surveys/df_response.csv"))
    human_df.drop(human_df[human_df['individual_id'].apply(lambda x: not (str(x).isdigit()))].index, inplace=True)
    human_df["individual_id"] = human_df["individual_id"].astype("int64")
    
    data_df = pd.read_csv(cache("https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores-two-years.csv"))
    
    data_df = pd.merge(data_df, human_df[["individual_id", "predicted_decision"]], left_on='id', right_on='individual_id', how='inner')
    
    # Create metadata
    metadata = {
        "name": "Recidivism Risk Assessment",
        "source": "Stanford Policy Lab",
        "n_samples": len(data_df),
        "n_features": data_df.shape[1],
        "feature_names": list(data_df.columns),
        "target_name": "two_year_recid",
        "description": "Recidivism risk assessment dataset with user responses"
    }
    
    return data_df, metadata

def get_dataset(name: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Get a dataset by name.
    
    Args:
        name: Name of the dataset ('housing' or 'recidivism')
        
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if name.lower() == 'housing':
        return load_housing_data(**kwargs)
    elif name.lower() == 'recidivism':
        return load_recidivism_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: 'housing', 'recidivism'") 
    
def cache(url: str, file_name: str | None = None) -> str:
    """Loads a file from the URL and caches it locally."""
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname(__file__), "cached_data")
    os.makedirs(data_dir, exist_ok=True)

    file_path: str = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path