"""
Shared test fixtures for Info Value Toolkit tests.
"""
import pytest
import pandas as pd
import numpy as np
from decision_infovalue.datasets import get_dataset


@pytest.fixture(scope="session")
def housing_data():
    """Load the housing dataset for testing."""
    df, meta = get_dataset('housing')
    return df, meta


@pytest.fixture(scope="session")
def recidivism_data():
    """Load the recidivism dataset for testing."""
    df, meta = get_dataset('recidivism')
    return df, meta


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.uniform(0, 1, 100)
    })
    # data['pred'] = data['feature1'] + data['feature2']
    data['target'] = data['feature1'] + data['feature2']
    
    metadata = {
        "name": "Sample Data",
        "source": "Test",
        "n_samples": len(data),
        "n_features": data.shape[1],
        "feature_names": list(data.columns),
        "target_name": "target",
        "description": "Sample data for testing"
    }
    return data, metadata