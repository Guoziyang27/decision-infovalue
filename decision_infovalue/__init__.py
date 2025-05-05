"""
Info Value Toolkit - A toolkit for information value analysis.
"""

__version__ = "0.1.0" 

from .datasets import get_dataset, load_housing_data, load_recidivism_data
from .model import DecisionInfoModel


__all__ = [
    'get_dataset',
    'load_housing_data',
    'load_recidivism_data',
    'DecisionInfoModel'
] 