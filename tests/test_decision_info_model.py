"""
Tests for the Info Value Toolkit API.
"""
import pytest
import pandas as pd
import numpy as np
from decision_infovalue.model import DecisionInfoModel
from decision_infovalue.datasets import get_dataset


def test_calculate_aciv(sample_data):
    """Test the ACIV endpoint with sample data."""
    data, metadata = sample_data
    model = DecisionInfoModel(data, "target", ['feature1', 'feature2'])
    result1 = model.complement_info_value(['feature1'], ret_confidence=0.95)
    result2 = model.complement_info_value(['feature2'], ret_confidence=0.95)
    result3 = model.complement_info_value(['feature1', 'feature2'], ret_confidence=0.95)
    result4 = model.complement_info_value(['feature2'], ['feature1'], ret_confidence=0.95)
    result5 = model.complement_info_value(['feature1'], ['feature2'], ret_confidence=0.95)
    # result6 = model.complement_info_value(['feature1', 'feature2'], ['feature1', 'feature2'], ret_confidence=0.95)
    
    
    # Use pytest's built-in capfd fixture to capture output
    print(f"ACIV for feature1: {result1}")
    print(f"ACIV for feature2: {result2}") 
    print(f"ACIV for feature1 and feature2: {result3}")
    print(f"ACIV for feature1 and feature2, excluding feature1: {result4}")
    print(f"ACIV for feature1 and feature2, excluding feature2: {result5}")
    
    assert result3[0] > result1[0]
    assert result3[0] > result2[0]
    assert result3[0] > result4[0]
    assert result3[0] > result5[0]
    assert result2[0] > result4[0]
    assert result1[0] > result5[0]


def test_calculate_iliv(sample_data):
    """Test the ACIV endpoint with sample data."""
    data, metadata = sample_data
    model = DecisionInfoModel(data, "target", ['feature1', 'feature2'])
    result1 = model.instanse_complement_info_value(['feature1'], [0.1])
    result2 = model.instanse_complement_info_value(['feature2'], [0.5])
    # result3 = model.incomplete_complement_info_value(['feature1', 'feature2'], [0.1, 0.5])
    result4 = model.instanse_complement_info_value(['feature2'], [0.5], base_signals=['feature1'])
    result5 = model.instanse_complement_info_value(['feature1'], [0.1], base_signals=['feature2'])
    result6 = model.instanse_complement_info_value(['feature1'], [0.1], 
                                                     base_signals=['feature2'],
                                                     counterfactual_signal='feature1',
                                                     counterfactual_signal_values=[-0.1])
    
    
    # Use pytest's built-in capfd fixture to capture output
    print(f"ILIV for feature1: {result1}")
    print(f"ILIV for feature2: {result2}") 
    print(f"ILIV for feature2, excluding feature1: {result4}")
    print(f"ILIV for feature1, excluding feature2: {result5}")
    print(f"COunterfactual ILIV for feature1, excluding feature2: {result6}")
    
    # assert result3[0] > result1[0]
    # assert result3[0] > result2[0]
    # assert result3[0] > result4[0]
    # assert result3[0] > result5[0]
    # assert result2[0] > result4[0]
    # assert result1[0] > result5[0]


# def test_calculate_aciv_housing(housing_data):
#     """Test the ACIV endpoint with housing data."""
#     data, metadata = housing_data
#     model = DecisionInfoModel(data, "SalePrice", 
#                               ['Year Built', 'Overall Quality', 'Fireplaces', 'Car Garage', 'Year Remod/Add', 'Gr Liv Area'], 
#                               scoring_rule='mse')
#     result1 = model.complement_info_value(['Year Built'], ret_confidence=0.95)
#     result2 = model.complement_info_value(['Overall Quality'], ret_confidence=0.95)
#     result3 = model.complement_info_value(['Year Built', 'Overall Quality'], ret_confidence=0.95)
#     result4 = model.complement_info_value(['feature2'], ['feature1'], ret_confidence=0.95)
#     result5 = model.complement_info_value(['feature1'], ['feature2'], ret_confidence=0.95)
#     # result6 = model.complement_info_value(['feature1', 'feature2'], ['feature1', 'feature2'], ret_confidence=0.95)
    
    
#     # Use pytest's built-in capfd fixture to capture output
#     print(f"ACIV for feature1: {result1}")
#     print(f"ACIV for feature2: {result2}") 
#     print(f"ACIV for feature1 and feature2: {result3}")
#     print(f"ACIV for feature1 and feature2, excluding feature1: {result4}")
#     print(f"ACIV for feature1 and feature2, excluding feature2: {result5}")
    
#     assert result3[0] > result1[0]
#     assert result3[0] > result2[0]
#     assert result3[0] > result4[0]
#     assert result3[0] > result5[0]
#     assert result2[0] > result4[0]
#     assert result1[0] > result5[0]

# def test_calculate_aciv_housing(housing_data):
#     """Test the calculate_iv endpoint with housing data."""
#     data, metadata = housing_data
#     model = DecisionInfoModel(data, "target", ['feature1', 'feature2'])
#     result1 = model.complement_info_value(['feature1'], ret_confidence=0.95)
#     result2 = model.complement_info_value(['feature2'], ret_confidence=0.95)
#     result3 = model.complement_info_value(['feature1', 'feature2'], ret_confidence=0.95)
#     result4 = model.complement_info_value(['feature1', 'feature2'], ['feature1'], ret_confidence=0.95)
#     result = calculate_iv(X, y, 'median_income')
    
#     assert isinstance(result, dict)
#     assert result['feature'] == 'median_income'
#     assert result['status'] == 'success'
#     assert result['iv'] > 0  # IV should be positive for a relevant feature


# def test_analyze_feature(sample_data):
#     """Test the analyze_feature endpoint with sample data."""
#     data, target = sample_data
#     result = analyze_feature(data, target, 'feature1')
    
#     assert isinstance(result, dict)
#     assert 'feature' in result
#     assert 'iv' in result
#     assert 'chi_square' in result
#     assert 'woe' in result
#     assert 'statistics' in result
#     assert result['feature'] == 'feature1'
#     assert isinstance(result['woe'], dict)
#     assert all(key in result['statistics'] for key in ['mean', 'std', 'min', 'max'])


# def test_analyze_feature_recidivism(recidivism_data):
#     """Test the analyze_feature endpoint with recidivism data."""
#     X, y = recidivism_data
#     result = analyze_feature(X, y, 'predicted_decision')
    
#     assert isinstance(result, dict)
#     assert result['feature'] == 'predicted_decision'
#     assert isinstance(result['woe'], dict)
#     assert len(result['woe']) > 0  # Should have WoE values for each category


# def test_get_statistics(sample_data):
#     """Test the get_statistics endpoint with sample data."""
#     data, target = sample_data
#     result = get_statistics(data, target)
    
#     assert isinstance(result, dict)
#     assert 'features' in result
#     assert 'target' in result
#     assert all(key in result['target'] for key in ['mean', 'std', 'min', 'max', 'distribution'])
#     assert isinstance(result['target']['distribution'], dict)


# def test_get_statistics_housing(housing_data):
#     """Test the get_statistics endpoint with housing data."""
#     X, y = housing_data
#     result = get_statistics(X, y)
    
#     assert isinstance(result, dict)
#     assert len(result['features']) == X.shape[1]
#     assert all(col in result['features'] for col in X.columns)
#     assert isinstance(result['target']['distribution'], dict) 