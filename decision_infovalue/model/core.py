"""
Core API functionality for the Decision Info Model.
"""
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import numpy as np
import pandas as pd
from decision_infovalue.scoring_rules import _brier_score, _mse_score, _log_loss, _define_v_shaped_scoring_rule
from math import floor
from decision_infovalue.rational_agent import _calculate_rational_payoff

class DecisionInfoModel:
    """
    Core API class for decision information model.

    Attributes:
        dgp_data: pd.DataFrame, that is used to estimate the Data-generating Process
        state: str, Payoff-relevant state
        full_signals: List[str], signals that are included in the Data-generating Process
        scoring_rule: Callable, the scoring rule defining the decision making task
        binning_method: str, the binning method used to bin the signals
        overfit_tolerance: float, the tolerance for overfitting
        fit_test_ratio: float, the ratio of the data used to fit the model and the data used to test the model
        all_use_data: pd.DataFrame, the data after binning and the binning breaks
        all_breaks: Dict[str, List[float]], the binning breaks for each signal
        full_info_value: float, the full information value, i.e., the rational payoff when all signals are used
        no_info_value: float, the no information value, i.e., the rational payoff when no signal is used
    
    Methods:
        complement_info_value: Calculate the complement information value
        instanse_complement_info_value: Calculate the instance-level complement information value
    """
    
    def __init__(self, dgp_data: pd.DataFrame, 
                 state: str,
                 signals: List[str] | None = None,
                 scoring_rule: str = 'brier', 
                 binning_method: str = 'equal_probability',
                 overfit_tolerance: float = 0.1,
                 fit_test_ratio: float = 0.8, 
                 use_cache: bool = True,
                 verbose: bool = False):
        """Initialize the Decision Info Model.
        
        Args:
            dgp_data: pd.DataFrame, that is used to estimate the Data-generating Process
            state: str, Payoff-relevant state
            signals: List[str], signals that are included in the Data-generating Process
            scoring_rule: "brier", "mse", "log_loss", or "v_shaped_{kink}", the scoring rule defining the decision making task
            binning_method: "equal_probability", "equal_intervals", the binning method used to bin the signals. "equal_probability" specifies that the bins will be made at equal-probability intervals across the fitted distribution, i.e., the same number of observations in each bin, and "equal_intervals" specifies that the bins will be of equal length across the entire data set.
            overfit_tolerance: float, the tolerance for overfitting. If the overfit ratio, i.e., the ratio of the difference between the training and test rational payoffs to the training rational payoff, is greater than the overfit tolerance, the model will be overfitted.
            fit_test_ratio: float, the ratio of the data used to fit the model and the data used to test the model
            use_cache: bool, whether to use cache when calculating the complement information value and the instance-level complement information value
            verbose: bool, whether to print verbose output
        
        Raises:
            ValueError: If the input data is invalid
        """
        self._validate_input(dgp_data, state, signals)
        self.dgp_data = dgp_data.copy()

        # Define scoring rule
        if scoring_rule == 'brier':
            self.scoring_rule = _brier_score
        elif scoring_rule == 'mse':
            self.scoring_rule = _mse_score
        elif scoring_rule == 'log_loss':
            self.scoring_rule = _log_loss
        elif 'v_shaped' in scoring_rule:
            try:
                kink = float(scoring_rule.split('_')[-1])
            except ValueError:
                raise ValueError(f"Invalid kink in {scoring_rule}")
            if kink < 0 or kink > 1:
                raise ValueError(f"Invalid kink: {kink}")
            self.scoring_rule = _define_v_shaped_scoring_rule(kink)
        else:
            raise ValueError(f"Invalid scoring rule: {scoring_rule}")
        
        if binning_method != 'equal_probability' and binning_method != 'equal_intervals':
            raise ValueError(f"Invalid binning method: {binning_method}")

        # Define state
        self.state = state

        # Define signals
        if signals is None:
            self.full_signals = [col for col in dgp_data.columns if col != self.state]
        else:
            self.full_signals = signals
        if use_cache:
            self._cache: Dict[str, Tuple[float, Tuple[float, float]]] = {}
        else:
            self._cache = None

        self.binning_method = binning_method
        self.overfit_tolerance = overfit_tolerance
        self.fit_test_ratio = fit_test_ratio

        self.all_use_data, self.all_breaks = self._find_opt_binning(self.full_signals, binning_method, overfit_tolerance, fit_test_ratio, verbose)

        self.full_info_value = _calculate_rational_payoff(self.full_signals, self.all_use_data, self.all_use_data, self.state, self.scoring_rule)
        self.no_info_value = _calculate_rational_payoff([], self.all_use_data, self.all_use_data, self.state, self.scoring_rule)

    def _test_overfit(self, signals: List[str], data: pd.DataFrame, training_ratio: float, overfit_tolerance: float, verbose: bool = False) -> bool:
        """
        Test if the model is overfitting.
        """
        training_data = data[:floor(len(data) * training_ratio)]
        test_data = data[floor(len(data) * training_ratio):]
        training_rational_payoff = _calculate_rational_payoff(signals, 
                                                              training_data, training_data, 
                                                              self.state, self.scoring_rule)
        test_rational_payoff = _calculate_rational_payoff(signals, 
                                                          training_data, test_data, 
                                                          self.state, self.scoring_rule)
        if verbose:
            print(f"Training rational payoff: {training_rational_payoff}, Test rational payoff: {test_rational_payoff}, Overfit ratio: {(training_rational_payoff - test_rational_payoff) / abs(training_rational_payoff)}")
        return (training_rational_payoff - test_rational_payoff) < overfit_tolerance * abs(training_rational_payoff)

    def _find_opt_binning(self, signals: List[str], binning_method: str, overfit_tolerance: float, fit_test_ratio: float, verbose: bool = False) -> None:
        """
        Find the optimal binning for a feature.
        """
        if binning_method != 'equal_probability' and binning_method != 'equal_intervals':
            raise ValueError(f"Invalid binning method: {binning_method}")

        use_data = self.dgp_data[signals + [self.state]].copy()
        all_breaks = {feature: None for feature in signals}

        while not self._test_overfit(signals, use_data, fit_test_ratio, overfit_tolerance, verbose):
            if verbose:
                print("Overfitting detected, finding optimal binning...")
            feature_unique_values = use_data[signals].nunique()
            max_unique_num_feature = signals[np.argmax(feature_unique_values)]
            max_unique_num = feature_unique_values[max_unique_num_feature]
            if binning_method == 'equal_probability':
                _, breaks = pd.qcut(self.dgp_data[max_unique_num_feature], 
                                    q=int(max_unique_num/2), 
                                    retbins=True, 
                                    duplicates='drop')
            elif binning_method == 'equal_intervals':
                _, breaks = pd.cut(self.dgp_data[max_unique_num_feature], 
                                   bins=int(max_unique_num/2), 
                                   retbins=True, 
                                   duplicates='drop', include_lowest=True)
            use_data[max_unique_num_feature] = pd.cut(self.dgp_data[max_unique_num_feature], bins=breaks, labels=False, include_lowest=True)
            all_breaks[max_unique_num_feature] = breaks
        return use_data, all_breaks

    def _validate_input(self, data: pd.DataFrame, state: str, signals: List[str] | None) -> None:
        """
        Validate input data and target.
        
        Args:
            data: Input Data-generating Process
            
        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError("Data must be a pandas DataFrame")
        if state not in data.columns:
            raise ValueError(f"State {state} not found in data")
        if signals is not None:
            if not isinstance(signals, (list)):
                raise ValueError("Signals must be a list")
            if not all(isinstance(signal, str) for signal in signals):
                raise ValueError("All signals must be strings")
            if not all(signal in data.columns for signal in signals):
                raise ValueError("All signals must be in data")
            
    def _check_signals(self, signals: List[str] | str) -> None:
        """
        Check if the signals are valid.
        """
        if isinstance(signals, str):
            signals = [signals]
        if not all(signal in self.full_signals for signal in signals):
            raise ValueError("All signals must be in the definition of full signals")
        
        
    def complement_info_value(self, signals: List[str] | str, 
                              base_signals: List[str] | str | None = None, 
                              ret_confidence: float | None = None) -> float:
        
        '''
        Calculate the complement information value and the instance-level complement information value

        Args:
            signals: List[str] | str, the complement information value of which signals is calculated
            base_signals: List[str] | str | None, the signals that are substracted to calculate the complement information value
            ret_confidence: float | None, the confidence level to calculate the complement information value

        Returns:
            float, the complement information value, if `ret_confidence` is `None`, otherwise, a tuple of the complement information value and the confidence interval
        '''
        if base_signals is None:
            base_signals = []
        if isinstance(signals, str):
            signals = [signals]
        if isinstance(base_signals, str):
            base_signals = [base_signals]
        self._check_signals(signals)
        self._check_signals(base_signals)
        if self._cache is not None:
            cache_key = "global_" + str(tuple(signals)) + "_" + str(tuple(base_signals)) + "_" + str(ret_confidence)
            if cache_key in self._cache:
                if ret_confidence is not None:
                    return self._cache[cache_key]
                else:
                    return self._cache[cache_key][0]
        # all_use_data, _ = self._find_opt_binning(signals + base_signals, self.binning_method, self.overfit_tolerance, self.test_fit_ratio)
        if ret_confidence is not None:
            all_payoff, all_confidence = _calculate_rational_payoff(signals + base_signals, self.all_use_data, self.all_use_data, self.state, self.scoring_rule, ret_confidence)
            no_payoff, no_confidence = _calculate_rational_payoff(base_signals, self.all_use_data, self.all_use_data, self.state, self.scoring_rule, ret_confidence)
            if self._cache is not None:
                self._cache[cache_key] = (all_payoff - no_payoff, (all_confidence[0] - no_confidence[0], all_confidence[1] - no_confidence[1]))
            return all_payoff - no_payoff, (all_confidence[0] - no_confidence[0], all_confidence[1] - no_confidence[1])
        else:
            all_payoff = _calculate_rational_payoff(signals + base_signals, self.all_use_data, self.all_use_data, self.state, self.scoring_rule)
            no_payoff = _calculate_rational_payoff(base_signals, self.all_use_data, self.all_use_data, self.state, self.scoring_rule)
            if self._cache is not None:
                self._cache[cache_key] = (all_payoff - no_payoff, (None, None))
            return all_payoff - no_payoff
    
    def instanse_complement_info_value(self, signals: List[str] | str, 
                                       instance_signal_values: List[str] | List[float] | List[int] | str | float | int,
                                       counterfactual_signal: List[str] | str | None = None,
                                       counterfactual_signal_values: List[str] | List[float] | List[int] | str | float | int | None = None,
                                       base_signals: List[str] | str | None = None, 
                                       ret_confidence: float | None = None) -> float:
       
        '''
        Calculate the instance-level complement information value

        Args:
            signals: List[str] | str, the complement information value of which signals is calculated
            instance_signal_values: List[str] | List[float] | List[int] | str | float | int, the signal values of the instance
            counterfactual_signal: List[str] | str | None, the counterfactual signal
            counterfactual_signal_values: List[str] | List[float] | List[int] | str | float | int | None, the signal values of the counterfactual signal
            base_signals: List[str] | str | None, the signals that are substracted to calculate the complement information value
            ret_confidence: float | None, the confidence level to calculate the complement information value

        Returns:
            float, the complement information value, if `ret_confidence` is `None`, otherwise, a tuple of the complement information value and the confidence interval
        '''

        if base_signals is None:
            base_signals = []
        if isinstance(signals, str):
            signals = [signals]
        if isinstance(base_signals, str):
            base_signals = [base_signals]
        if isinstance(counterfactual_signal, str):
            counterfactual_signal = [counterfactual_signal]
        if not isinstance(instance_signal_values, list):
            instance_signal_values = [instance_signal_values]
        if counterfactual_signal is not None and not isinstance(counterfactual_signal_values, list):
            counterfactual_signal_values = [counterfactual_signal_values]

        if not all(signal in self.full_signals for signal in signals):
            raise ValueError(f"Signals must be in the definition of full signals: {signals} not in {self.full_signals}")
        if counterfactual_signal is not None:
            if not all(signal in self.full_signals for signal in counterfactual_signal):
                raise ValueError(f"Counterfactual signal must be in the definition of full signals: {counterfactual_signal} not in {self.full_signals}")
        
        if isinstance(instance_signal_values, list) and len(instance_signal_values) != len(signals):
            raise ValueError(f"Instance signal values must be the same length as signals: {len(instance_signal_values)} != {len(signals)}")
        if counterfactual_signal is not None and counterfactual_signal_values is not None:
            if len(counterfactual_signal_values) != len(counterfactual_signal):
                raise ValueError(f"Counterfactual signal values must be the same length as counterfactual signal: {len(counterfactual_signal_values)} != {len(counterfactual_signal)}")
        if counterfactual_signal is not None and counterfactual_signal_values is None:
            raise ValueError(f"Counterfactual signal values must be provided if counterfactual signal is provided")
        if counterfactual_signal_values is not None and counterfactual_signal is None:
            raise ValueError(f"Counterfactual signal must be provided if counterfactual signal values are provided")

        

        self._check_signals(signals)
        self._check_signals(base_signals)
        # all_use_data, breaks = self._find_opt_binning(signals + base_signals, self.binning_method, self.overfit_tolerance, self.test_fit_ratio)
        if counterfactual_signal is None:
            counterfactual_signal = signals
        if counterfactual_signal_values is None:
            counterfactual_signal_values = instance_signal_values

        instance_signal_values = [min(max(value, self.dgp_data[signals[i]].min()), self.dgp_data[signals[i]].max()) 
                            for i, value in enumerate(instance_signal_values)]
        counterfactual_signal_values = [min(max(value, self.dgp_data[counterfactual_signal[i]].min()), self.dgp_data[counterfactual_signal[i]].max()) 
                            for i, value in enumerate(counterfactual_signal_values)]
        
        for i, signal in enumerate(signals):
            if self.all_breaks[signal] is not None:
                instance_signal_values[i] = pd.cut([instance_signal_values[i]], bins=self.all_breaks[signal], labels=False,include_lowest=True)[0]
        for i, signal in enumerate(counterfactual_signal):
            if self.all_breaks[signal] is not None:
                counterfactual_signal_values[i] = pd.cut([counterfactual_signal_values[i]], bins=self.all_breaks[signal], labels=False,include_lowest=True)[0]
        if self._cache is not None:
            cache_key = "instance_" + str(tuple(signals)) + "_" + str(tuple(base_signals)) + "_" + str(tuple(instance_signal_values)) + "_" + str(tuple(counterfactual_signal)) + "_" + str(tuple(counterfactual_signal_values)) + "_" + str(ret_confidence)
            if cache_key in self._cache:
                if ret_confidence is not None:
                    return self._cache[cache_key]
                else:
                    return self._cache[cache_key][0]
        eval_data = self.all_use_data.loc[np.all(self.all_use_data[signals] == instance_signal_values, axis=1), :]
        use_data = self.all_use_data.loc[np.all(self.all_use_data[counterfactual_signal] == counterfactual_signal_values, axis=1), :]
        if ret_confidence is not None:
            all_payoff, all_confidence = _calculate_rational_payoff(base_signals, use_data, eval_data, self.state, self.scoring_rule, ret_confidence)
            no_payoff, no_confidence = _calculate_rational_payoff(base_signals, self.all_use_data, eval_data, self.state, self.scoring_rule, ret_confidence)
            if self._cache is not None:
                self._cache[cache_key] = (all_payoff - no_payoff, (all_confidence[0] - no_confidence[0], all_confidence[1] - no_confidence[1]))
            return all_payoff - no_payoff, (all_confidence[0] - no_confidence[0], all_confidence[1] - no_confidence[1])
        else:
            all_payoff = _calculate_rational_payoff(base_signals, use_data, eval_data, self.state, self.scoring_rule)
            no_payoff = _calculate_rational_payoff(base_signals, self.all_use_data, eval_data, self.state, self.scoring_rule)
            if self._cache is not None:
                self._cache[cache_key] = (all_payoff - no_payoff, (None, None))
            return all_payoff - no_payoff
    
    