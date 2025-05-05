import numpy as np
import scipy.stats as st
import pdb
def _calculate_rational_payoff(signals, use_data, eval_data, gt, scoring_rule, ret_confidence: float | None = None):
    prior_action = use_data[gt].mean()

    if signals == []:
        score = eval_data[[gt]].copy()
        score['calibrated_action'] = prior_action
    else:
        grouped = use_data[signals + [gt]].groupby(signals, dropna=False).agg(calibrated_action=(gt, 'mean')).reset_index()
        score = eval_data[signals + [gt]].merge(grouped, on=signals, how="left").copy()
        score['calibrated_action'] = score['calibrated_action'].fillna(prior_action)
    
    score['payoff'] = scoring_rule(score['calibrated_action'].to_numpy(), score[gt].to_numpy())
    if ret_confidence is not None:
        return np.mean(score['payoff']), st.t.interval(ret_confidence, len(score['payoff']) - 1, loc=np.mean(score['payoff']), scale=st.sem(score['payoff']))
    return np.mean(score['payoff'])