import numpy as np

def _brier_score(action, gt):
    return 1-(action - gt) ** 2


def _mse_score(action, gt):
    return -(action - gt) ** 2

def _log_loss(action, gt):
    return gt * np.log(action) + (1 - gt) * np.log(1 - action)

def _define_v_shaped_scoring_rule(kink):
    def _v_shaped_scoring_rule(action, state):
        if action <= kink:
            return 1/2 - 0.5 * (state - kink) / max(1 - kink, kink)
        else:
            return 1/2 + 0.5 * (state - kink) / max(1 - kink, kink)
    return _v_shaped_scoring_rule