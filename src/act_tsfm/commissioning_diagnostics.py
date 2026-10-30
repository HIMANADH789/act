"""Evaluator-side diagnostics. Never imported by acquisition or policy code."""
import numpy as np

def regret_contract(loss: np.ndarray,w: np.ndarray,epsilon: float) -> dict:
    """All inputs are on the same calibration-normalised loss scale.

    The primary weighted objective and the all-cell stopping contract differ.
    This function diagnoses their finite-reference feasibility, not population truth.
    """
    loss=np.asarray(loss,float);w=np.asarray(w,float)
    if loss.ndim!=2 or loss.shape[1]!=len(w) or not np.isfinite(loss).all():
        raise ValueError('Complete finite model-by-cell means are required')
    if not np.all(w>0) or not np.isclose(w.sum(),1) or epsilon<0:
        raise ValueError('Positive normalised target weights and nonnegative epsilon required')
    per_cell=loss-loss.min(axis=0)
    score=loss@w;chosen=int(score.argmin())
    worst=per_cell.max(axis=1);minimax=int(worst.argmin())
    return dict(weighted_global_model=chosen,minimax_model=minimax,
        weighted_global_loss=float(score[chosen]),regime_oracle_loss=float(loss.min(0)@w),
        price_of_single_model=float(score[chosen]-loss.min(0)@w),
        primary_weighted_regret=0.,global_weighted_contract_feasible=True,
        chosen_worst_cell_regret=float(worst[chosen]),minimum_worst_cell_regret=float(worst[minimax]),
        all_cell_single_model_feasible=bool(worst[minimax]<=epsilon),
        regime_specific_oracle_max_regret=0.,regime_specific_contract_feasible=True,
        cell_winners=loss.argmin(axis=0),cell_regrets=per_cell)

def posterior_contracts(draws: np.ndarray,selected: int,w: np.ndarray,epsilon: float) -> dict:
    d=np.asarray(draws).reshape(len(draws),-1,len(w))
    per_cell=d-d.min(axis=1,keepdims=True)
    score=d@w
    mean=d.mean(0)
    routing=mean.argmin(0)
    route_regret=per_cell[:,routing,np.arange(len(w))]
    return dict(primary_selected_joint_probability=float((per_cell[:,selected,:]<=epsilon).all(1).mean()),
        any_single_model_joint_feasibility_probability=float((per_cell.max(2)<=epsilon).any(1).mean()),
        exploratory_global_weighted_probability=float((score[:,selected]-score.min(1)<=epsilon).mean()),
        exploratory_routing_joint_probability=float((route_regret<=epsilon).all(1).mean()),
        routing=routing,cell_probabilities=(route_regret<=epsilon).mean(0))

def corrected_recommendation_flags(recommend: bool,probability: float,worst: float,
                                   epsilon: float,confidence: float=.95) -> dict:
    incorrect=bool(recommend and worst>epsilon)
    return dict(incorrect_recommendation=int(incorrect),
                confidently_incorrect=int(incorrect and probability>=confidence))
