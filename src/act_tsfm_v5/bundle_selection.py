"""Bundle selection and generation for paired evaluations on shared test windows."""

from __future__ import annotations
import numpy as np
from typing import Sequence


def generate_candidate_bundles(
    cell_id: str,
    window_id: str,
    leader: str,
    active_arms: Sequence[str],
    evaluator,
    max_bundle_size: int = 2
) -> list[tuple[list[str], str, float]]:
    """Generates candidate evaluation bundles a = (S, w, c).

    Returns list of (models_subset, window_id, bundle_cost).
    """
    unrevealed = [m for m in active_arms if (m, window_id) not in evaluator.revealed_observations]
    if not unrevealed:
        return []

    bundles = []
    
    # 1. Leader + Challenger paired bundle
    if leader in unrevealed:
        challengers = [m for m in unrevealed if m != leader]
        for ch in challengers:
            pair = [leader, ch]
            if evaluator.can_afford(pair, window_id):
                cost = evaluator.measure_bundle_cost(pair, window_id)
                bundles.append((pair, window_id, cost))
    else:
        # Leader already revealed on this window: evaluate single challengers paired against cached leader
        for ch in unrevealed:
            if evaluator.can_afford([ch], window_id):
                cost = evaluator.measure_bundle_cost([ch], window_id)
                bundles.append(([ch], window_id, cost))

    # 2. Single unrevealed arm fallback
    for m in unrevealed:
        if evaluator.can_afford([m], window_id):
            cost = evaluator.measure_bundle_cost([m], window_id)
            bundles.append(([m], window_id, cost))

    return bundles

