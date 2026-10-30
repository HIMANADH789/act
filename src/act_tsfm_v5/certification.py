"""Per-cell and global model certification manager."""

from __future__ import annotations
import numpy as np


class CertificationManager:
    """Certifies provisional leaders as epsilon-optimal in each cell:

        UCB(Delta_{Leader, B, c}) <= epsilon_c for all active challengers B,
        or when all challengers in the cell have been eliminated.
    """
    def __init__(
        self,
        cell_list: list[str],
        epsilon_cell: float = 0.05
    ):
        self.cells = list(cell_list)
        self.epsilon_cell = epsilon_cell
        self.certified_cells: dict[str, str] = {}
        self.global_certified: bool = False

    def evaluate_cell_certification(
        self,
        cell_id: str,
        provisional_leader: str,
        active_challengers: list[str],
        ucb_func
    ) -> bool:
        """Evaluates whether the provisional leader is certified in cell_id.

        ucb_func(m_lead, m_challenger, cell_id) -> float
        """
        # If leader is the only remaining arm in the cell, it is certified
        if len(active_challengers) <= 1:
            self.certified_cells[cell_id] = provisional_leader
            return True

        all_certified = True
        for ch in active_challengers:
            if ch == provisional_leader:
                continue
            # Difference Delta_{leader, ch} = Loss(leader) - Loss(ch)
            ucb = ucb_func(provisional_leader, ch, cell_id)
            if ucb > self.epsilon_cell:
                all_certified = False
                break

        if all_certified:
            self.certified_cells[cell_id] = provisional_leader
            return True
        return False

    def check_global_certification(self) -> bool:
        """Global certification is achieved when all 9 cells are certified."""
        if len(self.certified_cells) == len(self.cells):
            self.global_certified = True
        return self.global_certified

