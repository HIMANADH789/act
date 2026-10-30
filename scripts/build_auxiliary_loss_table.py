"""Builds auxiliary cheap-model (HAR and GARCH) loss table across commissioning windows.

Records:
1. Common auxiliary preparation cost.
2. Loss profiles across all 258 commissioning windows.
"""

import os
import pandas as pd


def build_auxiliary_loss_table():
    print("=" * 60)
    print("BUILDING AUXILIARY CHEAP-MODEL (HAR & GARCH) LOSS TABLE")
    print("=" * 60)

    comm_ref = pd.read_csv("artifacts/reference/commissioning_reference.csv")
    
    # Filter HAR and GARCH
    cheap_df = comm_ref[comm_ref["model_id"].isin(["har", "garch"])].copy()
    
    total_prep_cost = float(cheap_df["inference_cost"].sum())
    n_records = len(cheap_df)
    
    os.makedirs("artifacts/reference", exist_ok=True)
    out_path = "artifacts/reference/auxiliary_cheap_losses.csv"
    cheap_df.to_csv(out_path, index=False)
    
    print(f"Total auxiliary cheap-model records: {n_records}")
    print(f"Common auxiliary preparation cost: {total_prep_cost:.4f} seconds")
    print(f"Saved to: {out_path}")
    
    return cheap_df, total_prep_cost


if __name__ == "__main__":
    build_auxiliary_loss_table()

