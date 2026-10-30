"""
End-to-end run on the SYNTHETIC data:

  1. generate synthetic data (if missing)
  2. preprocess the three tables
  3. VALIDATION: link persons <-> discharges with RUN hidden, evaluate against the true RUN
  4. PRODUCTION-STYLE RUN: link persons <-> discharges without RUN and assign RUNs
  5. save metrics, figures and the assignments in results/

Usage:  python run_pipeline.py [--threshold 0.85]
"""
import argparse
from pathlib import Path

import pandas as pd

from src import evaluation as ev
from src import model as md
from src import preprocessing as pp
from src import synthetic_data

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"


def main(threshold):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)

    if not (DATA / "dat_pers.csv").exists():
        synthetic_data.generate()

    clean = pp.process_all(
        pd.read_csv(DATA / "dat_pers.csv"),
        pd.read_csv(DATA / "dat_egr.csv"),
        pd.read_csv(DATA / "dat_egr_sin_rut.csv"),
    )
    persons = clean["persons"]

    # ------------------------------------------------------------------ 1. validation
    print("\n=== VALIDATION: RUN hidden, evaluated against the true RUN ===")
    val_input = md.make_model_input(persons, clean["discharges_with_run"], hide_run=True)
    linker = md.train(val_input)
    val_pairs = md.orient_pairs(md.predict_pairs(linker))

    persons_in = val_input[val_input["source_dataset"] == "PERS"]
    discharges_in = val_input[val_input["source_dataset"] == "EGR"]
    n_true = ev.count_true_pairs(persons_in, discharges_in)
    sweep = ev.threshold_sweep(val_pairs, n_true)
    sweep.round(4).to_csv(RESULTS / "validation_metrics_by_threshold.csv", index=False)

    val_assign = md.assign_runs(val_pairs, threshold)
    val_report = ev.assignment_report(val_assign, persons_in, discharges_in)
    pd.Series(val_report).to_csv(RESULTS / "validation_assignment_report.csv", header=["value"])
    ev.save_figures(val_pairs, sweep, RESULTS / "figures", threshold)

    print(f"True pairs (same RUN): {n_true:,}")
    print(sweep.round(3).to_string(index=False))
    print(f"\nAssignment report at threshold {threshold}:")
    for k, v in val_report.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------ 2. production-style
    print("\n=== PRODUCTION-STYLE RUN: discharges without RUN ===")
    prod_input = md.make_model_input(persons, clean["discharges_without_run"], hide_run=False)
    prod_linker = md.train(prod_input)
    prod_pairs = ev.add_uncertainty(md.orient_pairs(md.predict_pairs(prod_linker)))
    assignments = md.assign_runs(prod_pairs, threshold)

    n_discharges = int((prod_input["source_dataset"] == "EGR").sum())
    print(f"Discharges without RUN: {n_discharges:,}")
    print(f"Discharges that received a RUN (threshold {threshold}): {len(assignments):,}")
    print(f"Ambiguous (more than one candidate above threshold): {(assignments['n_candidates'] > 1).sum():,}")

    # Oracle check: only possible because the data are synthetic (true RUN of each discharge is known)
    prod_persons = prod_input[prod_input["source_dataset"] == "PERS"]
    prod_discharges = prod_input[prod_input["source_dataset"] == "EGR"]
    oracle = ev.assignment_report(assignments, prod_persons, prod_discharges)
    pd.Series(oracle).to_csv(RESULTS / "production_oracle_report_synthetic_only.csv", header=["value"])
    print("\nOracle check (synthetic data only):")
    for k, v in oracle.items():
        print(f"  {k}: {v}")

    # Save assignments WITHOUT the columns that reveal the truth
    public = assignments.drop(columns=["RUN_TRUE_pers", "RUN_TRUE_egr"])
    public.to_csv(RESULTS / "rut_assignments_synthetic.csv", index=False)
    print(f"\nResults saved in {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85, help="minimum match probability to assign a RUN")
    main(parser.parse_args().threshold)
