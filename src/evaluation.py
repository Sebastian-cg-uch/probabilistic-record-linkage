"""
Evaluation of the linkage with an INDEPENDENT gold standard.

Why not evaluate with exact agreement on names? Because those are the same fields the model
compares: pairs that agree exactly would be classified as matches almost by construction, and
the metrics would look better than they are.

Instead, the validation experiment uses the RUN (national ID), which the model never sees:
  1. take persons and discharges that DO have a RUN,
  2. hide the RUN and link them using names, country and birth year only,
  3. compare the links with the truth "same RUN = same person".

A pair is a true match when both records carry the same RUN (RUN_TRUE).
"""
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


# --------------------------------------------------------------------------------------
# Pair-level metrics
# --------------------------------------------------------------------------------------
def count_true_pairs(persons_input, discharges_input):
    """All (person, discharge) pairs sharing a RUN, regardless of blocking: the recall denominator."""
    p = persons_input[["unique_id", "RUN_TRUE"]].dropna()
    e = discharges_input[["unique_id", "RUN_TRUE"]].dropna()
    return len(p.merge(e, on="RUN_TRUE"))


def mark_true_pairs(oriented):
    out = oriented.copy()
    out["is_true_pair"] = (
        out["RUN_TRUE_pers"].notna()
        & out["RUN_TRUE_egr"].notna()
        & (out["RUN_TRUE_pers"] == out["RUN_TRUE_egr"])
    )
    return out


def threshold_sweep(oriented, n_true_pairs, thresholds=DEFAULT_THRESHOLDS):
    """Precision, recall, FDR (false discovery rate) and FNR (false negative rate) by threshold."""
    pairs = mark_true_pairs(oriented)
    rows = []
    for t in thresholds:
        predicted = pairs[pairs["match_probability"] >= t]
        tp = int(predicted["is_true_pair"].sum())
        fp = len(predicted) - tp
        fn = n_true_pairs - tp
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / n_true_pairs if n_true_pairs else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan
        rows.append({
            "threshold": t, "predicted_pairs": len(predicted), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "FDR": 1 - precision if not np.isnan(precision) else np.nan,
            "FNR": 1 - recall if not np.isnan(recall) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Assignment-level report (what the pipeline actually does with the model)
# --------------------------------------------------------------------------------------
def assignment_report(assignments, persons_input, discharges_input):
    """
    Classify every discharge by outcome. A discharge "has a person" when its true RUN exists
    in the registry; discharges of people missing from the registry should stay unassigned.
    """
    egr = discharges_input[["unique_id", "RUN_TRUE"]].dropna().rename(columns={"unique_id": "unique_id_egr"})
    egr["has_person_in_registry"] = egr["RUN_TRUE"].isin(persons_input["RUN_TRUE"].dropna())
    merged = egr.merge(
        assignments[["unique_id_egr", "RUN_TRUE_pers"]], on="unique_id_egr", how="left"
    )
    assigned = merged["RUN_TRUE_pers"].notna()
    correct = assigned & (merged["RUN_TRUE_pers"] == merged["RUN_TRUE"])
    outcome = np.select(
        [correct, assigned & ~correct, ~assigned & merged["has_person_in_registry"]],
        ["correct_assignment", "wrong_assignment", "missed_assignment"],
        default="correctly_unassigned",
    )
    counts = pd.Series(outcome).value_counts()
    n = len(merged)
    n_with_person = int(merged["has_person_in_registry"].sum())
    report = {
        "discharges_evaluated": n,
        "discharges_with_person_in_registry": n_with_person,
        "correct_assignment": int(counts.get("correct_assignment", 0)),
        "wrong_assignment": int(counts.get("wrong_assignment", 0)),
        "missed_assignment": int(counts.get("missed_assignment", 0)),
        "correctly_unassigned": int(counts.get("correctly_unassigned", 0)),
    }
    done = report["correct_assignment"] + report["wrong_assignment"]
    report["assignment_precision"] = round(report["correct_assignment"] / done, 4) if done else np.nan
    report["assignment_coverage"] = (
        round(report["correct_assignment"] / n_with_person, 4) if n_with_person else np.nan
    )
    return report


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------
def add_uncertainty(oriented):
    """Binary entropy of the match probability: 0 = certain, 1 = a coin flip."""
    out = oriented.copy()
    p = out["match_probability"].clip(1e-9, 1 - 1e-9)
    out["entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    out["confidence_band"] = pd.cut(
        out["match_probability"], bins=[0, 0.5, 0.7, 0.9, 0.99, 1.0],
        labels=["<50%", "50-70%", "70-90%", "90-99%", ">=99%"], include_lowest=True,
    )
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def save_figures(oriented, sweep, out_dir, operating_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = mark_true_pairs(oriented)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(pairs.loc[~pairs["is_true_pair"], "match_probability"], bins=bins, alpha=0.7,
            label="Different people (same RUN = no)", color="#c0504d")
    ax.hist(pairs.loc[pairs["is_true_pair"], "match_probability"], bins=bins, alpha=0.7,
            label="Same person (same RUN)", color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("Estimated match probability")
    ax.set_ylabel("Candidate pairs (log scale)")
    ax.set_title("Match probability by true status (hidden-RUN validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probability_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision")
    axes[0].plot(sweep["threshold"], sweep["recall"], marker="o", label="Recall")
    axes[0].plot(sweep["threshold"], sweep["f1"], marker="o", label="F1")
    axes[0].axvline(operating_threshold, color="gray", linestyle="--", label="Operating threshold")
    axes[0].set_xlabel("Probability threshold")
    axes[0].set_title("Precision, recall and F1")
    axes[0].legend()
    axes[1].plot(sweep["threshold"], sweep["FDR"], marker="o", label="FDR (false discovery rate)")
    axes[1].plot(sweep["threshold"], sweep["FNR"], marker="o", label="FNR (false negative rate)")
    axes[1].axvline(operating_threshold, color="gray", linestyle="--")
    axes[1].set_xlabel("Probability threshold")
    axes[1].set_title("Error rates")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_by_threshold.png", dpi=150)
    plt.close(fig)
