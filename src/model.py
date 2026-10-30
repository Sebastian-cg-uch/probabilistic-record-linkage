"""
Probabilistic record linkage model (Fellegi-Sunter) built with Splink 5.

Task: link each hospital discharge that has no RUN (national ID) to a person in the
registry that does have one, so the discharge can inherit that RUN.

Model parameters are NOT fixed by hand:
  * u probabilities (chance agreement between different people) are estimated by random sampling,
  * m probabilities (agreement between records of the same person) are estimated with
    expectation-maximisation (EM), using three different blocking rules so that every
    comparison is trained at least once,
  * the prior match probability is estimated from a deterministic rule and an assumed recall.
"""
import pandas as pd
import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

MODEL_COLUMNS = [
    "unique_id", "source_dataset", "PRI_NOM", "SEG_NOM", "APELL_PAT", "APELL_MAT",
    "PAIS", "FDN", "ANIO_NAC", "RUT", "RUN_TRUE",
]


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------
def make_model_input(persons, discharges, hide_run=False):
    """
    Stack persons and discharges into one table for Splink.

    RUN_TRUE keeps the true RUN only to evaluate the model afterwards; it is never compared.
    hide_run=True also blanks the RUN column, which is what the validation experiment does.
    """
    p = persons.copy()
    e = discharges.copy()
    p["RUN_TRUE"] = p["RUT"]
    e["RUN_TRUE"] = e["_truth_run"] if "_truth_run" in e.columns else e["RUT"]
    if hide_run:
        p["RUT"], e["RUT"] = pd.NA, pd.NA
    p["source_dataset"], e["source_dataset"] = "PERS", "EGR"
    out = pd.concat([p[MODEL_COLUMNS], e[MODEL_COLUMNS]], ignore_index=True)
    for col in ("RUT", "RUN_TRUE"):
        out[col] = out[col].astype("string").str.replace(r"\.0$", "", regex=True)
    return out


# --------------------------------------------------------------------------------------
# Model definition
# --------------------------------------------------------------------------------------
def _name_comparison(col):
    return cl.CustomComparison(
        output_column_name=col,
        comparison_levels=[
            cll.NullLevel(col),
            cll.ExactMatchLevel(col),
            cll.JaroWinklerLevel(col, 0.92),
            cll.JaroWinklerLevel(col, 0.85),
            cll.ElseLevel(),
        ],
    )


def _exact_comparison(col):
    return cl.CustomComparison(
        output_column_name=col,
        comparison_levels=[cll.NullLevel(col), cll.ExactMatchLevel(col), cll.ElseLevel()],
    )


def build_settings():
    birth_year = cl.CustomComparison(
        output_column_name="ANIO_NAC",
        comparison_levels=[
            cll.NullLevel("ANIO_NAC"),
            cll.ExactMatchLevel("ANIO_NAC"),
            cll.CustomLevel("abs(ANIO_NAC_l - ANIO_NAC_r) = 1", label_for_charts="Differ by 1 year"),
            cll.CustomLevel("abs(ANIO_NAC_l - ANIO_NAC_r) = 2", label_for_charts="Differ by 2 years"),
            cll.ElseLevel(),
        ],
    )
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="unique_id",
        source_dataset_column_name="source_dataset",
        blocking_rules_to_generate_predictions=[
            block_on("substr(PRI_NOM, 1, 3)", "substr(APELL_PAT, 1, 4)"),
            block_on("substr(APELL_PAT, 1, 3)", "ANIO_NAC"),
            block_on("substr(PRI_NOM, 1, 4)", "ANIO_NAC"),
            block_on("substr(PRI_NOM, 1, 2)", "substr(APELL_PAT, 1, 1)", "substr(APELL_MAT, 1, 1)"),
            block_on("APELL_PAT", "APELL_MAT"),
        ],
        comparisons=[
            _name_comparison("PRI_NOM"),
            _exact_comparison("SEG_NOM"),
            _name_comparison("APELL_PAT"),
            _name_comparison("APELL_MAT"),
            _exact_comparison("PAIS"),
            birth_year,
        ],
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=False,
        additional_columns_to_retain=["RUT", "RUN_TRUE", "FDN"],
        max_iterations=20,
        em_convergence=0.0001,
    )


# --------------------------------------------------------------------------------------
# Training and prediction
# --------------------------------------------------------------------------------------
def train(model_input, max_pairs=5e6, assumed_recall=0.7, seed=7):
    db_api = DuckDBAPI()
    input_table = db_api.register(model_input)   # Splink 5: inputs must be registered
    linker = Linker(input_table, build_settings())
    linker.training.estimate_probability_two_random_records_match(
        ["l.PRI_NOM = r.PRI_NOM AND l.APELL_PAT = r.APELL_PAT "
         "AND l.APELL_MAT = r.APELL_MAT AND l.ANIO_NAC = r.ANIO_NAC"],
        recall=assumed_recall,
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs, seed=seed)
    for rule in (
        block_on("PRI_NOM", "APELL_PAT"),     # trains SEG_NOM, APELL_MAT, PAIS, ANIO_NAC
        block_on("APELL_PAT", "APELL_MAT"),   # trains PRI_NOM, SEG_NOM, PAIS, ANIO_NAC
        block_on("PRI_NOM", "APELL_MAT"),     # trains APELL_PAT
    ):
        linker.training.estimate_parameters_using_expectation_maximisation(
            rule, estimate_without_term_frequencies=True
        )
    return linker


def predict_pairs(linker, min_probability=0.05):
    pairs = linker.inference.predict(threshold_match_probability=min_probability)
    return pairs.as_pandas_dataframe()


def orient_pairs(pairs):
    """Rename *_l / *_r columns to *_pers / *_egr so the registry side is always explicit."""
    left_is_pers = pairs["source_dataset_l"] == "PERS"
    out = pairs[["match_probability", "match_weight"]].copy()
    bases = [c[:-2] for c in pairs.columns if c.endswith("_l") and c[:-2] + "_r" in pairs.columns]
    for base in bases:
        left, right = pairs[base + "_l"], pairs[base + "_r"]
        out[base + "_pers"] = left.where(left_is_pers, right)
        out[base + "_egr"] = right.where(left_is_pers, left)
    return out


def assign_runs(oriented, threshold=0.85):
    """
    For each discharge, keep the best-scoring person above the threshold and take its RUN.
    n_candidates counts how many persons passed the threshold (more than one = ambiguous).
    """
    above = oriented[oriented["match_probability"] >= threshold]
    n_candidates = above.groupby("unique_id_egr").size().rename("n_candidates")
    best = above.sort_values("match_probability", ascending=False).drop_duplicates("unique_id_egr")
    best = best.merge(n_candidates, on="unique_id_egr", how="left")
    cols = [
        "unique_id_egr", "unique_id_pers", "RUT_pers", "match_probability", "n_candidates",
        "PRI_NOM_egr", "APELL_PAT_egr", "PAIS_egr", "PRI_NOM_pers", "APELL_PAT_pers",
        "RUN_TRUE_pers", "RUN_TRUE_egr",
    ]
    return best[cols].rename(columns={"RUT_pers": "RUT_assigned"}).reset_index(drop=True)
