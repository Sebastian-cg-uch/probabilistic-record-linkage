"""
Preprocessing of the three input tables into one common schema.

Inputs (same schema as the thesis files):
  * persons registry with RUN             -> columns: Nombre, País, Fecha_Nacimiento, RUN, DV
  * hospital discharges with RUN          -> columns: NOMBRES, APELL_PATE, APELL_MATE, GLOSA_PAIS,
                                             NACIMIENTO_FMT_FECHA, RUT, DV
  * hospital discharges without RUN       -> same, without RUT / DV

Flow:
  1. persons: split names, keep foreigners only, normalize country, check RUN, deduplicate
  2. discharges with RUN: records whose RUN is a placeholder (0, 99999999, "***"...) are moved
     to the "without RUN" group, because they cannot be trusted as identifiers
  3. discharges without RUN: original ones + the ones moved in step 2, deduplicated
"""
import pandas as pd

from . import cleaning as cl

FINAL_COLS = [
    "unique_id", "source_dataset", "PRI_NOM", "SEG_NOM", "APELL_PAT", "APELL_MAT",
    "PAIS", "FDN", "ANIO_NAC", "RUT", "DV", "dv_ok",
]
DEDUP_KEY_COLS = ["PRI_NOM", "SEG_NOM", "APELL_PAT", "APELL_MAT", "PAIS", "ANIO_NAC"]
# Only present in synthetic data: the true RUN, used to evaluate the model, never to train it
PASSTHROUGH_COLS = ["_truth_run"]


def _finalize(df, source, id_prefix):
    """Add ids, make sure every expected column exists and keep only those."""
    df = df.copy().reset_index(drop=True)
    df["source_dataset"] = source
    df["unique_id"] = id_prefix + df.index.astype(str)
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    extras = [c for c in PASSTHROUGH_COLS if c in df.columns]
    return df[FINAL_COLS + extras]


def _parse_dates(df):
    df = df.copy()
    df["FDN"] = pd.to_datetime(df["FDN"], errors="coerce")
    df["ANIO_NAC"] = df["FDN"].dt.year
    return df


def _discharge_common(df):
    """Steps shared by both discharge tables."""
    df = df.copy()
    if "NOMBRES" in df.columns:
        df = cl.split_given_names_column(df, "NOMBRES")
    df = df.rename(columns={
        "APELL_PATE": "APELL_PAT", "APELL_MATE": "APELL_MAT", "NACIMIENTO_FMT_FECHA": "FDN",
    })
    df = cl.clean_name_columns(df)
    if "GLOSA_PAIS" in df.columns:
        df = df[df["GLOSA_PAIS"].astype(str).str.upper() != "CHILE"]
        df = cl.normalize_countries(df, "GLOSA_PAIS")
    return _parse_dates(df).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------
def _identity_key(df, cols):
    cols = [c for c in cols if c in df.columns]
    key = df[cols].fillna("").astype(str).apply(
        lambda row: "|".join(v.strip().upper() for v in row if v.strip()), axis=1
    )
    return key.replace("", pd.NA)


def deduplicate(df, strategy="first"):
    """
    Remove rows with an identical identity key (names + country + birth year).
    strategy="rut" keeps, for each key, the row that has a RUN.
    Returns (deduplicated table, removed rows).
    """
    df = df.copy()
    df["_key"] = _identity_key(df, DEDUP_KEY_COLS)
    valid = df["_key"].notna()
    is_dup = pd.Series(False, index=df.index)
    is_dup[valid] = df.loc[valid, "_key"].duplicated(keep=False)
    if strategy == "rut":
        df["_has_run"] = df["RUT"].notna()
        df = df.sort_values(["_key", "_has_run"], ascending=[True, False])
    keep = "last" if strategy == "last" else "first"
    kept_valid = df[df["_key"].notna()].drop_duplicates(subset="_key", keep=keep)
    kept = pd.concat([kept_valid, df[df["_key"].isna()]]).sort_index()
    removed = df.loc[is_dup & ~df.index.isin(kept.index)]
    drop_cols = [c for c in ["_key", "_has_run"] if c in kept.columns]
    return kept.drop(columns=drop_cols).reset_index(drop=True), removed.drop(columns=drop_cols)


def find_multiple_runs(df):
    """Data-quality report: identities (same name, country, birth year) with more than one RUN."""
    with_run = df[df["RUT"].notna()].copy()
    with_run["_key"] = _identity_key(with_run, DEDUP_KEY_COLS)
    runs_per_key = with_run.dropna(subset=["_key"]).groupby("_key")["RUT"].nunique()
    multiple = runs_per_key[runs_per_key > 1].index
    return with_run[with_run["_key"].isin(multiple)].sort_values(["_key", "RUT"])


# --------------------------------------------------------------------------------------
# The three tables
# --------------------------------------------------------------------------------------
def process_persons(raw):
    df = raw.copy()
    if "Nombre" in df.columns and "PRI_NOM" not in df.columns:
        df = cl.split_persons_name_column(df, "Nombre")
    df = cl.clean_name_columns(df)
    if "País" in df.columns:
        country = df["País"].astype(str).str.upper().str.strip()
        foreign = ~(country.isin(cl.NOT_FOREIGN_OR_UNKNOWN) | country.str.startswith("REGION"))
        df = cl.normalize_countries(df[foreign], "País")
    df = df.rename(columns={"Fecha_Nacimiento": "FDN", "RUN": "RUT"})
    df = cl.verify_runs(df, "RUT", "DV")
    df = _parse_dates(df)
    return _finalize(df, "PERS", "PERS_")


def process_discharges_with_run(raw):
    """Returns (discharges with a usable RUN, discharges whose RUN is a placeholder)."""
    df = _discharge_common(raw)
    problematic = df["RUT"].apply(cl.is_problematic_run)
    with_run = df[~problematic].reset_index(drop=True)
    moved = df[problematic].reset_index(drop=True)
    with_run = cl.verify_runs(with_run, "RUT", "DV")
    moved["RUT"] = pd.NA
    moved["DV"] = pd.NA
    return (
        _finalize(with_run, "EGR_CON", "EGR_CON_"),
        _finalize(moved, "EGR_SIN_FROM_CON", "EGR_SIN_FROM_CON_"),
    )


def process_discharges_without_run(raw, moved_from_with_run=None):
    df = _discharge_common(raw)
    df["RUT"], df["DV"], df["dv_ok"] = pd.NA, pd.NA, pd.NA
    df = _finalize(df, "EGR_SIN", "EGR_SIN_")
    if moved_from_with_run is not None and len(moved_from_with_run) > 0:
        df = pd.concat([df, moved_from_with_run], ignore_index=True)
    return df


def process_all(persons_raw, discharges_with_run_raw, discharges_without_run_raw, verbose=True):
    """Run the whole preprocessing. Returns a dict with the three clean tables and QA info."""
    persons = process_persons(persons_raw)
    persons, dup_persons = deduplicate(persons, strategy="rut")

    with_run, moved = process_discharges_with_run(discharges_with_run_raw)
    multiple_runs = find_multiple_runs(with_run)

    without_run = process_discharges_without_run(discharges_without_run_raw, moved)
    without_run, dup_without = deduplicate(without_run, strategy="first")

    out = {
        "persons": persons,
        "discharges_with_run": with_run,
        "discharges_without_run": without_run,
        "qa": {
            "duplicate_persons_removed": len(dup_persons),
            "duplicate_discharges_removed": len(dup_without),
            "discharges_moved_to_without_run": len(moved),
            "identities_with_multiple_runs": multiple_runs["_key"].nunique() if len(multiple_runs) else 0,
        },
    }
    if verbose:
        print("=== Preprocessing summary ===")
        print(f"  persons (foreign, deduplicated):      {len(persons):>7,}")
        print(f"  discharges with usable RUN:           {len(with_run):>7,}")
        print(f"  discharges without RUN (total):       {len(without_run):>7,}")
        for k, v in out["qa"].items():
            print(f"  {k}: {v:,}")
    return out
