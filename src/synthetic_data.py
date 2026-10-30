"""
Generate SYNTHETIC versions of the three input tables, with the same schema as the real
(confidential) files used in the thesis:

  data/synthetic/dat_pers.csv          persons registry with RUN
  data/synthetic/dat_egr.csv           hospital discharges with RUN (some RUNs are placeholders)
  data/synthetic/dat_egr_sin_rut.csv   hospital discharges without RUN

Everything is invented. RUN-like numbers are random and carry a valid check digit, but they
do not belong to real people. The column `_truth_run` (true RUN of each discharge) exists only
because the data are synthetic; it is never used to train the model, only to evaluate it.

Typical administrative-data problems are injected: typos, missing second names or surnames,
swapped day/month, accents and country spelling variants, placeholder values ("***", "N.N"),
placeholder RUNs (0, 99999999), non-Chilean/Chilean mixes and repeated rows.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import run_check_digit

SEED = 42
N_UNIVERSE = 7000            # people who exist in the synthetic world
SHARE_IN_REGISTRY = 0.80     # share of them that appear in the persons registry
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

GIVEN_NAMES = [
    "MARIA", "JOSE", "JUAN", "LUIS", "CARLOS", "ANA", "ROSA", "PEDRO", "MIGUEL", "CARMEN",
    "JORGE", "SOFIA", "DANIEL", "VALENTINA", "ANDRES", "CAMILA", "DIEGO", "PATRICIA",
    "FERNANDO", "GABRIELA", "RICARDO", "DANIELA", "ALEJANDRO", "VERONICA", "MANUEL",
    "CLAUDIA", "RAFAEL", "YENIFER", "WILMER", "YULEIDY", "JHON", "KIMBERLY", "DARWIN",
    "MAYERLIN", "EDWIN", "LEIDY", "JEAN", "MARIE", "WILSON", "NADEGE", "ROOSEVELT", "STEPHANIE",
]
SURNAMES = [
    "GONZALEZ", "RODRIGUEZ", "PEREZ", "GARCIA", "MARTINEZ", "LOPEZ", "SANCHEZ", "RAMIREZ",
    "TORRES", "FLORES", "RIVERA", "GOMEZ", "DIAZ", "REYES", "MORALES", "CRUZ", "ORTIZ",
    "GUTIERREZ", "CHAVEZ", "RAMOS", "MENDOZA", "CASTILLO", "VARGAS", "ROJAS", "HERRERA",
    "MEDINA", "AGUILAR", "SALAZAR", "PIERRE", "JOSEPH", "JEAN", "MEJIA", "QUISPE", "MAMANI",
    "CONDORI", "HERNANDEZ", "SUAREZ", "BAPTISTE", "CHARLES",
]
# Compound surnames are rarer; they matter because the persons registry stores the full name in
# a single field, so the splitting heuristic can misplace the words of a compound surname.
COMPOUND_SURNAMES = ["DE LA CRUZ", "DEL VALLE", "DE LEON", "DE LOS SANTOS"]
P_COMPOUND = 0.03
COUNTRIES = {
    "VENEZUELA": 0.36, "PERU": 0.17, "HAITI": 0.15, "COLOMBIA": 0.12, "BOLIVIA": 0.07,
    "ECUADOR": 0.04, "ARGENTINA": 0.03, "REPUBLICA DOMINICANA": 0.04, "ESTADOS UNIDOS": 0.02,
}
COUNTRY_VARIANTS = {
    "VENEZUELA": ["VENEZUELA BOLIVARIANA", "VENEZUELAN"], "PERU": ["PERÚ", "PERUVIAN"],
    "HAITI": ["HAITÍ", "HAITIAN"], "COLOMBIA": ["COLOMBIAN"], "BOLIVIA": ["BOLIVIAN"],
    "ECUADOR": ["ECUADORIAN"], "REPUBLICA DOMINICANA": ["REP DOMINICANA", "REPÚBLICA DOMINICANA"],
    "ESTADOS UNIDOS": ["USA", "UNITED STATES"],
}
JUNK_NAMES = ["***", "N.N", "XXX", "-", "A.", "SIN DATO", "2145"]
PLACEHOLDER_RUNS = [0, 99999999, ".", "*", 123456]


# --------------------------------------------------------------------------------------
# Noise helpers
# --------------------------------------------------------------------------------------
def typo(text, rng):
    """Single-character error: delete, swap or double a letter."""
    if not isinstance(text, str) or len(text) < 4:
        return text
    i = rng.randrange(1, len(text) - 1)
    kind = rng.choice(["delete", "swap", "double"])
    if kind == "delete":
        return text[:i] + text[i + 1:]
    if kind == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text[:i] + text[i] + text[i:]


def noisy_identity(person, rng, noise):
    """Return the identity fields of a person as they could appear in one record."""
    rec = {k: person[k] for k in ("first", "second", "pat", "mat", "country", "birth")}
    for key in ("first", "pat", "mat"):
        if rng.random() < noise:
            rec[key] = typo(rec[key], rng)
    if rec["second"] and rng.random() < 0.25:
        rec["second"] = None                       # second name often missing
    if rng.random() < 0.05:
        rec["mat"] = None                          # maternal surname missing
    if rng.random() < noise / 3:
        rec["pat"], rec["mat"] = rec["mat"], rec["pat"]   # surnames swapped
    for key in ("first", "pat"):
        if rng.random() < 0.008:
            rec[key] = rng.choice(JUNK_NAMES)
    if rng.random() < noise / 3 and rec["birth"] is not None and rec["birth"].day <= 12:
        rec["birth"] = date(rec["birth"].year, rec["birth"].day, rec["birth"].month)
    if rng.random() < 0.03:
        rec["birth"] = None
    if rng.random() < 0.06 and rec["country"] in COUNTRY_VARIANTS:
        rec["country"] = rng.choice(COUNTRY_VARIANTS[rec["country"]])
    return rec


# --------------------------------------------------------------------------------------
# World and tables
# --------------------------------------------------------------------------------------
def pick_surname(rng):
    return rng.choice(COMPOUND_SURNAMES) if rng.random() < P_COMPOUND else rng.choice(SURNAMES)


def build_universe(rng):
    countries, weights = zip(*COUNTRIES.items())
    start = date(1935, 1, 1)
    span = (date(2023, 12, 31) - start).days
    used_runs, people = set(), []
    for pid in range(N_UNIVERSE):
        run = rng.randrange(70_000_000, 79_999_999)
        while run in used_runs:
            run = rng.randrange(70_000_000, 79_999_999)
        used_runs.add(run)
        people.append({
            "pid": pid,
            "first": rng.choice(GIVEN_NAMES),
            "second": rng.choice(GIVEN_NAMES) if rng.random() < 0.5 else None,
            "pat": pick_surname(rng),
            "mat": pick_surname(rng),
            "country": rng.choices(countries, weights)[0],
            "birth": start + timedelta(days=rng.randrange(span)),
            "run": run,
            "dv": run_check_digit(run),
        })
    return people


def _fmt_date(d):
    return None if d is None else d.isoformat()


def build_persons_table(people_in_registry, rng):
    rows = []
    for p in people_in_registry:
        rec = noisy_identity(p, rng, noise=0.06)
        name = " ".join(x for x in [rec["first"], rec["second"], rec["pat"], rec["mat"]] if x)
        if rng.random() < 0.10:
            name = name.lower()
        rows.append({
            "Nombre": name, "País": rec["country"], "Fecha_Nacimiento": _fmt_date(rec["birth"]),
            "RUN": p["run"], "DV": p["dv"],
        })
    # exact duplicated rows and Chilean nationals (the pipeline must drop them)
    rows += rng.sample(rows, int(0.015 * len(rows)))
    for _ in range(int(0.03 * len(rows))):
        rows.append({"Nombre": f"{rng.choice(GIVEN_NAMES)} {rng.choice(SURNAMES)} {rng.choice(SURNAMES)}",
                     "País": "CHILE", "Fecha_Nacimiento": "1980-01-01",
                     "RUN": rng.randrange(8_000_000, 20_000_000), "DV": "0"})
    return pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_discharges(people, rng, noise, with_run, n_persons, max_episodes=3):
    rows = []
    for p in rng.sample(people, n_persons):
        for _ in range(rng.choices(range(1, max_episodes + 1), weights=[0.55, 0.3, 0.15])[0]):
            rec = noisy_identity(p, rng, noise)
            row = {
                "NOMBRES": " ".join(x for x in [rec["first"], rec["second"]] if x),
                "APELL_PATE": rec["pat"], "APELL_MATE": rec["mat"],
                "GLOSA_PAIS": rec["country"], "NACIMIENTO_FMT_FECHA": _fmt_date(rec["birth"]),
                "_truth_run": p["run"],
            }
            if with_run:
                bad = rng.random() < 0.06
                row["RUT"] = rng.choice(PLACEHOLDER_RUNS) if bad else p["run"]
                row["DV"] = None if bad else p["dv"]
            rows.append(row)
    if with_run:
        rows += [{"NOMBRES": "PEDRO", "APELL_PATE": "SOTO", "APELL_MATE": "DIAZ", "GLOSA_PAIS": "CHILE",
                  "NACIMIENTO_FMT_FECHA": "1975-05-05", "RUT": 12345678, "DV": "5", "_truth_run": 12345678}] * 30
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    cols = ["NOMBRES", "APELL_PATE", "APELL_MATE", "GLOSA_PAIS", "NACIMIENTO_FMT_FECHA"]
    cols += (["RUT", "DV"] if with_run else []) + ["_truth_run"]
    return df[cols]


def generate(out_dir=OUT_DIR):
    rng = random.Random(SEED)
    np.random.seed(SEED)
    people = build_universe(rng)
    in_registry = [p for p in people if rng.random() < SHARE_IN_REGISTRY]
    persons = build_persons_table(in_registry, rng)
    with_run = build_discharges(people, rng, noise=0.08, with_run=True, n_persons=4500)
    without_run = build_discharges(people, rng, noise=0.12, with_run=False, n_persons=2500)
    out_dir.mkdir(parents=True, exist_ok=True)
    persons.to_csv(out_dir / "dat_pers.csv", index=False)
    with_run.to_csv(out_dir / "dat_egr.csv", index=False)
    without_run.to_csv(out_dir / "dat_egr_sin_rut.csv", index=False)
    print(f"Synthetic data written to {out_dir}")
    print(f"  persons: {len(persons):,} | discharges with RUN: {len(with_run):,} | "
          f"discharges without RUN: {len(without_run):,}")


if __name__ == "__main__":
    generate()
