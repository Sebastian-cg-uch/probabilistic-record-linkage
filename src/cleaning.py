"""
Field-level cleaning functions for administrative health records.

Handles the problems that matter for Chilean and migrant names:
  * full names stored in one field (split into first/second name and two surnames,
    keeping surname particles such as DE, DEL, DE LA together),
  * RUN check digit (dígito verificador) validation,
  * country names written in many ways,
  * placeholder or corrupted values in name fields ("***", "N.N", "A.", "2145"...).

The original thesis code removed junk values with long hard-coded lists. Here the same
goal is achieved with rules, so that no data-specific strings live in the code.
"""
import re

import pandas as pd
from unidecode import unidecode

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
NULL_TOKENS = {"NAN", "NONE", "NN", "N", "NA", "<NA>", ""}

# Words that belong to a surname ("DE LA CRUZ", "VAN DER BERG", ...)
PARTICLES = {
    "DE", "DA", "DO", "DOS", "DAS", "DEL", "LA", "LAS", "EL", "VAN", "VON", "DI", "LE",
    "AL", "BIN", "IBN", "MAC", "MC", "O", "DELLA", "DELLE", "DELLO", "DEGLI",
}

# Obvious placeholders (no digits, no punctuation): everything else is handled by rules.
NAME_PLACEHOLDERS = {
    "NN", "NO", "SIN", "SIN DATO", "SIN INFORMACION", "DESCONOCIDO", "NO INFORMADO",
    "OMITIDO", "XX", "XXX", "XXXX", "NOMBRE", "APELLIDO",
}

COUNTRY_REPLACEMENTS = {
    "VENEZUELA BOLIVARIANA": "VENEZUELA", "VENEZUELA REPUBLIC": "VENEZUELA",
    "VENEZUEL": "VENEZUELA", "VENEZUELAN": "VENEZUELA",
    "ESTADOS UNIDOS DE AMERICA": "ESTADOS UNIDOS", "ESTADOS UNIDOS AMERICA": "ESTADOS UNIDOS",
    "USA": "ESTADOS UNIDOS", "UNITED STATES": "ESTADOS UNIDOS", "EE.UU.": "ESTADOS UNIDOS",
    "REPUBLICA DOMINICANA": "DOMINICANA", "REP DOMINICANA": "DOMINICANA",
    "R DOMINICANA": "DOMINICANA",
    "COL": "COLOMBIA", "COLO": "COLOMBIA", "COLOMBIAN": "COLOMBIA",
    "PERU REPUBLIC": "PERU", "PERUVIAN": "PERU",
    "ESPA@A": "ESPANA", "SPANISH": "ESPANA",
    "BRAZIL": "BRASIL", "BRAZILIAN": "BRASIL",
    "ECUADORIAN": "ECUADOR",
    "BOLIVIAN": "BOLIVIA",
    "CHINE": "CHINA", "CHINESE": "CHINA",
    "HAITI REPUBLIC": "HAITI", "HAITIAN": "HAITI",
}

# Values of the country field that mean "not a foreign national" or "unknown"
NOT_FOREIGN_OR_UNKNOWN = {
    "CHILE", "NACIONALIDAD OMITIDA", "DESCONOCIDO", "SIN INFORMACION", "NO INFORMADO",
    "ASIA", "CARIBE",
}


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def to_na(value):
    """Return pd.NA for null-like tokens, otherwise the stripped string."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    return pd.NA if text.upper() in NULL_TOKENS else text


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
def split_person_name(full_name):
    """
    Split a single-field name ("JUAN CARLOS DE LA CRUZ PEREZ") into
    (first name, second name, paternal surname, maternal surname).
    """
    name = to_na(full_name)
    if pd.isna(name):
        return (None, None, None, None)
    name = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name.upper())).strip()
    words = name.split()
    n = len(words)
    if n == 0:
        return (None, None, None, None)
    if n == 1:
        return (words[0], None, None, None)
    if n == 2:
        return (words[0], None, words[1], None)
    if n == 3:
        if words[1] not in PARTICLES:
            return (words[0], None, words[1], words[2])
        return (words[0], None, " ".join(words[1:]), None)
    if n == 4:
        if words[1] in PARTICLES:
            return (words[0], None, " ".join(words[1:3]), words[3])
        if words[2] in PARTICLES:
            return (words[0], words[1], " ".join(words[2:]), None)
        return (words[0], words[1], words[2], words[3])
    # 5 or more words: look for the first particle to find where the surnames start
    start = next((i for i, w in enumerate(words[1:], 1) if w in PARTICLES), None)
    if start:
        given, surnames = words[:start], words[start:]
        half = len(surnames) // 2
        return (
            given[0],
            " ".join(given[1:]) if len(given) > 1 else None,
            " ".join(surnames[:half]),
            " ".join(surnames[half:]) if len(surnames) > 1 else None,
        )
    return (words[0], " ".join(words[1:-2]), words[-2], words[-1])


def split_persons_name_column(df, column):
    """Replace a full-name column by PRI_NOM, SEG_NOM, APELL_PAT, APELL_MAT."""
    parts = df[column].apply(split_person_name).apply(pd.Series)
    parts.columns = ["PRI_NOM", "SEG_NOM", "APELL_PAT", "APELL_MAT"]
    return pd.concat([df.drop(columns=[column]), parts], axis=1)


def split_given_names_column(df, column):
    """Discharge files store given names together: first word = first name, rest = second."""
    def _split(value):
        name = to_na(value)
        if pd.isna(name):
            return (None, None)
        words = name.split()
        return (words[0], " ".join(words[1:]) if len(words) > 1 else None)

    parts = df[column].apply(_split).apply(pd.Series)
    parts.columns = ["PRI_NOM", "SEG_NOM"]
    return pd.concat([df.drop(columns=[column]), parts], axis=1)


def clean_name_value(value):
    """
    Validate one name or surname. Returns the upper-cased value or pd.NA if it looks like
    junk: placeholders, digits, punctuation, dashes at the edges or only one letter.
    Compound names are allowed when every part (split by space or hyphen) has 2+ letters.
    """
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if text in NAME_PLACEHOLDERS or text == "":
        return pd.NA
    if re.search(r"\d", text):
        return pd.NA
    if re.search(r"[?*+=#@.,_()\\/\[\]{}|~`!^:;\"']", text):
        return pd.NA
    if "--" in text or text.startswith("-") or text.endswith("-"):
        return pd.NA
    parts = [p for p in re.split(r"[-\s]+", text) if p]
    if parts and all(p.isalpha() and len(p) >= 2 for p in parts):
        return text
    return pd.NA


def clean_name_columns(df, columns=("PRI_NOM", "SEG_NOM", "APELL_PAT", "APELL_MAT")):
    """Apply clean_name_value to name columns and drop rows where all of them are empty."""
    df = df.copy()
    present = [c for c in columns if c in df.columns]
    for col in present:
        df[col] = df[col].apply(clean_name_value)
    return df[df[present].notna().any(axis=1)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# RUN (Chilean national ID) check digit
# --------------------------------------------------------------------------------------
def run_check_digit(run):
    """Compute the check digit (módulo 11) of a RUN. Returns None if it cannot be computed."""
    if pd.isna(run) or str(run).strip() == "":
        return None
    try:
        digits = [int(d) for d in reversed(str(int(float(run))))]
    except (ValueError, TypeError):
        return None
    factors = [2, 3, 4, 5, 6, 7] * (len(digits) // 6 + 1)
    remainder = 11 - (sum(d * f for d, f in zip(digits, factors)) % 11)
    return "0" if remainder == 11 else "K" if remainder == 10 else str(remainder)


def verify_runs(df, run_col="RUT", dv_col="DV"):
    """Add DV_CAL (computed check digit) and dv_ok (True when it matches the reported one)."""
    df = df.copy()
    df["DV_CAL"] = df[run_col].apply(run_check_digit)
    df[dv_col] = df[dv_col].astype(str).str.upper().replace(["NAN", "NONE", "", "NN"], pd.NA)
    df["dv_ok"] = df["DV_CAL"].notna() & df[dv_col].notna() & (df[dv_col] == df["DV_CAL"])
    return df


def is_problematic_run(run):
    """True for missing RUNs, placeholders (0, 99999999, '***') and implausibly low numbers."""
    if pd.isna(run):
        return True
    text = str(run).strip()
    if text in {"0", "99999999", "0.0"} or re.match(r"^[*\-\.,_\sXx]+$", text):
        return True
    try:
        return float(text.replace(".", "").replace(",", "")) < 1_000_000
    except ValueError:
        return False


# --------------------------------------------------------------------------------------
# Countries
# --------------------------------------------------------------------------------------
def normalize_countries(df, column):
    """Upper-case, remove accents, unify spelling variants and rename the column to PAIS."""
    df = df.copy()
    values = df[column].apply(to_na)
    valid = values.notna()
    values[valid] = values[valid].apply(unidecode).str.upper().str.strip()
    values[valid] = (
        values[valid].replace(COUNTRY_REPLACEMENTS)
        .str.replace(r"\s*\([^)]+\)", "", regex=True).str.strip()
    )
    df[column] = values
    return df.rename(columns={column: "PAIS"})
