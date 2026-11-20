from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SRC = BASE_DIR / "data" / "raw" / "clean_NOMENCLATURE.VERSION.AOUT_.2026-.xlsx"
DEFAULT_DRUG_LIST = BASE_DIR / "data" / "drug_list" / "Projet2_Liste_40_medicaments.xlsx"
OUT_DIR = BASE_DIR / "clean_data"

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
DRUG_LIST = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DRUG_LIST

SHEETS = {
    "Nomenclature Aout 2026": "current",
    "Non Renouvelés": "non_renewed",
    "Retraits": "withdrawn",
}

SALTS = (
    r"\b("
    r"sodique|sodium|potassique|calcique|magnesique|"
    r"trihydrate[e]?|monohydrate[e]?|dihydrate[e]?|hemihydrate[e]?|anhydre|"
    r"chlorhydrate|dichlorhydrate|sulfate|phosphate|acetate|maleate|tartrate|"
    r"besilate|mesilate|fumarate|citrate|nitrate|lactate|gluconate|valerate|"
    r"axetil|proxetil|pivoxil|etexilate|medoxomil|dipropionate|propionate|"
    r"micronisee?|lysine|dl|d|l"
    r")\b"
)


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def normalize_substance_token(token: str) -> str:
    """Canonicalizes a single active substance name for reliable matching."""
    t = strip_accents(token).lower().strip()
    t = re.sub(r"\bexprime(e)? en [^/+]*", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(SALTS, " ", t)
    t = re.sub(r"\bde\b|\bd'", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Standardize known French/English spelling differences in the 40-drug list
    aliases = {
        "trimethoprime": "trimethoprim",
        "sulfamethoxazole": "sulfamethoxazole",
        "acetylsalicylique": "acide acetylsalicylique",
        "clavulanique": "acide clavulanique",
    }
    return aliases.get(t, t)


def norm_dci(s: object) -> str:
    """Normalizes single or combined DCI strings into sorted canonical '+' components."""
    if not isinstance(s, str) or not s.strip():
        return ""
    text = strip_accents(s).lower()
    # Expand "cotrimoxazole (sulfamethoxazole + trimethoprime)" -> "sulfamethoxazole + trimethoprime"
    text = re.sub(r"cotrimoxazole\s*\(([^)]+)\)", r"\1", text)
    text = text.replace("/", "+")
    parts = [normalize_substance_token(p) for p in text.split("+")]
    return " + ".join(sorted(p for p in parts if p))


def parse_dosage(s: object) -> tuple[float | None, str | None]:
    if not isinstance(s, str):
        return (None, None)
    m = re.fullmatch(
        r"\s*(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|µG|UI|MUI|ML|%)\s*(?:/\s*(\d+(?:[.,]\d+)?)?\s*(ML|DOSE|G|L))?\s*",
        s.upper(),
    )
    if not m:
        return (None, None)
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2) + (f"/{m.group(4)}" if m.group(4) else "")
    return (val, unit)


# 1. Load each sheet from its real header row and drop blank columns/rows
def load_sheet(xl_path: Path, sheet_name: str, label: str) -> pd.DataFrame:
    raw = pd.read_excel(xl_path, sheet_name=sheet_name, header=None, dtype=object)
    hdr_row = raw.index[raw[0].astype(str).str.strip() == "N°"][0]
    df = raw.iloc[hdr_row + 1 :].copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.iloc[hdr_row]]
    df = df.loc[:, ~df.columns.isin(["nan", ""])]
    df = df.dropna(how="all")
    df = df[pd.to_numeric(df["N°"], errors="coerce").notna()]
    df["source_sheet"] = label
    return df


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Nomenclature Excel file not found: {SRC.resolve()}")
    if not DRUG_LIST.exists():
        raise FileNotFoundError(f"40-drug Excel file not found: {DRUG_LIST.resolve()}")

    xl = pd.ExcelFile(SRC)
    sheet_lookup = {s.strip(): s for s in xl.sheet_names}
    frames = [load_sheet(SRC, sheet_lookup[k.strip()], v) for k, v in SHEETS.items()]
    df = pd.concat(frames, ignore_index=True)

    # 2. Strip whitespace and standardize status/type casing
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
            )
    for col in ["STATUT", "TYPE"]:
        if col in df.columns:
            df[col] = df[col].str.upper()

    # 3. Normalize DCI and count active substances
    df["dci_norm"] = df["DENOMINATION COMMUNE INTERNATIONALE"].map(norm_dci)
    df["n_substances"] = df["dci_norm"].map(lambda s: len(s.split(" + ")) if s else 0)

    # 4. Parse simple dosages (keep original DOSAGE column intact)
    df[["dosage_value", "dosage_unit"]] = df["DOSAGE"].apply(lambda s: pd.Series(parse_dosage(s)))

    # 5. Flag duplicates within each sheet without deleting
    df["dup_flag"] = df.duplicated(["source_sheet", "N°ENREGISTREMENT"], keep=False)

    # 6. Load the 40-drug target list (read-only) and build exact normalized DCI lookup
    dl = pd.read_excel(DRUG_LIST, sheet_name="Médicaments")
    dl["dci_norm"] = dl["DCI (nom de la substance)"].map(norm_dci)

    target_map: dict[str, dict[str, object]] = {}
    for _, row in dl.iterrows():
        key = str(row["dci_norm"])
        target_map[key] = {
            "drug_num": int(row["N°"]),
            "target_dci": str(row["DCI (nom de la substance)"]).strip(),
            "therapeutic_class": str(row["Classe"]).strip(),
            "atc_indicative": str(row["Code ATC (indicatif)"]).strip(),
            "manual_present_flag": str(row["Présent dans la nomenclature algérienne ?"]).strip().upper(),
        }

    # 7. Filter nomenclature to KEEP ONLY rows matching one of the 40 target drugs
    df_40 = df[df["dci_norm"].isin(target_map)].copy()
    df_40["drug_num"] = df_40["dci_norm"].map(lambda k: target_map[k]["drug_num"])
    df_40["target_dci"] = df_40["dci_norm"].map(lambda k: target_map[k]["target_dci"])
    df_40["therapeutic_class"] = df_40["dci_norm"].map(lambda k: target_map[k]["therapeutic_class"])
    df_40["atc_indicative"] = df_40["dci_norm"].map(lambda k: target_map[k]["atc_indicative"])

    # Sort cleanly by drug number (1..40), sheet status, and trade name
    df_40 = df_40.sort_values(
        by=["drug_num", "source_sheet", "NOM DE MARQUE"],
        na_position="last",
    ).reset_index(drop=True)

    # 8. Save filtered & cleaned dataset
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = OUT_DIR / "nomenclature_40_drugs_clean.csv"
    df_40.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Total rows in raw nomenclature: {len(df)}")
    print(f"Rows kept for the 40 target drugs: {len(df_40)}")
    print(f"Saved cleaned 40-drug dataset to: {out_csv.resolve()}")
    print(
        "Breakdown by source_sheet:",
        df_40.groupby("source_sheet").size().to_dict(),
        "| dup rows flagged:",
        int(df_40["dup_flag"].sum()),
    )
    print(
        "Dosage parsed:",
        int(df_40["dosage_value"].notna().sum()),
        "of",
        len(df_40),
    )

    # 9. Audit summary per drug (1..40) vs manual column E (read-only)
    print("\nSummary check against the 40-drug list:")
    n_diff = 0
    for key, info in target_map.items():
        sub = df_40[df_40["dci_norm"] == key]
        n_cur = int((sub["source_sheet"] == "current").sum())
        other_sheets = sorted(set(sub["source_sheet"]) - {"current"})
        expected_flag = "YES" if n_cur > 0 else "NO"
        if info["manual_present_flag"] != expected_flag:
            n_diff += 1
            print(
                f" - #{info['drug_num']:02d} {info['target_dci']}: "
                f"manual={info['manual_present_flag']}, current_rows={n_cur}, other_sheets={other_sheets}"
            )
    print(f"Audit complete: {n_diff} difference(s) found. Original drug list file was not modified.")


if __name__ == "__main__":
    main()