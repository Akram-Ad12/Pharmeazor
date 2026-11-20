import os, re, sys, unicodedata
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/raw/clean_NOMENCLATURE.VERSION.AOUT_.2026-.xlsx"
SHEETS = {"Nomenclature Aout 2026": "current", "Non Renouvelés ": "non_renewed", "Retraits": "withdrawn"}

def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

# 1. load each sheet from its real header row, drop empty columns
def load_sheet(name, label):
    raw = pd.read_excel(SRC, sheet_name=name, header=None, dtype=object)
    hdr_row = raw.index[raw[0].astype(str).str.strip() == "N°"][0]
    df = raw.iloc[hdr_row + 1:].copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.iloc[hdr_row]]
    df = df.loc[:, ~df.columns.isin(["nan", ""])]          # blank columns
    df = df.dropna(how="all")
    df = df[pd.to_numeric(df["N°"], errors="coerce").notna()]  # keep real data rows only
    df["source_sheet"] = label                                # 3. keep the origin
    return df

xl = pd.ExcelFile(SRC)
names = {s.strip(): s for s in xl.sheet_names}
frames = [load_sheet(names[k.strip()], v) for k, v in SHEETS.items()]
df = pd.concat(frames, ignore_index=True)      # columns missing in one sheet become empty

# 2. strip whitespace, uppercase codes
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].map(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)
for c in ["STATUT", "TYPE"]:
    df[c] = df[c].str.upper()

# 4. normalised DCI
SALTS = r"\b(sodique|sodium|potassique|trihydrate[e]?|monohydrate|dihydrate|anhydre|chlorhydrate|dichlorhydrate|micronisee|lysine|dl)\b"
def norm_dci(s):
    if not isinstance(s, str): return ""
    s = strip_accents(s).lower()
    s = re.sub(r"\bexprime(e)? en [^/+]*", "", s)      # drop "exprimé en ..." (up to next / or +)
    s = re.sub(r"\([^)]*\)", "", s)                    # drop parentheses
    s = re.sub(SALTS, "", s)
    s = re.sub(r"\bde\b|\bd'", " ", s)
    s = s.replace("/", "+")
    parts = [re.sub(r"[^a-z0-9 ]", " ", p) for p in s.split("+")]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return " + ".join(sorted(p for p in parts if p))
df["dci_norm"] = df["DENOMINATION COMMUNE INTERNATIONALE"].map(norm_dci)

# 5. number of substances
df["n_substances"] = df["dci_norm"].map(lambda s: len(s.split(" + ")) if s else 0)

# 6. dosage split (simple cases only; raw kept)
def parse_dosage(s):
    if not isinstance(s, str): return (None, None)
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|µG|UI|MUI|ML|%)\s*(?:/\s*(\d+(?:[.,]\d+)?)?\s*(ML|DOSE|G|L))?\s*", s.upper())
    if not m: return (None, None)
    return (float(m.group(1).replace(",", ".")), m.group(2) + (f"/{m.group(4)}" if m.group(4) else ""))
df[["dosage_value", "dosage_unit"]] = df["DOSAGE"].apply(lambda s: pd.Series(parse_dosage(s)))

# 7. flag duplicates inside each sheet, do not delete
df["dup_flag"] = df.duplicated(["source_sheet", "N°ENREGISTREMENT"], keep=False)

# Output folder for all cleaned files (created if it does not exist yet)
OUT_DIR = "clean_data"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "nomenclature_clean.csv")
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("Saved:", OUT_CSV)
print(df.groupby("source_sheet").size().to_dict(), "| dup rows flagged:", int(df.dup_flag.sum()))
print("dosage parsed:", int(df.dosage_value.notna().sum()), "of", len(df))

# 8. CHECK (read-only) your manual answers in the drug list against the nomenclature.
#    The drug list file is only READ. It is never modified or saved.
DRUG_LIST = sys.argv[2] if len(sys.argv) > 2 else "data/drug_list/Projet2_Liste_40_medicaments.xlsx"
dl = pd.read_excel(DRUG_LIST, sheet_name=0)
COL_NAME = "DCI (nom de la substance)"
COL_E = "Présent dans la nomenclature algérienne ?"

def components(name):
    s = strip_accents(name).lower()
    s = re.sub(r"cotrimoxazole \((.*)\)", r"\1", s)
    s = s.replace("trimethoprime", "trimethoprim")
    return [re.sub(r"^acide ", "", p.strip()).split()[0] for p in s.split("+") if p.strip()]

def row_words(dci_norm):
    return [re.sub(r"^acide ", "", p.strip()).split()[0].replace("trimethoprime", "trimethoprim")
            for p in dci_norm.split(" + ") if p.strip()]

df["_words"] = df["dci_norm"].map(row_words)
print("\nChecking your column E against the nomenclature (differences only):")
n_diff = 0
for _, r in dl.iterrows():
    comp = components(r[COL_NAME])
    single = df[df["_words"].map(lambda w: len(w) == len(comp) and all(any(x.startswith(c[:8]) for x in w) for c in comp))]
    anyform = df[df["_words"].map(lambda w: all(any(x.startswith(c[:8]) for x in w) for c in comp))]  # includes combinations
    n_cur = int((single.source_sheet == "current").sum())
    n_cur_any = int((anyform.source_sheet == "current").sum())
    mine = str(r[COL_E]).strip().upper()
    script = "YES" if n_cur else "NO"
    if mine != script:
        n_diff += 1
        print(f"- {r[COL_NAME]}: you wrote {mine}; single-substance current products = {n_cur}; "
              f"current products incl. combinations = {n_cur_any}; other sheets = {sorted(set(single.source_sheet) - {'current'})}")
print(f"{n_diff} difference(s). Nothing was written to the drug list.")