import hashlib
import json
import re
import unicodedata
from pathlib import Path

import fitz  # type: ignore # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup  # type: ignore

RAW_ROOT = Path("data/raw")
CLEAN_ROOT = Path("clean_data")
MANIFEST_PATH = CLEAN_ROOT / ".clean_manifest.json"

EXCEL_CANDIDATES = [
    Path("data/drug_list/Projet2_Liste_40_medicaments.xlsx"),
    Path("data/drug_list/Projet2_Liste_40_medicaments_2.xlsx"),
    Path("data/raw/Projet2_Liste_40_medicaments.xlsx"),
    Path("Projet2_Liste_40_medicaments.xlsx"),
    Path("Projet2_Liste_40_medicaments_2.xlsx"),
]

SUPPORTED_EXTENSIONS = {".html", ".htm", ".pdf", ".xlsx", ".xls", ".json"}

# Excluded content per specification: prices, stock, ads, supplements, medical devices, forums
EXCLUDED_PATTERNS = (
    "prix",
    "ppa",
    "rupture de stock",
    "en stock",
    "disponibilite",
    "publicite",
    "sponsorise",
    "complement alimentaire",
    "dispositif medical",
    "forum",
    "blog",
)

# Target categories required by the project specification
TARGET_CATEGORY_KEYWORDS = {
    "INDICATIONS": (
        "indication",
        "indications therapeutiques",
        "dans quel cas",
        "utilise dans",
        "traitement de",
        "soulagement de",
    ),
    "CONTRAINDICATIONS": (
        "contre-indication",
        "contre indication",
        "contre-indications",
        "ne jamais prendre",
        "ne jamais utiliser",
        "allergie a la substance",
        "hypersensibilite",
    ),
    "DOSAGE": (
        "posologie",
        "dosage",
        "mode d'administration",
        "comment prendre",
        "dose maximale",
        "dose journaliere",
        "par prise",
        "surdosage",
        "mg/kg",
        "comprime",
        "gelule",
    ),
    "INTERACTIONS": (
        "interaction",
        "interactions medicamenteuses",
        "autres medicaments",
        "association deconseillee",
        "precaution d'emploi",
        "a prendre en compte",
        "thesaurus",
    ),
    "EFFECTS": (
        "effet indesirable",
        "effets indesirables",
        "effet secondaire",
        "effets secondaires",
        "mises en garde",
        "precautions d'emploi",
        "grossesse",
        "allaitement",
        "conduite de vehicules",
    ),
}

ALL_CATEGORY_TERMS = tuple(
    term for terms in TARGET_CATEGORY_KEYWORDS.values() for term in terms
)


def normalize_text(text: str) -> str:
    """Lowercase, remove diacritics/accents, and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def find_drug_list_excel() -> Path | None:
    for candidate in EXCEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_40_drug_terms() -> set[str]:
    """Load DCIs, active components, ATC codes, and noted names from the 40-drug Excel list."""
    terms: set[str] = set()
    excel_path = find_drug_list_excel()
    if excel_path is None:
        print("[WARN] 40-drug Excel file not found; using category keywords only.")
        return terms

    df = pd.read_excel(excel_path, sheet_name="Médicaments")
    for _, row in df.iterrows():
        dci_norm = normalize_text(str(row.get("DCI (nom de la substance)", "") or ""))
        atc_norm = normalize_text(str(row.get("Code ATC (indicatif)", "") or ""))
        rem_norm = normalize_text(str(row.get("Remarques", "") or ""))

        if dci_norm and dci_norm != "nan":
            terms.add(dci_norm)
            cleaned = re.sub(r"[()]", " ", dci_norm)
            for part in re.split(r"\s*\+\s*|\s+et\s+|\s*/\s*", cleaned):
                part = part.strip()
                if len(part) >= 4:
                    terms.add(part)

        if atc_norm and atc_norm != "nan" and len(atc_norm) >= 5:
            terms.add(atc_norm)

        if rem_norm and rem_norm != "nan":
            for token in re.findall(r"[a-z]{5,}", rem_norm):
                if token not in {
                    "only",
                    "entries",
                    "compared",
                    "others",
                    "imported",
                    "brands",
                    "mylan",
                    "chlorhydrate",
                }:
                    terms.add(token)

    return terms


def is_excluded_line(line: str) -> bool:
    norm = normalize_text(line)
    return any(normalize_text(pat) in norm for pat in EXCLUDED_PATTERNS)


def detect_category(line: str) -> str | None:
    norm = normalize_text(line)
    for cat, kws in TARGET_CATEGORY_KEYWORDS.items():
        if any(normalize_text(kw) in norm for kw in kws):
            return cat
    return None


def extract_html_blocks(html_path: Path) -> list[str]:
    raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()

    container = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[str] = []

    for elem in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if elem.name == "tr":
            cells = [
                re.sub(r"\s+", " ", c.get_text(" ", strip=True)).strip()
                for c in elem.find_all(["th", "td"])
            ]
            cells = [c for c in cells if c]
            line = " | ".join(cells)
        else:
            line = re.sub(r"\s+", " ", elem.get_text(" ", strip=True)).strip()

        if not line or len(line) < 2 or is_excluded_line(line):
            continue
        if not blocks or blocks[-1] != line:
            blocks.append(line)

    return blocks


def extract_pdf_blocks(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    blocks: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line or len(line) < 2 or is_excluded_line(line):
                continue
            tagged_line = f"[p.{page_num}] {line}"
            if not blocks or blocks[-1] != tagged_line:
                blocks.append(tagged_line)

    doc.close()
    return blocks


def extract_excel_blocks(excel_path: Path) -> list[str]:
    blocks: list[str] = []
    xl = pd.ExcelFile(excel_path)

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name).fillna("")
        for idx, row in df.iterrows():
            parts = []
            for col, val in row.items():
                s_val = re.sub(r"\s+", " ", str(val)).strip()
                if s_val and s_val.lower() != "nan":
                    parts.append(f"{col}: {s_val}")
            if not parts:
                continue
            line = f"[{sheet_name} #{idx + 1}] " + " | ".join(parts)
            if len(line) < 2 or is_excluded_line(line):
                continue
            if not blocks or blocks[-1] != line:
                blocks.append(line)

    return blocks


def _flatten_json_node(node, prefix: str = "", out: list[str] | None = None) -> list[str]:
    if out is None:
        out = []
    if isinstance(node, dict):
        items = []
        nested = []
        for k, v in node.items():
            key_path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                nested.append((key_path, v))
            elif v is not None:
                s_val = re.sub(r"\s+", " ", str(v)).strip()
                if s_val:
                    items.append(f"{k}: {s_val}")
        if items:
            tag = f"[{prefix}] " if prefix else ""
            out.append(tag + " | ".join(items))
        for key_path, child in nested:
            _flatten_json_node(child, key_path, out)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            item_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_json_node(item, item_prefix, out)
    elif node is not None:
        s_val = re.sub(r"\s+", " ", str(node)).strip()
        if s_val:
            out.append(f"[{prefix}] {s_val}" if prefix else s_val)
    return out


def extract_json_blocks(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    raw_lines = _flatten_json_node(data)
    blocks: list[str] = []
    for line in raw_lines:
        clean_line = re.sub(r"\s+", " ", line).strip()
        if not clean_line or len(clean_line) < 2 or is_excluded_line(clean_line):
            continue
        if not blocks or blocks[-1] != clean_line:
            blocks.append(clean_line)
    return blocks


def filter_useful_medicine_info(blocks: list[str], drug_terms: set[str]) -> list[str]:
    """
    Keep only blocks that relate to the 40 medicines and cover target clinical/safety
    categories (indications, contraindications, dosage, interactions, effects).
    Maintains active section context so paragraphs under a matching heading/drug stay attached.
    """
    if not blocks:
        return []

    full_norm = normalize_text(" ".join(blocks))
    doc_has_40_drug = any(term in full_norm for term in drug_terms) if drug_terms else True
    if not doc_has_40_drug:
        return []

    kept: list[str] = []
    current_category: str | None = None
    active_drug_context = False

    for block in blocks:
        norm = normalize_text(block)
        if is_excluded_line(block):
            continue

        matched_drugs = [t for t in drug_terms if t in norm] if drug_terms else []
        cat = detect_category(block)

        if matched_drugs:
            active_drug_context = True
        elif cat is not None:
            current_category = cat

        has_category_term = any(normalize_text(kw) in norm for kw in ALL_CATEGORY_TERMS)

        if matched_drugs and (has_category_term or len(block) <= 180):
            prefix = f"[{cat}] " if cat else ""
            out_line = f"{prefix}{block}"
            if not kept or kept[-1] != out_line:
                kept.append(out_line)
        elif active_drug_context and (has_category_term or current_category is not None):
            if has_category_term or any(ch.isdigit() for ch in block) or len(block) <= 220:
                tag = cat or current_category or "INFO"
                out_line = f"[{tag}] {block}"
                if not kept or kept[-1] != out_line:
                    kept.append(out_line)

    return kept


def is_already_cleaned(src_file: Path, dst_file: Path, src_hash: str, manifest: dict) -> bool:
    """True if src_file hasn't changed since last clean and dst_file still exists and matches."""
    entry = manifest.get(src_file.as_posix())
    if not entry or not dst_file.exists():
        return False
    stat = src_file.stat()
    if entry.get("src_sha256") != src_hash:
        return False
    if entry.get("src_size") != stat.st_size or entry.get("src_mtime_ns") != stat.st_mtime_ns:
        return False
    if entry.get("dst_sha256") != sha256_file(dst_file):
        return False
    return True


def clean_file(
    src_file: Path,
    dst_file: Path,
    drug_terms: set[str],
    manifest: dict,
) -> tuple[str, int]:
    """
    Returns (status, line_count) where status is 'skipped', 'cleaned', or 'empty'.
    """
    src_hash = sha256_file(src_file)
    rel_key = src_file.as_posix()

    if is_already_cleaned(src_file, dst_file, src_hash, manifest):
        existing_lines = sum(
            1 for ln in dst_file.read_text(encoding="utf-8").splitlines() if ln.strip()
        )
        return ("skipped", existing_lines)

    ext = src_file.suffix.lower()
    if ext in {".html", ".htm"}:
        blocks = extract_html_blocks(src_file)
    elif ext == ".pdf":
        blocks = extract_pdf_blocks(src_file)
    elif ext in {".xlsx", ".xls"}:
        blocks = extract_excel_blocks(src_file)
    elif ext == ".json":
        blocks = extract_json_blocks(src_file)
    else:
        return ("empty", 0)

    useful_lines = filter_useful_medicine_info(blocks, drug_terms)
    if not useful_lines:
        if dst_file.exists():
            dst_file.unlink()
        manifest.pop(rel_key, None)
        return ("empty", 0)

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text("\n".join(useful_lines), encoding="utf-8")

    stat = src_file.stat()
    manifest[rel_key] = {
        "dst_file": dst_file.as_posix(),
        "src_sha256": src_hash,
        "src_size": stat.st_size,
        "src_mtime_ns": stat.st_mtime_ns,
        "dst_sha256": sha256_file(dst_file),
    }
    return ("cleaned", len(useful_lines))


def run_all_folders() -> None:
    raw_dir = RAW_ROOT if RAW_ROOT.exists() else Path("raw")
    if not raw_dir.exists():
        print(f"[ERROR] Raw folder not found at {RAW_ROOT} or raw/")
        return

    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    drug_terms = load_40_drug_terms()
    print(f"Loaded {len(drug_terms)} normalized terms from the 40-drug list.")

    subfolders = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    print(f"Found {len(subfolders)} folders in {raw_dir}: {[p.name for p in subfolders]}")

    total_files = 0
    total_cleaned = 0
    total_skipped = 0
    total_kept_lines = 0

    for folder in subfolders:
        target_folder = CLEAN_ROOT / folder.name
        target_folder.mkdir(parents=True, exist_ok=True)

        media_files = sorted(
            f
            for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        folder_kept = 0
        folder_cleaned = 0
        folder_skipped = 0

        for src_file in media_files:
            # Suffix with _<ext> before .txt so e.g. data.xlsx and data.json don't collide
            rel_base = src_file.relative_to(folder)
            dst_file = target_folder / rel_base.parent / f"{rel_base.stem}_{src_file.suffix.lstrip('.').lower()}.txt"

            status, kept_count = clean_file(src_file, dst_file, drug_terms, manifest)
            total_files += 1
            total_kept_lines += kept_count
            folder_kept += kept_count

            if status == "cleaned":
                folder_cleaned += 1
                total_cleaned += 1
                print(f"  [CLEANED] [{folder.name}] {src_file.name} -> {dst_file.name} ({kept_count} lines)")
            elif status == "skipped":
                folder_skipped += 1
                total_skipped += 1
                print(f"  [SKIPPED] [{folder.name}] {src_file.name} (already clean, {kept_count} lines)")

        print(
            f"Summary [{folder.name}]: {folder_cleaned} cleaned, {folder_skipped} skipped "
            f"({folder_kept} lines total).\n"
        )

    save_manifest(manifest)
    print(
        f"All done! {total_cleaned} cleaned, {total_skipped} skipped out of {total_files} files "
        f"across {len(subfolders)} folders ({total_kept_lines} useful lines total)."
    )


if __name__ == "__main__":
    run_all_folders()