import re
import unicodedata
from pathlib import Path
import fitz  # type: ignore # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup # type: ignore

RAW_ROOT = Path("data/raw")
CLEAN_ROOT = Path("clean_data")

EXCEL_CANDIDATES = [
    Path("data/drug_list/Projet2_Liste_40_medicaments.xlsx"),
    Path("data/drug_list/Projet2_Liste_40_medicaments_2.xlsx"),
    Path("data/raw/Projet2_Liste_40_medicaments.xlsx"),
    Path("Projet2_Liste_40_medicaments.xlsx"),
    Path("Projet2_Liste_40_medicaments_2.xlsx"),
]

SUPPORTED_EXTENSIONS = {".html", ".htm", ".pdf"}

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

        # Extract quoted or capitalized brand names if mentioned in Remarques
        if rem_norm and rem_norm != "nan":
            for token in re.findall(r"[a-z]{5,}", rem_norm):
                if token not in {"only", "entries", "compared", "others", "imported", "brands", "mylan", "chlorhydrate"}:
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
            cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)).strip() for c in elem.find_all(["th", "td"])]
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

        # Keep if:
        # 1) Block mentions one of the 40 medicines and has a target category (or is a heading/row), OR
        # 2) We are inside a drug/category section and the block contains category/dosage/safety info
        if matched_drugs and (has_category_term or len(block) <= 180):
            prefix = f"[{cat}] " if cat else ""
            out_line = f"{prefix}{block}"
            if not kept or kept[-1] != out_line:
                kept.append(out_line)
        elif active_drug_context and (has_category_term or current_category is not None):
            # Avoid keeping completely unrelated filler lines even inside a section
            if has_category_term or any(ch.isdigit() for ch in block) or len(block) <= 220:
                tag = cat or current_category or "INFO"
                out_line = f"[{tag}] {block}"
                if not kept or kept[-1] != out_line:
                    kept.append(out_line)

    return kept


def clean_file(src_file: Path, dst_file: Path, drug_terms: set[str]) -> int:
    ext = src_file.suffix.lower()
    if ext in {".html", ".htm"}:
        blocks = extract_html_blocks(src_file)
    elif ext == ".pdf":
        blocks = extract_pdf_blocks(src_file)
    else:
        return 0

    useful_lines = filter_useful_medicine_info(blocks, drug_terms)
    if not useful_lines:
        # Remove any stale 0-line file from earlier runs so folders stay clean
        if dst_file.exists():
            dst_file.unlink()
        return 0

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text("\n".join(useful_lines), encoding="utf-8")
    return len(useful_lines)


def run_all_folders() -> None:
    raw_dir = RAW_ROOT if RAW_ROOT.exists() else Path("raw")
    if not raw_dir.exists():
        print(f"[ERROR] Raw folder not found at {RAW_ROOT} or raw/")
        return

    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    drug_terms = load_40_drug_terms()
    print(f"Loaded {len(drug_terms)} normalized terms from the 40-drug list.")

    subfolders = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    print(f"Found {len(subfolders)} folders in {raw_dir}: {[p.name for p in subfolders]}")

    total_files = 0
    total_saved_files = 0
    total_kept_lines = 0

    for folder in subfolders:
        target_folder = CLEAN_ROOT / folder.name
        target_folder.mkdir(parents=True, exist_ok=True)

        # Only .html/.htm and .pdf files (ignores .xlsx, etc.)
        media_files = sorted(
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        folder_kept = 0
        folder_saved = 0
        for src_file in media_files:
            rel_path = src_file.relative_to(folder).with_suffix(".txt")
            dst_file = target_folder / rel_path
            kept_count = clean_file(src_file, dst_file, drug_terms)
            total_files += 1
            total_kept_lines += kept_count
            folder_kept += kept_count
            if kept_count > 0:
                folder_saved += 1
                total_saved_files += 1
                print(f"  [{folder.name}] {src_file.name} -> {dst_file.name} ({kept_count} lines)")

        print(
            f"Summary [{folder.name}]: {folder_saved}/{len(media_files)} html/pdf files had useful lines "
            f"({folder_kept} lines total).\n"
        )

    print(
        f"All done! Kept {total_saved_files}/{total_files} html/pdf files across {len(subfolders)} folders "
        f"({total_kept_lines} useful lines total)."
    )

if __name__ == "__main__":
    run_all_folders()