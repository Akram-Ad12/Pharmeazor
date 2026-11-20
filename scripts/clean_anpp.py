import re
import unicodedata
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup

RAW_ANPP_DIR = Path("data/raw/anpp")
CLEAN_ANPP_DIR = Path("clean_data/anpp")

EXCEL_CANDIDATES = [
    Path("data/drug_list/Projet2_Liste_40_medicaments.xlsx"),
    Path("data/raw/Projet2_Liste_40_medicaments.xlsx"),
    Path("Projet2_Liste_40_medicaments.xlsx"),
]

NOISE_LINE_PATTERNS = (
    "prix",
    "ppa",
    "rupture de stock",
    "en stock",
    "publicite",
    "publicité",
    "sponsorise",
    "sponsorisé",
    "complement alimentaire",
    "complément alimentaire",
    "dispositif medical",
    "dispositif médical",
)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def find_drug_list_excel() -> Path | None:
    for candidate in EXCEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_drug_terms() -> set[str]:
    terms: set[str] = set()
    excel_path = find_drug_list_excel()
    if excel_path is None:
        return terms

    df = pd.read_excel(excel_path, sheet_name="Médicaments")
    for _, row in df.iterrows():
        dci_norm = normalize_text(str(row.get("DCI (nom de la substance)", "") or ""))
        atc_norm = normalize_text(str(row.get("Code ATC (indicatif)", "") or ""))

        if dci_norm and dci_norm != "nan":
            terms.add(dci_norm)
            cleaned_parens = re.sub(r"[()]", " ", dci_norm)
            for part in re.split(r"\s*\+\s*|\s+et\s+|\s*/\s*", cleaned_parens):
                part_clean = part.strip()
                if len(part_clean) >= 4:
                    terms.add(part_clean)

        if atc_norm and atc_norm != "nan" and len(atc_norm) >= 5:
            terms.add(atc_norm)

    return terms


def is_noise_line(line: str) -> bool:
    norm = normalize_text(line)
    return any(normalize_text(pat) in norm for pat in NOISE_LINE_PATTERNS)


def extract_html_lines(html_path: Path) -> list[str]:
    raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()

    root = soup.body or soup
    lines: list[str] = []

    for tr in root.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)).strip() for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            row_str = " | ".join(cells)
            if not is_noise_line(row_str) and (not lines or lines[-1] != row_str):
                lines.append(row_str)
        tr.decompose()

    for raw_line in root.get_text("\n").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) < 2 or is_noise_line(line):
            continue
        if not lines or lines[-1] != line:
            lines.append(line)

    return lines


def extract_pdf_lines(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    lines: list[str] = []
    for page_idx, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        if not page_text:
            continue
        lines.append(f"--- PAGE {page_idx} ---")
        for raw_line in page_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line or is_noise_line(line):
                continue
            if not lines or lines[-1] != line:
                lines.append(line)
    doc.close()
    return lines


def find_matched_drugs(lines: list[str], drug_terms: set[str]) -> list[str]:
    joined_norm = normalize_text(" ".join(lines))
    return sorted(t for t in drug_terms if t in joined_norm)


def run_cleaner() -> None:
    if not RAW_ANPP_DIR.exists():
        print(f"Raw folder not found: {RAW_ANPP_DIR}")
        return

    CLEAN_ANPP_DIR.mkdir(parents=True, exist_ok=True)
    drug_terms = load_drug_terms()

    files = sorted(
        f for f in RAW_ANPP_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".html", ".pdf")
    )
    print(f"Found {len(files)} raw files (.html/.pdf) in {RAW_ANPP_DIR}.")

    for f in files:
        if f.suffix.lower() == ".html":
            lines = extract_html_lines(f)
        else:
            lines = extract_pdf_lines(f)

        matched = find_matched_drugs(lines, drug_terms)
        out_file = CLEAN_ANPP_DIR / f"{f.stem}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")

        drug_info = f" | 40-drug matches: {', '.join(matched[:6])}" if matched else " | 40-drug matches: none (general/regulatory page)"
        print(f"[SAVED] {out_file.name} ({len(lines)} lines){drug_info}")


if __name__ == "__main__":
    run_cleaner()