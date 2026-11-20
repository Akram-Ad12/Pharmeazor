from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pdfplumber
import pymupdf
from pydantic import BaseModel, Field

try:
    from PIL import Image
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


ExtractionMethod = Literal["pymupdf", "pdfplumber", "ocr_tesseract", "ocr_easyocr"]

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed"


class PageRecord(BaseModel):
    page_number: int
    method: ExtractionMethod
    raw_text: str
    cleaned_text: str


class ExtractedNotice(BaseModel):
    document_id: str
    source_file: str
    source_path: str
    sha256: str
    collected_at: str
    page_count: int
    extraction_methods_used: list[ExtractionMethod]
    metadata_hints: dict[str, list[str]] = Field(default_factory=dict)
    sections: dict[str, str] = Field(default_factory=dict)
    full_cleaned_text: str
    pages: list[PageRecord]


SECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "indications": [
        re.compile(
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:indications?\s+th[ée]rapeutiques?|qu['’]est-ce\s+que\s+.+\s+et\s+dans\s+quel\s+cas)\s*:?\s*$"
        ),
    ],
    "contre_indications": [
        re.compile(
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:contre-?indications?|ne\s+jamais\s+prendre|quand\s+ne\s+pas\s+utiliser)\s*:?\s*$"
        ),
    ],
    "posologie": [
        re.compile(
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:posologie(?:\s+et\s+mode\s+d['’]administration)?|comment\s+prendre|dose\s+usuelle)\s*:?\s*$"
        ),
    ],
    "interactions": [
        re.compile(
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:interactions?\s+m[ée]dicamenteuses?|autres\s+m[ée]dicaments\s+et)\s*:?\s*$"
        ),
    ],
    "effets_indesirables": [
        re.compile(
            r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:effets?\s+ind[ée]sirables(?:\s+eventuels)?)\s*:?\s*$"
        ),
    ],
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_medical_text(text: str) -> str:
    """
    Cleans extracted PDF text while preserving exact numbers, units,
    punctuation, and medical phrasing for downstream fact/number verification.
    """
    if not text:
        return ""

    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—−]", "-", text)

    # Fix hyphenated line breaks across lines (e.g., "anti-\ninflammatoire" -> "anti-inflammatoire")
    text = re.sub(r"([A-Za-zÀ-ÿ]{2,})-\s*\n\s*([A-Za-zÀ-ÿ]{2,})", r"\1-\2", text)

    # Remove common page headers/footers like "Page 1/8" or "Page : 2/8"
    text = re.sub(r"(?im)^\s*page\s*:?\s*\d+\s*/\s*\d+\s*$", "", text)

    # Collapse excessive spaces/tabs on each line while preserving structure
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _ocr_page_pymupdf(
    page: pymupdf.Page, lang: str = "fra+ara+eng"
) -> tuple[str, ExtractionMethod] | None:
    pix = page.get_pixmap(dpi=250)
    img_bytes = pix.tobytes("png")

    if HAS_TESSERACT:
        image = Image.open(io.BytesIO(img_bytes))
        try:
            txt = pytesseract.image_to_string(image, lang=lang)
            if txt.strip():
                return txt, "ocr_tesseract"
        except Exception:
            pass

    if HAS_EASYOCR:
        reader = easyocr.Reader(["fr", "ar", "en"], gpu=False, verbose=False)
        results = reader.readtext(img_bytes, detail=0)
        txt = "\n".join(results)
        if txt.strip():
            return txt, "ocr_easyocr"

    return None


def extract_pages(pdf_path: Path, min_text_chars: int = 40) -> list[PageRecord]:
    records: list[PageRecord] = []

    with pymupdf.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumber_doc:
        for idx, page in enumerate(doc):
            page_num = idx + 1
            raw_text = page.get_text("text") or ""
            method: ExtractionMethod = "pymupdf"

            # Fallback to pdfplumber for tabular or complex layout pages
            if len(raw_text.strip()) < min_text_chars and idx < len(plumber_doc.pages):
                plumber_text = plumber_doc.pages[idx].extract_text() or ""
                tables = plumber_doc.pages[idx].extract_tables() or []
                table_lines = [
                    " | ".join(cell.strip() for cell in row if cell and cell.strip())
                    for table in tables
                    for row in table
                    if any(cell and cell.strip() for cell in row)
                ]
                combined_plumber = "\n".join([plumber_text] + table_lines).strip()
                if len(combined_plumber) > len(raw_text.strip()):
                    raw_text = combined_plumber
                    method = "pdfplumber"

            # Fallback to OCR if the page is scanned / has no embedded text
            if len(raw_text.strip()) < min_text_chars:
                ocr_res = _ocr_page_pymupdf(page)
                if ocr_res is not None:
                    raw_text, method = ocr_res

            cleaned = clean_medical_text(raw_text)
            records.append(
                PageRecord(
                    page_number=page_num,
                    method=method,
                    raw_text=raw_text,
                    cleaned_text=cleaned,
                )
            )

    return records


def extract_metadata_hints(text: str) -> dict[str, list[str]]:
    """Extracts helpful identifiers such as ATC codes or dosage mentions."""
    atc_matches = sorted(set(re.findall(r"\b[A-Z]\d{2}[A-Z]{2}\d{2}\b", text)))
    dosage_matches = sorted(
        set(
            re.findall(
                r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|ml|mcg|µg|UI|%)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    return {
        "atc_codes": atc_matches,
        "dosages_found": dosage_matches,
    }


def extract_sections(full_text: str) -> dict[str, str]:
    """
    Splits cleaned notice/RCP text into core sections using heading rules.
    Preserves original wording inside each section.
    """
    heading_positions: list[tuple[int, int, str]] = []

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(full_text):
                heading_positions.append((match.start(), match.end(), section_name))

    heading_positions.sort(key=lambda item: item[0])

    sections: dict[str, str] = {k: "" for k in SECTION_PATTERNS}
    if not heading_positions:
        return sections

    for i, (_, content_start, section_name) in enumerate(heading_positions):
        next_start = (
            heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(full_text)
        )
        snippet = full_text[content_start:next_start].strip()
        if snippet:
            if sections[section_name]:
                sections[section_name] += "\n\n" + snippet
            else:
                sections[section_name] = snippet

    return sections


def process_pdf_file(pdf_path: Path) -> ExtractedNotice:
    pages = extract_pages(pdf_path)
    full_cleaned = "\n\n".join(p.cleaned_text for p in pages if p.cleaned_text).strip()
    methods_used = list(dict.fromkeys(p.method for p in pages))
    file_hash = compute_sha256(pdf_path)

    return ExtractedNotice(
        document_id=f"{pdf_path.stem}_{file_hash[:10]}",
        source_file=pdf_path.name,
        source_path=str(pdf_path.resolve()),
        sha256=file_hash,
        collected_at=datetime.now(timezone.utc).isoformat(),
        page_count=len(pages),
        extraction_methods_used=methods_used,
        metadata_hints=extract_metadata_hints(full_cleaned),
        sections=extract_sections(full_cleaned),
        full_cleaned_text=full_cleaned,
        pages=pages,
    )


def process_raw_folder(
    raw_dir: Path = DEFAULT_RAW_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[ExtractedNotice]:
    """Reads all *.pdf files in `raw_dir` (including subfolders) and saves structured JSON."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw PDF directory not found: {raw_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.rglob("*.pdf"))

    results: list[ExtractedNotice] = []
    for pdf_file in pdf_files:
        notice = process_pdf_file(pdf_file)
        out_file = output_dir / f"{notice.document_id}.json"
        out_file.write_text(notice.model_dump_json(indent=2), encoding="utf-8")
        results.append(notice)

    manifest_path = output_dir / "manifest.json"
    manifest_data = [
        {
            "document_id": n.document_id,
            "source_file": n.source_file,
            "sha256": n.sha256,
            "collected_at": n.collected_at,
            "page_count": n.page_count,
            "methods": n.extraction_methods_used,
            "sections_found": [k for k, v in n.sections.items() if v],
        }
        for n in results
    ]
    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(f"Looking for PDFs in: {DEFAULT_RAW_DIR.resolve()}")
    notices = process_raw_folder(raw_dir=DEFAULT_RAW_DIR, output_dir=DEFAULT_OUTPUT_DIR)
    print(f"Processed {len(notices)} PDF file(s) -> saved to {DEFAULT_OUTPUT_DIR.resolve()}")
    for n in notices:
        found = [k for k, v in n.sections.items() if v]
        print(f" - {n.source_file} -> id={n.document_id}, pages={n.page_count}, sections={found}")