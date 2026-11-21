"""WEEK 2, step 1: cut every cleaned document into CHUNKS (small, self-contained passages).

Run from the repo root (works on your current Python, needs only pandas):
    python scripts/chunk_documents.py

Output: data/chunks/chunks.jsonl   (one chunk per line, JSON)

WHY WE CUT DOCUMENTS
  - The LLM can only read ~4,000 tokens per call, and search must return the RIGHT passage,
    not a whole 263-page PDF.
  - One chunk should hold ONE idea (one interaction, one drug record) so that a citation can
    point to an exact passage.

THE KEY DESIGN IDEA: every chunk has two texts
  text         = the exact original words (what we show and cite; numbers stay verbatim)
  search_text  = text + helpful context (class name, class members, level names...)
                 used ONLY to find the chunk. It never gets shown as a quote.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "clean_data"
OUT_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
SKIP_DIRS = {"venv", ".venv", ".git", "node_modules", "__pycache__"}

MAX_CHARS = 1500          # a chunk longer than this is split (embedding models read ~500 tokens)

# source authority levels decided by the teacher (1 = most trusted)
AUTHORITY = {
    "ANSM Thesaurus": 3,
    "Nomenclature officielle (MIPH)": 2,
    "Pharm'Net": 4,
}

# The 4 official interaction levels (ANSM) and their abbreviations used in the thesaurus
LEVELS = {
    "CI": "Contre-indication",
    "ASDEC": "Association déconseillée",
    "PE": "Précaution d'emploi",
    "APEC": "A prendre en compte",
}
SEVERITY_ORDER = ["CI", "ASDEC", "PE", "APEC"]     # most severe first


# ------------------------------------------------------------------ helpers
def norm(text: str) -> str:
    """lowercase + no accents: used to compare names ('Précaution' == 'precaution')."""
    d = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"\s+", " ", "".join(c for c in d if not unicodedata.combining(c))).strip()


def join_lines(lines: list[str]) -> str:
    """Re-glue lines that the PDF wrapped. 'anti-' + 'inflammatoires' -> 'anti-inflammatoires'."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out += ln
        else:
            out += " " + ln
    return out


def is_upper_line(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split a too-long text at sentence ends, never in the middle of a number."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def find_files(pattern: str) -> list[Path]:
    return sorted(p for p in ROOT.rglob(pattern)
                  if not (set(p.relative_to(ROOT).parts) & SKIP_DIRS))


# ------------------------------------------------------------ 1. ANSM thesaurus
TAGGED = re.compile(r"^(?:\[([A-Z]+)\]\s*)?\[p\.(\d+)\]\s?(.*)$")


def parse_level(line: str):
    """'ASDEC - APEC' -> (['ASDEC','APEC']);  'Précaution d'emploi' -> ['PE'];  else None."""
    s = line.strip()
    if re.fullmatch(r"(?:CI|ASDEC|PE|APEC)(?:\s*-\s*(?:CI|ASDEC|PE|APEC))*", s):
        return re.findall(r"CI|ASDEC|PE|APEC", s)
    n = norm(s)
    for prefix, code in (("contre-indication", "CI"), ("association deconseillee", "ASDEC"),
                         ("precaution d'emploi", "PE"), ("a prendre en compte", "APEC")):
        if n.startswith(prefix):
            return [code]
    return None


def looks_like_header(rows, i: int) -> bool:
    """A real header is an UPPERCASE line that:
         - does not end with punctuation (a wrapped sentence end like 'CYP3A4.' does)
         - is not a level line ('CI - ASDEC', 'PE'...)
         - is followed by a line that does not continue a sentence (lowercase) or give a level."""
    text = rows[i][1]
    if not is_upper_line(text) or text[-1] in ".,;:" or text.startswith(("+", "(")):
        return False
    if parse_level(text):
        return False
    if i + 1 < len(rows):
        nxt = rows[i + 1][1]
        if nxt[:1].islower() or parse_level(nxt):
            return False
    return True


def read_thesaurus_lines(path: Path) -> list[tuple[int, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").replace("\r", "").split("\n"):
        m = TAGGED.match(raw)
        if not m:
            continue
        page, text = int(m.group(2)), m.group(3).strip()
        if not text or re.fullmatch(r"\d{1,3}", text):          # empty line or printed page number
            continue
        rows.append((page, text))
    return rows


def chunk_thesaurus(path: Path) -> list[dict]:
    """Structure of the thesaurus:
         HEADER (substance or class)           e.g. ANTI-INFLAMMATOIRES NON STÉROÏDIENS
         (members of the class ...)            e.g. (diclofenac, ibuprofene, ...)
         + PARTNER                             e.g. + INHIBITEURS DE L'ENZYME DE CONVERSION
         Level line                            e.g. Précaution d'emploi
         description text...
       -> one 'class_definition' chunk per header, one 'interaction' chunk per '+ PARTNER'.
    """
    rows = read_thesaurus_lines(path)
    headers, entries = [], []          # collected in pass 1
    cur_header = None
    cur_entry = None
    state = "scan"                     # scan | header | members | partner | body

    def close_entry():
        nonlocal cur_entry
        if cur_entry:
            entries.append(cur_entry)
            cur_entry = None

    for i, (page, text) in enumerate(rows):
        if text.startswith("+ "):                                  # a new interaction entry
            close_entry()
            cur_entry = {"header": cur_header["name"] if cur_header else None,
                         "page": page, "partner_lines": [text[2:].strip()],
                         "level_line": None, "body": []}
            state = "partner"
            continue

        if state == "partner":
            lv = parse_level(text)
            if lv:
                cur_entry["level_line"] = text
                state = "body"
                continue
            if is_upper_line(text) and not looks_like_header(rows, i):
                cur_entry["partner_lines"].append(text)            # partner name wrapped on 2 lines
                continue
            state = "body"                                         # no level line: it's already text

        if state == "body":
            if looks_like_header(rows, i):                         # a new HEADER begins
                close_entry()
                cur_header = {"name": text, "page": page, "members": [], "note": []}
                headers.append(cur_header)
                state = "header"
            else:
                cur_entry["body"].append(text)
            continue

        # state is "scan", "header" or "members": we are between a header and its first '+' entry
        if state == "members":
            cur_header["members"].append(text)
            if text.rstrip().endswith(")"):
                state = "header"
        elif text.startswith("(") and not norm(text).startswith("(voir"):
            cur_header["members"].append(text)                     # class members: (a, b, c)
            state = "header" if text.rstrip().endswith(")") else "members"
        elif looks_like_header(rows, i):
            if state == "header" and not cur_header["members"] and not cur_header["note"]:
                cur_header["name"] += " " + text                   # wrapped header (2 lines)
            else:
                cur_header = {"name": text, "page": page, "members": [], "note": []}
                headers.append(cur_header)
                state = "header"
        elif cur_header:
            cur_header["note"].append(text)                        # e.g. "Voir aussi : ..."
    close_entry()

    # --- members dictionary: header name -> list of substances (used to enrich partners)
    members_of = {}
    for h in headers:
        raw = join_lines(h["members"]).strip("() ")
        h["member_list"] = [m.strip() for m in raw.split(",") if m.strip()]
        members_of[norm(h["name"])] = h["member_list"]

    chunks = []
    src = "ANSM Thesaurus"

    for i, h in enumerate(headers):
        if not h["member_list"]:
            continue                                               # single substance: no class list
        members = ", ".join(h["member_list"])
        text = f"{h['name']} ({members})"
        chunks.append({
            "chunk_id": f"ansm-p{h['page']}-class-{i:04d}", "source": src,
            "authority_level": AUTHORITY[src], "chunk_type": "class_definition",
            "page": h["page"], "title": h["name"], "text": text,
            "search_text": f"Classe de médicaments : {h['name']}. Substances de cette classe : {members}.",
            "metadata": {"header": h["name"], "members": h["member_list"]},
            "source_file": path.relative_to(ROOT).as_posix(),
        })

    for j, e in enumerate(entries):
        partner = join_lines(e["partner_lines"])
        body = join_lines(e["body"])
        codes = parse_level(e["level_line"]) if e["level_line"] else None
        level_names = [LEVELS[c] for c in codes] if codes else []
        severity = next((c for c in SEVERITY_ORDER if codes and c in codes), None)
        header = e["header"] or "(substance principale non disponible)"
        h_members = members_of.get(norm(header), [])
        p_members = members_of.get(norm(partner), [])

        head_line = f"{header} + {partner}"
        level_line = e["level_line"] or ""
        base_text = join_lines([head_line, level_line, body])
        # exact words shown to the user; split if too long
        pieces = split_long(base_text)
        for k, piece in enumerate(pieces):
            search = (f"Interaction entre {header}"
                      + (f" (contient : {', '.join(h_members)})" if h_members else "")
                      + f" et {partner}"
                      + (f" (contient : {', '.join(p_members)})" if p_members else "")
                      + (f". Niveau : {' / '.join(level_names)}" if level_names else "")
                      + f". {piece}")
            chunks.append({
                "chunk_id": f"ansm-p{e['page']}-int-{j:05d}" + (f"-{k}" if len(pieces) > 1 else ""),
                "source": src, "authority_level": AUTHORITY[src], "chunk_type": "interaction",
                "page": e["page"], "title": head_line, "text": piece, "search_text": search,
                "metadata": {"header": e["header"], "partner": partner, "level_raw": e["level_line"],
                             "level_codes": codes, "level_names": level_names, "most_severe": severity,
                             "orphan_header": e["header"] is None,
                             "header_members": h_members, "partner_members": p_members},
                "source_file": path.relative_to(ROOT).as_posix(),
            })
    return chunks


# ---------------------------------------------------------- 2. Nomenclature CSV
def chunk_nomenclature(path: Path) -> list[dict]:
    """One chunk per registered product (one CSV row). Records are already 'one idea each'."""
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    src = "Nomenclature officielle (MIPH)"
    status_names = {"current": "enregistré (en cours)", "withdrawn": "retiré du marché",
                    "non_renewed": "enregistrement non renouvelé"}
    chunks = []
    for i, r in df.iterrows():
        def d(col):                                              # date without the useless 00:00:00
            return str(r.get(col, "")).split(" ")[0]
        status = r.get("source_sheet", "")
        fields = [("Marque", r["NOM DE MARQUE"]), ("DCI", r["DENOMINATION COMMUNE INTERNATIONALE"]),
                  ("Forme", r["FORME"]), ("Dosage", r["DOSAGE"]), ("Conditionnement", r["CONDITIONNEMENT"]),
                  ("Laboratoire", r["LABORATOIRES DETENTEUR DE LA DECISION D'ENREGISTREMENT"]),
                  ("Pays", r["PAYS DU LABORATOIRE DETENTEUR DE LA DECISION D'ENREGISTREMENT"]),
                  ("N° enregistrement", r["N°ENREGISTREMENT"]), ("Liste", r["LISTE"]),
                  ("Enregistrement initial", d("DATE D'ENREGISTREMENT INITIAL")),
                  ("Enregistrement final", d("DATE D'ENREGISTREMENT FINAL")),
                  ("Retrait", d("DATE DE RETRAIT")), ("Motif de retrait", r["MOTIF DE RETRAIT"]),
                  ("Statut", status_names.get(status, status))]
        text = " | ".join(f"{k} : {v}" for k, v in fields if str(v).strip())
        search = (f"{text} | Médicament cible : {r['target_dci']} | Classe : {r['therapeutic_class']}"
                  f" | ATC indicatif : {r['atc_indicative']}")
        chunks.append({
            "chunk_id": f"nomenclature-{i:05d}", "source": src, "authority_level": AUTHORITY[src],
            "chunk_type": "drug_record", "page": None,
            "title": f"{r['NOM DE MARQUE']} ({r['DENOMINATION COMMUNE INTERNATIONALE']}) {r['DOSAGE']}",
            "text": text, "search_text": search,
            "metadata": {"brand": r["NOM DE MARQUE"], "dci": r["DENOMINATION COMMUNE INTERNATIONALE"],
                         "target_dci": r["target_dci"], "registration_no": r["N°ENREGISTREMENT"],
                         "registration_status": status, "form": r["FORME"], "dosage": r["DOSAGE"],
                         "atc_indicative": r["atc_indicative"]},
            "source_file": path.relative_to(ROOT).as_posix(),
        })
    return chunks


# ------------------------------------------------------ 3. generic text documents
NOISE = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M$"),      # print date "11/20/26, 7:26 PM"
    re.compile(r"^\d/\d$"),                                                   # page "1/3"
    re.compile(r"^https?://\S+$"),                                            # URL line (kept separately)
    re.compile(r"^(En ligne|GP|Grand public|Base Supabase.*|Non renseigné|Local|Substituable)$"),
    re.compile(r"^Pharm'Net, Le référentiel.*|^présentées relèvent.*", re.I),
]


def chunk_generic(path: Path) -> list[dict]:
    """For small text files (Pharm'Net product pages, brand lists...): remove print noise and
    the meaningless [DOSAGE]/[p.N] tags, then cut in windows of <= MAX_CHARS with 150 chars overlap.
    (Overlap = the end of one chunk is repeated at the start of the next, so a sentence cut
    in two can still be found.)"""
    lines, url = [], ""
    for raw in path.read_text(encoding="utf-8").replace("\r", "").split("\n"):
        m = TAGGED.match(raw)
        text = (m.group(3) if m else re.sub(r"^\[[A-Z]+\]\s*", "", raw)).strip()
        if not text:
            continue
        if text.startswith("http"):
            url = text
        if any(p.match(text) for p in NOISE):
            continue
        lines.append(text)

    is_pharmnet = "pharm_net" in norm(path.name) or "pharm'net" in norm(path.name)
    source = "Pharm'Net" if is_pharmnet else f"A identifier ({path.name})"
    authority = AUTHORITY.get(source)                 # None = unknown, to be filled by the team
    title = lines[0] if lines else path.stem
    body = join_lines([ln if not re.fullmatch(r"[A-Za-zÀ-ÿ' ()\-]+", ln) or len(ln) > 40 else ln + " ;" for ln in lines])
    # simple window cut
    windows, start = [], 0
    while start < len(body):
        end = min(start + MAX_CHARS, len(body))
        if end < len(body):                           # cut at a space, not inside a word
            end = body.rfind(" ", start, end) or end
        windows.append(body[start:end].strip())
        if end >= len(body):
            break
        start = max(end - 150, start + 1)
    chunks = []
    for k, w in enumerate(windows):
        chunks.append({
            "chunk_id": f"{re.sub(r'[^a-z0-9]+', '-', norm(path.stem)).strip('-')}-{k:02d}",
            "source": source, "authority_level": authority, "chunk_type": "text_window",
            "page": None, "title": title, "text": w, "search_text": f"{title}. {w}",
            "metadata": {"source_url": url}, "source_file": path.relative_to(ROOT).as_posix(),
        })
    return chunks


# --------------------------------------------------------------------- main
def main():
    all_chunks = []
    for csv_path in find_files("nomenclature_40_drugs_clean.csv"):
        all_chunks += chunk_nomenclature(csv_path)
        break
    for txt in find_files("*.txt"):
        if CLEAN_DIR not in txt.parents:
            continue
        if "thesaurus" in norm(txt.name):
            all_chunks += chunk_thesaurus(txt)
        else:
            all_chunks += chunk_generic(txt)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- report
    print(f"\nWrote {len(all_chunks)} chunks -> {OUT_FILE.relative_to(ROOT)}\n")
    by = {}
    for c in all_chunks:
        by.setdefault((c["source"], c["chunk_type"]), []).append(len(c["text"]))
    print(f"{'source':38} {'type':17} {'count':>6} {'avg chars':>10} {'max':>6}")
    for (s, t), lens in sorted(by.items()):
        print(f"{s[:37]:38} {t:17} {len(lens):6d} {sum(lens)//len(lens):10d} {max(lens):6d}")
    orphans = sum(1 for c in all_chunks if c["metadata"].get("orphan_header"))
    print(f"\nInteraction chunks with a missing header (orphans): {orphans}")
    unknown = {c['source_file'] for c in all_chunks if c["authority_level"] is None}
    if unknown:
        print("Files with UNKNOWN source/authority (tell me where they came from):", sorted(unknown))


if __name__ == "__main__":
    main()
