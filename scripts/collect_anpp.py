import hashlib
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup # type: ignore

HEADERS = {
    "User-Agent": "PharmeazorCollector/1.0 (M1-IA Academic Project)"
}
START_URL = "https://anpp.dz/fr/"
BASE_DOMAIN = "anpp.dz"
RAW_ANPP_DIR = Path("data/raw/anpp")

# Keywords to prioritize drug/RCP/notice/nomenclature links on anpp.dz/fr/
PRIORITY_KEYWORDS = (
    "rcp",
    "notice",
    "medicament",
    "médicament",
    "produit",
    "nomenclature",
    "enregistrement",
    "pharmaceutique",
    "pharmacovigilance",
    "interaction",
    "wpfd_file",
    "liste",
    "decision",
    "décision",
)


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and BASE_DOMAIN not in parsed.netloc.lower():
        return False
    path = parsed.path.lower()
    if path.startswith("/wp-admin/") and path != "/wp-admin/admin-ajax.php":
        return False
    # Skip media assets that aren't documents
    if any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".mp4", ".css", ".js")):
        return False
    return True


def safe_filename(url: str, content_hash: str, ext: str) -> str:
    path = urlparse(url).path.strip("/") or "fr_home"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path).strip("_")[:75] or "page"
    return f"{slug}_{content_hash[:8]}{ext}"


def existing_hashes(folder: Path) -> set[str]:
    hashes = set()
    if not folder.exists():
        return hashes
    for f in folder.iterdir():
        if f.is_file():
            hashes.add(hashlib.sha256(f.read_bytes()).hexdigest())
    return hashes


def link_priority(url: str, anchor_text: str) -> int:
    combined = f"{url} {anchor_text}".lower()
    return sum(1 for kw in PRIORITY_KEYWORDS if kw in combined)


def crawl_anpp_fr(max_pages: int = 30, delay: float = 2.0) -> None:
    RAW_ANPP_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes = existing_hashes(RAW_ANPP_DIR)
    visited: set[str] = set()

    # Queue items: (priority_score, url)
    queue: deque[str] = deque([START_URL])
    queued: set[str] = {START_URL}

    saved_html = 0
    saved_docs = 0

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited or not is_allowed_url(url):
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=25)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Failed {url}: {exc}")
            time.sleep(delay)
            continue

        content_type = resp.headers.get("Content-Type", "").lower()
        raw_bytes = resp.content
        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Case 1: Direct PDF / Excel / Word document (e.g., /wpfd_file/... or .pdf)
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            if file_hash not in seen_hashes:
                out_path = RAW_ANPP_DIR / safe_filename(url, file_hash, ".pdf")
                out_path.write_bytes(raw_bytes)
                seen_hashes.add(file_hash)
                saved_docs += 1
                print(f"[PDF] Saved -> {out_path.name} ({url})")
            time.sleep(delay)
            continue

        if "spreadsheet" in content_type or "excel" in content_type or url.lower().endswith((".xlsx", ".xls")):
            ext = ".xlsx" if ".xlsx" in url.lower() else ".xls"
            if file_hash not in seen_hashes:
                out_path = RAW_ANPP_DIR / safe_filename(url, file_hash, ext)
                out_path.write_bytes(raw_bytes)
                seen_hashes.add(file_hash)
                saved_docs += 1
                print(f"[EXCEL] Saved -> {out_path.name} ({url})")
            time.sleep(delay)
            continue

        # Case 2: HTML page
        if "html" not in content_type and not resp.text.lstrip().startswith("<"):
            time.sleep(delay)
            continue

        if file_hash not in seen_hashes:
            out_path = RAW_ANPP_DIR / safe_filename(url, file_hash, ".html")
            out_path.write_bytes(raw_bytes)
            seen_hashes.add(file_hash)
            saved_html += 1
            print(f"[HTML] Saved -> {out_path.name} ({url})")

        # Parse internal /fr/ links and downloadable documents
        soup = BeautifulSoup(resp.text, "html.parser")
        discovered: list[tuple[int, str]] = []

        for a in soup.find_all("a", href=True):
            full_url = urljoin(url, a["href"]).split("#")[0].strip()
            if not full_url or full_url in visited or full_url in queued:
                continue
            if not is_allowed_url(full_url):
                continue

            parsed = urlparse(full_url)
            path_low = parsed.path.lower()

            # Keep /fr/ pages or document download links (/wpfd_file/, .pdf, .xlsx)
            is_fr_page = path_low == "/fr" or path_low.startswith("/fr/")
            is_doc_link = "/wpfd_file/" in path_low or path_low.endswith((".pdf", ".xlsx", ".xls"))

            if is_fr_page or is_doc_link:
                score = link_priority(full_url, a.get_text(" ", strip=True))
                discovered.append((score, full_url))

        # Put high-priority drug/RCP/document links at the front of the queue
        discovered.sort(key=lambda x: x[0], reverse=True)
        for score, link in discovered:
            if link not in queued:
                queued.add(link)
                if score > 0:
                    queue.appendleft(link)
                else:
                    queue.append(link)

        time.sleep(delay)

    print(f"\nDone! Visited {len(visited)} URLs | Saved {saved_html} HTML pages and {saved_docs} PDF/Excel files in {RAW_ANPP_DIR}.")


if __name__ == "__main__":
    crawl_anpp_fr(max_pages=30, delay=2.0)