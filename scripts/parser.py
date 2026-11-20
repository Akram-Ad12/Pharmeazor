
from __future__ import annotations

import re
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# Project paths
BASE_DIR = Path(r"F:\Coding projects\RCR\Pharmeazor")
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "MEDAlgerie"

# Search pages
SOURCES = {
    "AMOXICILLINE": "https://medal-dz.com/search.php?filter=DCI&dci=AMOXICILLINE&dosage=all&forme=all",

    "CEFIXIME": "https://medal-dz.com/search.php?filter=DCI&dci=CEFIXIME&dosage=all&forme=all",

    "CLINDAMYCINE": "https://medal-dz.com/search.php?filter=DCI&dci=CLINDAMYCINE&dosage=all&forme=all",
}


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def download_pages():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    for dci, url in SOURCES.items():

        print(f"\nDownloading {dci}...")

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        # Keep the original downloaded HTML
        html = response.text

        html_file = OUTPUT_DIR / f"{dci.lower()}.html"

        html_file.write_bytes(response.content)

        downloaded[dci] = {
            "html": html,
            "url": url,
            "file": html_file
        }

        print(f"Saved: {html_file}")

    return downloaded


def parse_medal_search_html(html: str, page_url: str, dci: str):

    soup = BeautifulSoup(html, "html.parser")

    collected_at = datetime.now(timezone.utc).isoformat()

    rows = []

    for link in soup.select("main h3 a[href*='medicament.php']"):

        card = link.find_parent("div")

        while card is not None and card.find_parent("main") is not None:

            if card.find("ul") and re.search(
                r"\bDCI\b", card.get_text()
            ):
                break

            card = card.find_parent("div")

        scope = card if card is not None else link.parent

        rows.append({

            "trade_name": link.get_text(),

            "dci": dci,

            "detail_url": urljoin(
                page_url,
                link.get("href", "")
            ),

            "raw_href": link.get("href", ""),

            "country": (
                scope.find("address").get_text()
                if scope.find("address")
                else ""
            ),

            "meta_items": [
                li.get_text()
                for li in (
                    scope.find("ul").find_all("li", recursive=False)
                    if scope.find("ul")
                    else []
                )
            ],

            "raw_text": scope.get_text(
                separator="\n",
                strip=False
            ),

            "raw_html": str(scope),

            "source_url": page_url,

            "source_name": "MEDAlgerie",

            "collected_at": collected_at,

        })

    return rows


def save_data(records):

    # Save combined JSON
    json_file = OUTPUT_DIR / "medal_data_raw.json"

    with open(json_file, "w", encoding="utf-8") as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Save combined CSV
    csv_file = OUTPUT_DIR / "medal_data_raw.csv"

    if records:

        with open(
            csv_file,
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=records[0].keys()
            )

            writer.writeheader()

            for record in records:

                csv_record = record.copy()

                csv_record["meta_items"] = json.dumps(
                    record["meta_items"],
                    ensure_ascii=False
                )

                writer.writerow(csv_record)

    print("\nData saved successfully!")
    print(f"JSON: {json_file}")
    print(f"CSV: {csv_file}")


def main():

    downloaded = download_pages()

    all_records = []

    for dci, source in downloaded.items():

        print(f"\nExtracting {dci}...")

        records = parse_medal_search_html(
            source["html"],
            source["url"],
            dci
        )

        all_records.extend(records)

        print(f"Collected: {len(records)} records")

    save_data(all_records)

    print("\n========== SUMMARY ==========")

    print(f"Total records: {len(all_records)}")

    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
