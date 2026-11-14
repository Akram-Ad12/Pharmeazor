"""Check what each website's robots.txt says about automated access.

Usage (from the repo root):
    python scripts/check_robots.py                       # default sites
    python scripts/check_robots.py https://site/page     # exact pages you plan to download

IMPORTANT: robots.txt is only half of the rule. You must ALSO read each site's terms of use
and write both findings in reports/sources_terms.md. If a site forbids robots -> download
manually or discard it. NEVER bypass.
"""
import sys
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

# Add the exact URLs of the sources listed on the teacher's page (Pharm'Net, MEDAlgérie, ANPP RCP search...)
DEFAULT_URLS = [
    "https://www.miph.gov.dz/",
    "https://pharmnet-dz.com/",
    "https://dzair-pharma.net/",
    "https://dailymed.nlm.nih.gov/dailymed/",
    "https://www.ema.europa.eu/en",
    "https://ansm.sante.fr/",
    "https://medal-dz.com/",
    "https://anpp.dz/wpfd_file/regles-de-redaction-de-linformation-produit-rcp-notice-et-conditionnement-des-medicaments-destines-a-lusage-de-la-medecine-humaine/"
    #pass them on the command line:  python scripts/check_robots.py https://example.dz/page
]
USER_AGENT = "PharmeazorStudentProject/0.1 (university course project)"


def check(url: str) -> None:
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        r = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=20)
    except requests.RequestException as e:
        print(f"[?] {url}\n    could not fetch robots.txt ({e.__class__.__name__}) -> treat as UNKNOWN, read the ToS\n")
        return
    if r.status_code == 404:
        print(f"[ok] {url}\n    no robots.txt (404): robots.txt does not forbid anything. ToS still applies.\n")
        return
    if r.status_code != 200:
        print(f"[?] {url}\n    robots.txt returned HTTP {r.status_code} -> UNKNOWN, read the ToS\n")
        return
    parser = RobotFileParser()
    parser.parse(r.text.splitlines())
    allowed = parser.can_fetch(USER_AGENT, url)
    delay = parser.crawl_delay(USER_AGENT)
    print(f"[{'ok' if allowed else 'FORBIDDEN'}] {url}")
    print(f"    robots.txt: {robots_url}")
    print(f"    automated access: {'allowed' if allowed else 'FORBIDDEN -> manual download only'}"
          + (f" | crawl-delay: {delay}s" if delay else ""))
    print()


if __name__ == "__main__":
    for u in (sys.argv[1:] or DEFAULT_URLS):
        check(u)
