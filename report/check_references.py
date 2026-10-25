"""Check every DOI of report/common/references.bib against Crossref.

For each entry with a DOI, the Crossref record is downloaded and its title, year and first author
are compared with the .bib entry. The result is written to results/logs/references_check.json and
summarised on screen. Entries without DOI are listed so they can be checked by hand.

Run from the repository root (needs internet; uses only the Python standard library):
    python report/check_references.py
"""
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "report" / "common" / "references.bib"
OUT = ROOT / "results" / "logs" / "references_check.json"
HEADERS = {"User-Agent": "glioma-ml-report/1.0 (mailto:orianafe1606@gmail.com)"}


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        fields = dict((k.lower(), v.strip()) for k, v in
                      re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(3)))
        entries.append({"key": m.group(2).strip(), "type": m.group(1).lower(), **fields})
    return entries


def clean(s):
    s = re.sub(r"[{}\\$]", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def main():
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    results, no_doi = [], []
    for e in entries:
        doi = e.get("doi")
        if not doi:
            no_doi.append(e["key"])
            continue
        row = {"key": e["key"], "doi": doi}
        try:
            msg = crossref(doi)
            title = (msg.get("title") or [""])[0]
            year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
            first = (msg.get("author") or [{}])[0].get("family", "")
            sim = difflib.SequenceMatcher(None, clean(e.get("title")), clean(title)).ratio()
            row.update({"found": True, "crossref_title": title, "crossref_year": year,
                        "crossref_first_author": first, "title_similarity": round(sim, 3),
                        "year_matches": str(year) == e.get("year"),
                        "first_author_matches": clean(first) in clean(e.get("author", "")),
                        "ok": sim > 0.85 and str(year) == e.get("year")})
        except urllib.error.HTTPError as err:
            row.update({"found": False, "error": f"HTTP {err.code}", "ok": False})
        except Exception as err:  # network problems
            row.update({"found": False, "error": str(err), "ok": False})
        results.append(row)
        print(f"{'OK ' if row['ok'] else 'CHECK'}  {e['key']:<20} {doi}")
        time.sleep(0.5)   # be polite to the Crossref API
    summary = {"n_entries": len(entries), "n_with_doi": len(results),
               "n_ok": sum(r["ok"] for r in results), "to_check": [r for r in results if not r["ok"]],
               "without_doi": no_doi, "results": results}
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{summary['n_ok']} of {summary['n_with_doi']} DOIs match; "
          f"{len(no_doi)} entries without DOI: {no_doi}")
    print(f"Details in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
