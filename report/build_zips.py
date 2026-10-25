"""Build one Overleaf-ready zip per report in report/dist/.

Each zip contains: main.tex, sections/, numbers.tex, references.bib, figures/ (the course logo and
the template figures of the Ensemble Learning appendix, taken from the template zips, plus only the
result figures that the report uses) and tables/ (only the tables it uses).

Run from the repository root, after the notebooks and report/make_numbers.py:
    python report/build_zips.py
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
RESULTS = ROOT / "results"
DIST = REPORT / "dist"
TEMPLATES = {"prdl": REPORT / "templates" / "PRDL_Report_Overleaf.zip",
             "mllb": REPORT / "templates" / "MLLB_Report_Overleaf.zip"}

# Tables generated before the escaping fix in src/reporting.py printed "\$\textbackslash{}pm\$"
LEGACY_FIXES = {r"\$\textbackslash{}pm\$": r"$\pm$"}


def used_files(tex):
    figures = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    tables = set(re.findall(r"\\input\{tables/([^}]+?)(?:\.tex)?\}", tex))
    return figures, tables


def build(name):
    src = REPORT / name
    tex = (src / "main.tex").read_text(encoding="utf-8")
    sections = sorted((src / "sections").glob("*.tex"))
    full = tex + "".join(s.read_text(encoding="utf-8") for s in sections)
    figures, tables = used_files(full)

    DIST.mkdir(exist_ok=True)
    out = DIST / f"{name.upper()}_Overleaf.zip"
    missing = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", tex)
        for s in sections:
            z.write(s, f"sections/{s.name}")
        z.write(REPORT / "common" / "numbers.tex", "numbers.tex")
        z.write(REPORT / "common" / "references.bib", "references.bib")
        with zipfile.ZipFile(TEMPLATES[name]) as tz:          # logo + appendix figures
            for item in tz.namelist():
                if item.startswith("figures/") and not item.endswith("/"):
                    z.writestr(item, tz.read(item))
        for fig in sorted(figures):
            path = RESULTS / "figures" / fig
            if path.exists():
                z.write(path, f"figures/{fig}")
            elif not fig.startswith(("figures/", "etsit", "roc_curves", "threshold_tradeoff")):
                missing.append(fig)
        for tab in sorted(tables):
            path = RESULTS / "tables" / f"{tab}.tex"
            if not path.exists():
                missing.append(f"{tab}.tex")
                continue
            content = path.read_text(encoding="utf-8")
            for old, new in LEGACY_FIXES.items():
                content = content.replace(old, new)
            z.writestr(f"tables/{tab}.tex", content)
    print(f"{out.relative_to(ROOT)}: {len(figures)} figures, {len(tables)} tables")
    if missing:
        print("  MISSING:", missing)
    return not missing


if __name__ == "__main__":
    subprocess.run([sys.executable, str(REPORT / "make_numbers.py")], check=True)
    ok = all(build(n) for n in ("prdl", "mllb"))
    sys.exit(0 if ok else 1)
