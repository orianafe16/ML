"""Environment check for the project.

Run from the repository root:
    python -m src.check_env

It verifies that the installed package versions match requirements.txt, that the
raw data files are in place, and writes a summary to results/logs/environment.json.
"""
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from src import config


def read_requirements(path):
    """Return {package: pinned_version} from a requirements.txt file."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, ver = line.split("==")
            pins[name.strip()] = ver.strip()
    return pins


def main():
    pins = read_requirements(config.ROOT / "requirements.txt")
    packages = {}
    all_ok = True
    for name, pinned in pins.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        ok = installed == pinned
        all_ok &= ok
        packages[name] = {"required": pinned, "installed": installed, "ok": ok}

    data_files = {}
    for path in (config.RAW_MUTATIONS_ALL, config.RAW_INFO_WITH_GRADE):
        exists = path.exists()
        all_ok &= exists
        n_rows = None
        if exists:
            import pandas as pd
            n_rows = len(pd.read_csv(path))
        data_files[path.name] = {"exists": exists, "n_rows": n_rows}

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_ok": sys.version_info[:2] == (3, 11),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "data_files": data_files,
        "all_ok": bool(all_ok and sys.version_info[:2] == (3, 11)),
    }

    out = config.LOGS / "environment.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Python {summary['python']} ({'OK' if summary['python_ok'] else 'expected 3.11'})")
    for name, info in packages.items():
        flag = "OK" if info["ok"] else "MISMATCH"
        print(f"  {name:<14} required {info['required']:<8} installed {info['installed']}  {flag}")
    for name, info in data_files.items():
        print(f"  {name:<34} exists={info['exists']} rows={info['n_rows']}")
    print(f"\nAll checks passed: {summary['all_ok']}")
    print(f"Summary written to {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
