"""Fix Unicode chars that break Windows cp1252 console output."""
import pathlib

replacements = [
    ("\u2026", "..."),
    ("\u2192", "->"),
    ("\u2014", "--"),
    ("\u2013", "-"),
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
]

files = [
    "data_analysis/db_preprocessing.py",
    "models/db_promotion_engine.py",
    "train_db_aligned.py",
    "api/promotion_api.py",
]

for fpath in files:
    p = pathlib.Path(fpath)
    if not p.exists():
        print(f"SKIP (not found): {fpath}")
        continue
    text = p.read_text(encoding="utf-8")
    changed = False
    for bad, good in replacements:
        if bad in text:
            text = text.replace(bad, good)
            changed = True
    if changed:
        p.write_text(text, encoding="utf-8")
        print(f"Fixed: {fpath}")
    else:
        print(f"OK   : {fpath}")
