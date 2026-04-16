"""Diagnostic script — module thresholds and recent decisions."""
import sqlite3
import sys
from pathlib import Path

# Find all .db files under current dir and common locations
search_roots = [Path("."), Path("data"), Path("C:/project/metatrade")]
found = []
for root in search_roots:
    if root.exists():
        found += list(root.rglob("*.db"))

found = sorted(set(found))
print("=" * 70)
print("FILE .db TROVATI")
print("=" * 70)
for f in found:
    print(f"  {f}  ({f.stat().st_size} bytes)")

if not found:
    print("  Nessun file .db trovato.")
    sys.exit(0)

# Also check env for TELEMETRY_DB_PATH
import os
env_path = os.environ.get("TELEMETRY_DB_PATH") or os.environ.get("RUNNER_TELEMETRY_DB")
if env_path:
    print(f"\n  TELEMETRY_DB_PATH env = {env_path}")

# Pick the largest .db (most likely the real one)
db_path = max(found, key=lambda f: f.stat().st_size)
print(f"\nUso: {db_path}")
print()

con = sqlite3.connect(db_path)
cur = con.cursor()

print("=" * 70)
print("TABELLE NEL DB")
print("=" * 70)
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    count = cur.fetchone()[0]
    print(f"  {t:<50} {count:>6} righe")

print()

if "module_thresholds" in tables:
    print("=" * 70)
    print("SOGLIE ADATTIVE PER MODULO")
    print("=" * 70)
    cur.execute("""
        SELECT module_id, threshold, eval_count, correct_count, mean_score
        FROM module_thresholds ORDER BY mean_score ASC
    """)
    rows = cur.fetchall()
    if not rows:
        print("  (vuota)")
    else:
        print(f"  {'MODULO':<45} {'SOGLIA':>7} {'VALUT':>6} {'CORR':>5} {'SCORE':>6}")
        print("  " + "-" * 65)
        for module_id, threshold, eval_count, correct_count, mean_score in rows:
            flag = " <-- BASSO" if mean_score is not None and mean_score < 0.4 else ""
            print(f"  {module_id:<45} {threshold:>7.3f} {eval_count:>6} "
                  f"{correct_count:>5} {(mean_score or 0):>6.3f}{flag}")
    print()

decisions_table = next((t for t in tables if "decision" in t.lower()), None)
if decisions_table:
    print("=" * 70)
    print(f"COLONNE DI {decisions_table}")
    print("=" * 70)
    cur.execute(f"PRAGMA table_info([{decisions_table}])")
    cols = [r[1] for r in cur.fetchall()]
    print("  " + ", ".join(cols))
    print()

    print("=" * 70)
    print(f"ULTIME 20 RIGHE DI {decisions_table}")
    print("=" * 70)
    cur.execute(f"SELECT * FROM [{decisions_table}] ORDER BY rowid DESC LIMIT 20")
    rows = cur.fetchall()
    for row in rows:
        print("  " + str(row))
else:
    print("Nessuna tabella 'decisions' trovata.")

con.close()
