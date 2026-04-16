"""Diagnostic script — module thresholds and recent decisions."""
import sqlite3
import sys
from pathlib import Path

db_path = Path("data/telemetry.db")
if not db_path.exists():
    print(f"ERROR: {db_path} not found. Run from the repo root.", file=sys.stderr)
    sys.exit(1)

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
