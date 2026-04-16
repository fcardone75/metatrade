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
print("SOGLIE ADATTIVE PER MODULO")
print("=" * 70)
cur.execute("""
    SELECT module_id, threshold, eval_count, correct_count, mean_score
    FROM module_thresholds
    ORDER BY mean_score ASC
""")
rows = cur.fetchall()
if not rows:
    print("  (nessun record — i moduli non hanno ancora accumulato valutazioni)")
else:
    print(f"  {'MODULO':<45} {'SOGLIA':>7} {'VALUT':>6} {'CORR':>5} {'SCORE':>6}")
    print("  " + "-" * 65)
    for module_id, threshold, eval_count, correct_count, mean_score in rows:
        flag = " <-- BASSO" if mean_score is not None and mean_score < 0.4 else ""
        print(
            f"  {module_id:<45} {threshold:>7.3f} {eval_count:>6} "
            f"{correct_count:>5} {(mean_score or 0):>6.3f}{flag}"
        )

print()
print("=" * 70)
print("ULTIME 20 DECISIONI")
print("=" * 70)
cur.execute("""
    SELECT timestamp_utc, direction, final_vote_share, outcome, model_version
    FROM dashboard_decisions
    ORDER BY timestamp_utc DESC
    LIMIT 20
""")
rows = cur.fetchall()
if not rows:
    print("  (nessuna decisione registrata)")
else:
    for ts, direction, vote, outcome, model in rows:
        model_str = model if model else "(nessuno)"
        print(f"  {ts}  {str(direction):<5}  vote={float(vote or 0):.2f}  "
              f"esito={str(outcome):<22}  modello={model_str}")

con.close()
