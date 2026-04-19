"""Diagnostic script — module thresholds and recent decisions."""
import os
import sqlite3
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.log import configure_logging, get_logger

configure_logging()
log = get_logger("diag_modules")

# Find all .db files under current dir and common locations
search_roots = [Path("."), Path("data"), Path("C:/project/metatrade")]
found = []
for root in search_roots:
    if root.exists():
        found += list(root.rglob("*.db"))

found = sorted(set(found))
log.info("db_files_found", count=len(found), paths=[str(f) for f in found])

if not found:
    log.warning("no_db_files_found")
    sys.exit(0)

# Also check env for TELEMETRY_DB_PATH
env_path = os.environ.get("TELEMETRY_DB_PATH") or os.environ.get("RUNNER_TELEMETRY_DB")
if env_path:
    log.info("telemetry_db_env", path=env_path)

# Pick the largest .db (most likely the real one)
db_path = max(found, key=lambda f: f.stat().st_size)
log.info("using_db", path=str(db_path))

con = sqlite3.connect(db_path)
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    count = cur.fetchone()[0]
    log.info("table_row_count", table=t, rows=count)

if "module_thresholds" in tables:
    cur.execute("""
        SELECT module_id, threshold, eval_count, correct_count, mean_score
        FROM module_thresholds ORDER BY mean_score ASC
    """)
    rows = cur.fetchall()
    if not rows:
        log.info("module_thresholds_empty")
    else:
        for module_id, threshold, eval_count, correct_count, mean_score in rows:
            log.info(
                "module_threshold",
                module_id=module_id,
                threshold=round(threshold, 3),
                eval_count=eval_count,
                correct_count=correct_count,
                mean_score=round(mean_score or 0, 3),
                low=mean_score is not None and mean_score < 0.4,
            )

decisions_table = next((t for t in tables if "decision" in t.lower()), None)
if decisions_table:
    cur.execute(f"PRAGMA table_info([{decisions_table}])")
    cols = [r[1] for r in cur.fetchall()]
    log.info("decisions_columns", table=decisions_table, columns=cols)

    cur.execute(f"SELECT * FROM [{decisions_table}] ORDER BY rowid DESC LIMIT 20")
    rows = cur.fetchall()
    for row in rows:
        log.info("decision_row", table=decisions_table, data=str(row))
else:
    log.warning("no_decisions_table_found")

con.close()
