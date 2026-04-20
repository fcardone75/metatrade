"""GridFS upload/download for bar data (CSV+gzip) and model artifacts (.pkl)."""

from __future__ import annotations

import gzip
import io
from typing import TYPE_CHECKING, Any

from metatrade.core.log import get_logger

if TYPE_CHECKING:
    from metatrade.core.contracts.market import Bar

log = get_logger(__name__)

_DATA_BUCKET = "training_data"
_MODEL_BUCKET = "model_artifacts"


_CSV_DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def _bars_to_csv_gz(bars: list[Bar]) -> bytes:
    """Serialize bars to gzip-compressed CSV bytes.

    Writes the format expected by CsvCollector (default CsvColumnMap):
        datetime,open,high,low,close,volume
        2024-03-01 10:00:00,1.10000,...
    """
    lines = ["datetime,open,high,low,close,volume"]
    for b in bars:
        dt_str = b.timestamp_utc.strftime(_CSV_DT_FORMAT)
        lines.append(f"{dt_str},{b.open},{b.high},{b.low},{b.close},{b.volume}")
    csv_bytes = "\n".join(lines).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(csv_bytes)
    return buf.getvalue()


def _csv_gz_to_bars(data: bytes) -> list[dict[str, Any]]:
    """Deserialize gzip-compressed CSV bytes to list of raw dicts (worker side)."""
    buf = io.BytesIO(data)
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        text = gz.read().decode()
    rows = []
    lines = text.strip().split("\n")
    if not lines or lines == [""]:
        return []
    headers = lines[0].split(",")
    for line in lines[1:]:
        if line:
            values = line.split(",")
            rows.append(dict(zip(headers, values)))
    return rows


class DataTransfer:
    """Upload/download bars and model artifacts via GridFS."""

    def __init__(self, db: Any) -> None:
        try:
            import gridfs  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "pymongo is required for distributed training. "
                "Install it with: pip install 'pymongo[srv]>=4.6'"
            ) from exc
        self._data_fs = gridfs.GridFS(db, collection=_DATA_BUCKET)
        self._model_fs = gridfs.GridFS(db, collection=_MODEL_BUCKET)

    # ── Bar data ──────────────────────────────────────────────────────────────

    def upload_bars(self, job_id: str, symbol: str, timeframe: str, bars: list[Bar]) -> Any:
        filename = f"{symbol}_{timeframe}_{job_id}.csv.gz"
        data = _bars_to_csv_gz(bars)
        file_id = self._data_fs.put(
            data,
            filename=filename,
            job_id=job_id,
            symbol=symbol,
            timeframe=timeframe,
            content_type="application/gzip",
        )
        log.info(
            "data_transfer_bars_uploaded",
            job_id=job_id,
            filename=filename,
            size_kb=len(data) // 1024,
        )
        return file_id

    def download_bars_raw(self, gridfs_id: Any) -> list[dict[str, Any]]:
        """Download and decompress bars CSV, returning list of raw dicts."""
        grid_out = self._data_fs.get(gridfs_id)
        data = grid_out.read()
        rows = _csv_gz_to_bars(data)
        log.info("data_transfer_bars_downloaded", size_kb=len(data) // 1024, rows=len(rows))
        return rows

    def bars_to_csv_file(self, gridfs_id: Any, dest_path: str) -> int:
        """Download and decompress bars CSV to a local file. Returns row count."""
        rows = self.download_bars_raw(gridfs_id)
        headers = list(rows[0].keys()) if rows else []
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(str(row[h]) for h in headers))
        with open(dest_path, "w") as f:
            f.write("\n".join(lines))
        return len(rows)

    # ── Model artifacts ───────────────────────────────────────────────────────

    def upload_model(self, job_id: str, symbol: str, version: str, model_bytes: bytes) -> Any:
        filename = f"{symbol}_{version}.pkl"
        file_id = self._model_fs.put(
            model_bytes,
            filename=filename,
            job_id=job_id,
            symbol=symbol,
            version=version,
            content_type="application/octet-stream",
        )
        log.info(
            "data_transfer_model_uploaded",
            job_id=job_id,
            filename=filename,
            size_kb=len(model_bytes) // 1024,
        )
        return file_id

    def download_model(self, gridfs_id: Any) -> bytes:
        grid_out = self._model_fs.get(gridfs_id)
        data = grid_out.read()
        log.info("data_transfer_model_downloaded", size_kb=len(data) // 1024)
        return data  # type: ignore[return-value]
