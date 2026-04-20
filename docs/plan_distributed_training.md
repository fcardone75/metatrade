# Plan: Distributed ML Training via MongoDB Atlas

## Obiettivo

Separare il training ML dall'esecuzione live su due macchine fisiche distinte:

- **Master** — server lento con MT5, fa live/paper trading, scarica i dati, triggera il training
- **Worker** — macchina potente senza MT5, fa solo training, carica il modello risultante

La comunicazione tra le due macchine avviene esclusivamente via **MongoDB Atlas**
(nessuna VPN, nessuna porta aperta, nessun servizio aggiuntivo da gestire).

Quando `ML_TRAIN_MONGO=false` il sistema funziona esattamente come oggi (nessun impatto).

---

## Architettura

```
MASTER (MT5 server)                        WORKER (macchina potente)
──────────────────                         ────────────────────────
LiveRunner / PaperRunner                   scripts/train_worker.py
     │                                          │
     │  /retrain (Telegram)                     │  polling MongoDB ogni N sec
     │  → scheduler.trigger_remote()            │  (solo se idle)
     │                                          │
     ▼                                          ▼
 MongoJobQueue.push_job()          ←──  MongoJobQueue.poll_pending()
   training_jobs: {pending}                     │
   + upload barre → GridFS                      │ status: running
                                                │
                                                │  train.py (subprocess locale)
                                                │  → progress → MongoDB
                                                │
                                          training_jobs: {completed}
                                          + upload model → GridFS
                                                │
     ▼                                          │
 ModelDownloader.watch()  ◄─────────────────────┘
 → download .pkl da GridFS
 → ModelRegistry.register() + promote()
 → runner ricarica modello

TELEGRAM /training
  - se training locale (vecchio) → legge adaptive_progress.json
  - se training remoto (mongo) → legge training_jobs + progress da MongoDB
  - in entrambi i casi risponde solo il master
```

---

## Configurazione

### Variabili d'ambiente

```env
# ── Distributed training toggle ───────────────────────────────────────────────
ML_TRAIN_MONGO=false            # true = abilita distributed training via MongoDB

# ── Ruolo macchina ────────────────────────────────────────────────────────────
ML_MT5_MASTER=true              # true = questa macchina è il master (MT5 + Telegram)
                                # false = questa macchina è il worker (solo training)

# ── MongoDB Atlas ─────────────────────────────────────────────────────────────
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=metatrade

# ── Worker tuning ─────────────────────────────────────────────────────────────
ML_WORKER_POLL_SEC=30           # intervallo polling MongoDB quando idle (default 30s)
```

### Regola di comportamento

| `ML_TRAIN_MONGO` | `ML_MT5_MASTER` | Comportamento |
|---|---|---|
| `false` | qualsiasi | Training locale subprocess (comportamento attuale) |
| `true` | `true` | Master: triggera job su MongoDB, monitora, scarica modello |
| `true` | `false` | Worker: polling MongoDB, esegue training, carica modello |

---

## Struttura MongoDB

### Collection `training_jobs`

```json
{
  "_id": "job_eurusd_m1_20260420_1200",
  "status": "pending",        // pending | running | completed | failed | cancelled
  "symbol": "EURUSD",
  "timeframe": "M1",
  "backend": "lightgbm",
  "train_args": ["--symbol", "EURUSD", "--timeframe", "M1", "..."],
  "data_gridfs_id": "...",    // ObjectId del CSV in GridFS
  "model_gridfs_id": null,    // ObjectId del .pkl caricato dal worker
  "model_version": null,      // versione del modello (v20260420_1200_M1)
  "holdout_accuracy": null,
  "triggered_by": "manual",   // manual | scheduled
  "created_at": "2026-04-20T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "error": null
}
```

### Collection `training_progress`

Stessa struttura di `adaptive_progress.json` (già nota al sistema), scritta dal worker,
letta dal master. Un documento per job attivo.

```json
{
  "job_id": "job_eurusd_m1_20260420_1200",
  "status": "running",
  "symbol": "EURUSD",
  "timeframe": "M1",
  "target": 0.58,
  "attempts_done": 3,
  "max_attempts": 20,
  "best_holdout": 0.5512,
  "started_at_utc": "2026-04-20T12:01:00Z",
  "updated_at_utc": "2026-04-20T13:15:00Z",
  "attempts": [...],
  "fold_data": {...}           // training_progress.json corrente (nested)
}
```

### GridFS bucket `training_data`

- Bars CSV compressi (gzip) — fino a 50–100MB
- Filename: `{symbol}_{timeframe}_{job_id}.csv.gz`

### GridFS bucket `model_artifacts`

- Model .pkl serializzato
- Filename: `{symbol}_{version}.pkl`

---

## Nuovi file

```
src/metatrade/ml/distributed/
├── __init__.py
├── config.py             # MongoTrainConfig (pydantic-settings, prefisso MONGO_)
├── job_queue.py          # MongoJobQueue — push, poll, update status
├── data_transfer.py      # upload/download barre (GridFS) + model .pkl
├── progress_store.py     # MongoProgressStore — write (worker) / read (master)
└── worker_daemon.py      # polling loop + lancio training + upload modello

scripts/
└── train_worker.py       # entry point worker: avvia il daemon di polling
```

---

## Flusso dettagliato

### Master: trigger training

```python
# retrain_scheduler.py — quando ML_TRAIN_MONGO=true
def trigger_remote(self, symbol, timeframe, bars):
    job_id = MongoJobQueue.push_job(symbol, timeframe, train_args)
    DataTransfer.upload_bars(job_id, bars)          # → GridFS
    log.info("remote_training_job_queued", job_id=job_id)
```

`is_training` sul master diventa `True` finché il job risulta `running` o `pending` su MongoDB.

### Worker: polling e training

```python
# worker_daemon.py
while True:
    job = MongoJobQueue.poll_pending()    # atomic findAndModify → running
    if job is None:
        sleep(ML_WORKER_POLL_SEC)
        continue

    bars = DataTransfer.download_bars(job)
    MongoProgressStore.init(job)

    # lancia train.py come subprocess locale
    # train.py scrive adaptive_progress.json localmente →
    # un thread parallelo sincronizza il JSON su MongoDB ogni 10s
    run_training(job, bars)

    model_path = find_best_model(job)
    if model_path:
        DataTransfer.upload_model(job, model_path)
        MongoJobQueue.complete(job, model_path)
    else:
        MongoJobQueue.fail(job, "no model met target")
```

### Progress sync (worker → MongoDB)

Un thread separato nel worker legge `adaptive_progress.json` ogni 10 secondi
e lo scrive su `training_progress` in MongoDB. Questo mantiene la logica di
`train.py` invariata: continua a scrivere i suoi file JSON locali.

### Master: /training su Telegram

```python
# base.py _cmd_training — quando ML_TRAIN_MONGO=true
def _cmd_training(self):
    job = MongoJobQueue.get_active_job(symbol)
    if job is None:
        return "⏸ Nessun training remoto in corso."
    progress = MongoProgressStore.read(job["_id"])
    return _format_adaptive_progress(progress, progress.get("fold_data"))
```

### Master: download modello a completamento

```python
# worker_daemon-side: dopo upload modello, job status → completed
# master-side: ModelDownloader controlla MongoDB ogni N sec (stesso polling)
# quando trova job completed:
pkl_bytes = DataTransfer.download_model(job)
registry.register_from_bytes(pkl_bytes, ...)
registry.promote(version)
alerter.alert_retrain_complete(...)
```

---

## Telegram: chi risponde

| Situazione | Chi risponde a `/training` |
|---|---|
| Training locale in corso (ML_TRAIN_MONGO=false) | Master (legge file JSON locale) |
| Training remoto in corso (ML_TRAIN_MONGO=true) | Master (legge MongoDB) |
| Worker non ha Telegram | Worker non gestisce comandi Telegram |

Il master è sempre l'unico interlocutore Telegram.
Il worker non ha `TelegramAlerter` né `CommandHandler`.

---

## Ordine di sviluppo

```
Step 1: Config            src/metatrade/ml/distributed/config.py
Step 2: JobQueue          src/metatrade/ml/distributed/job_queue.py
Step 3: DataTransfer      src/metatrade/ml/distributed/data_transfer.py
Step 4: ProgressStore     src/metatrade/ml/distributed/progress_store.py
Step 5: WorkerDaemon      src/metatrade/ml/distributed/worker_daemon.py
Step 6: Integrazione      retrain_scheduler.py + base.py (_cmd_training)
Step 7: Entry point       scripts/train_worker.py
Step 8: Test              tests/ml/distributed/
```

---

## Dipendenze nuove

```toml
[project.optional-dependencies]
distributed = [
    "pymongo[srv]>=4.6",     # MongoDB Atlas driver + DNS seedlist
    "motor>=3.3",            # async MongoDB (per il daemon worker)
]
```

Opzionali: il sistema funziona senza se `ML_TRAIN_MONGO=false`.

---

## Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| MongoDB non raggiungibile | Job non parte, log errore, retry automatico al prossimo slot |
| Worker crash durante training | Job resta `running`; master rileva timeout (`updated_at` > 30min) e resetta a `pending` |
| Modello non caricato correttamente | Verifica holdout prima di promuovere; vecchio modello resta attivo |
| Upload bars lento su Atlas | Compressione gzip prima dell'upload; max ~10-20MB compressi |
| Più worker attivi (futuro) | `findAndModify` atomico garantisce che un solo worker prende ogni job |
| Sicurezza credenziali | `MONGO_URI` solo in `.env`, mai loggato |
