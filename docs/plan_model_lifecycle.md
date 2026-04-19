# Piano: Model Lifecycle — Session Filter, Live Tracking, Background Retraining

## Obiettivo

Tre funzionalità correlate che completano il ciclo di vita del modello ML:

1. **Session Gate** — il runner live/paper rispetta la stessa finestra oraria usata nel training.
2. **Live Accuracy Tracker** — traccia in tempo reale quante previsioni del modello attivo risultano corrette.
3. **Background Retraining + Auto-Promotion** — continua ad addestrare in background e sostituisce il modello se il candidato è migliore.

---

## Feature 1 — Session Gate nel Runner

### Problema
Il training usa `session_filter_utc_start/end` per escludere le ore di mercato sottile. Se il runner produce segnali anche fuori da quella finestra, il modello lavora su distribuzione diversa da quella su cui è stato addestrato → degradazione silenziosa.

### Dove intervenire
`BaseRunner.process_bar()` — subito dopo aver ricevuto la barra, prima di invocare i moduli tecnici e il consenso. Se la barra cade fuori sessione: skip completo (nessun segnale, nessun ordine).

### Architettura
```
process_bar(bar, account)
  └─ [NUOVO] session_gate.is_open(bar.timestamp_utc)  →  False → return early (HOLD implicito)
  └─ moduli tecnici → consenso → risk → broker
```

### Nuovi file / modifiche
| File | Tipo | Descrizione |
|---|---|---|
| `src/metatrade/runner/session_gate.py` | NEW | `SessionGate(start_utc, end_utc)` con `is_open(ts: datetime) -> bool` |
| `src/metatrade/runner/config.py` | MODIFY | Aggiunge `session_filter_utc_start: int \| None`, `session_filter_utc_end: int \| None` |
| `src/metatrade/runner/base.py` | MODIFY | Istanzia `SessionGate` dal config e chiama `is_open()` all'inizio di `process_bar()` |
| `tests/runner/test_session_gate.py` | NEW | Test logica normale + overnight wrap |

### Configurazione
```env
RUNNER_SESSION_FILTER_UTC_START=7
RUNNER_SESSION_FILTER_UTC_END=21
```

### Note
- Se entrambi i valori sono `None` (default), nessun filtro applicato (comportamento attuale).
- `SessionGate` è lo stesso algoritmo di `_in_session()` in `walk_forward.py` — estrarre in modulo condiviso `metatrade.core.utils.session`.

---

## Feature 2 — Live Accuracy Tracker

### Problema
Dopo il deployment del modello non sappiamo se continua a funzionare. Vogliamo una percentuale di successo "reale" calcolata sulle previsioni effettivamente fatte durante il live/paper.

### Principio
Per ogni previsione BUY/SELL del modello:
1. Salviamo: timestamp, direzione prevista, prezzo di ingresso.
2. Dopo `forward_bars` barre controlliamo se la direzione era corretta (stesso criterio del training: ATR threshold).
3. Aggiorniamo un contatore rolling.

### Soglia di affidabilità statistica
Con N campioni binari (BUY o SELL, ignoriamo HOLD) e ipotesi nulla 50% (random):

| N campioni | Margine errore (95% CI) | Nota |
|---|---|---|
| 50 | ±7% | Troppo rumoroso |
| 100 | ±5% | Indicativo |
| **200** | **±3.5%** | **Soglia raccomandata** |
| 300 | ±2.8% | Solido |

**Raccomandazione: min_samples = 200 prima che la live accuracy conti per decisioni di switch.**

### Architettura
```
MLModule.analyse(bar)
  └─ predict() → direzione + confidence
  └─ [NUOVO] LiveAccuracyTracker.record_prediction(bar, direction, price)

[N barre dopo] BaseRunner.process_bar()
  └─ [NUOVO] LiveAccuracyTracker.evaluate_pending(bar)
       → per ogni previsione con age >= forward_bars:
           calcola label reale (stesso ATR threshold del training)
           aggiorna contatore hit/miss
           persiste su SQLite
```

### Struttura dati (SQLite — TelemetryStore)
```sql
CREATE TABLE ml_live_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    bar_ts_utc  TEXT NOT NULL,
    direction   INTEGER NOT NULL,   -- 1=BUY, -1=SELL
    confidence  REAL NOT NULL,
    entry_close REAL NOT NULL,
    evaluated   INTEGER DEFAULT 0,  -- 0=pending, 1=done
    correct     INTEGER,            -- NULL=pending, 1=hit, 0=miss
    future_close REAL,
    evaluated_at_utc TEXT
);
```

### Classe principale
```
src/metatrade/ml/live_tracker.py
  LiveAccuracyTracker
    __init__(config: MLConfig, telemetry: TelemetryStore)
    record_prediction(bar: Bar, direction: int, confidence: float, model_version: str)
    evaluate_pending(bars_history: list[Bar]) -> int  # returns n evaluated
    live_accuracy(model_version: str, min_samples: int = 200) -> float | None
    live_accuracy_stats(model_version: str) -> LiveAccuracyStats
```

### LiveAccuracyStats (dataclass)
```python
@dataclass
class LiveAccuracyStats:
    model_version: str
    n_predictions: int      # BUY+SELL fatti
    n_evaluated: int        # quelli con future data disponibile
    n_correct: int
    accuracy: float | None  # None se n_evaluated < min_samples
    is_reliable: bool       # n_evaluated >= min_samples
    confidence_interval: tuple[float, float] | None
```

### Nuovi file
| File | Tipo |
|---|---|
| `src/metatrade/ml/live_tracker.py` | NEW |
| `src/metatrade/ml/contracts.py` (o espandi) | NEW — `LiveAccuracyStats` |
| `tests/ml/test_live_tracker.py` | NEW |
| `src/metatrade/observability/store.py` | MODIFY — aggiunge `record_ml_prediction()`, `mark_prediction_evaluated()`, `get_live_accuracy()` |

---

## Feature 3 — Background Retraining + Auto-Promotion

### Logica di switch

```
Modello attivo (online_model):
  - holdout_accuracy al momento del deploy: H_deploy  (es. 59.72%)
  - live_accuracy corrente: L_live  (es. 55%)
  - min_samples_reached: True/False

Candidato (candidate_model):
  - holdout_accuracy dal training: H_candidate

Regole di switch (in ordine di priorità):
  1. EMERGENCY SWITCH:  L_live < 50%  AND  min_samples_reached
       → switch immediato al candidato migliore in attesa (se esiste)
  2. UPGRADE:  H_candidate > H_deploy  AND  min_samples_reached
       → switch al candidato al prossimo momento sicuro (no posizioni aperte)
  3. NESSUN CANDIDATO disponibile e L_live < 50% → log warning, non tradare
```

### "Momento sicuro" per lo switch
- Nessuna posizione aperta.
- Fuori sessione (oppure a inizio sessione successiva).

### Scheduling del retraining
Il background retraining non deve bloccare il loop live. Due strategie:

**Opzione A — Thread separato (consigliata)**
Il training ML è CPU-bound e dura molti minuti. Un thread daemon con priorità bassa può girare in parallelo senza interferire con il loop async dell'event bus.

**Opzione B — Processo separato**
`scripts/train.py` già esiste. Un processo esterno (cron o subprocess) lancia il training e scrive il modello in `data/models/`. Il runner legge il nuovo snapshot periodicamente.

**Raccomandazione: Opzione B** — più semplice, nessun rischio di contesa su risorse, già infrastruttura pronta.

### Architettura (Opzione B)
```
Runner live
  └─ ModelWatcher (nuovo) — controlla ogni N minuti se esiste un nuovo snapshot
       └─ se sì: legge holdout_accuracy dal manifest
            → applica regole di switch
            → se switch approvato: ModelRegistry.promote(new_version)
            → ricarica MLModule con nuovo modello (senza restart)

Script esterno (cron / task scheduler)
  └─ scripts/train.py --auto-tune --holdout-fraction 0.2 ...
       → scrive nuovo snapshot in data/models/
       → aggiorna registry manifest
```

### Nuovi file
| File | Tipo | Descrizione |
|---|---|---|
| `src/metatrade/ml/model_watcher.py` | NEW | `ModelWatcher` — polling del registry, valuta regole di switch |
| `src/metatrade/ml/promotion_policy.py` | NEW | `PromotionPolicy` — logica switch separata dalla meccanica |
| `src/metatrade/ml/config.py` | MODIFY | Aggiunge `live_accuracy_min_samples`, `switch_threshold_pct`, `watcher_poll_interval_sec` |
| `src/metatrade/runner/base.py` | MODIFY | Inizializza `ModelWatcher`, aggancia `LiveAccuracyTracker` |
| `tests/ml/test_model_watcher.py` | NEW | Test regole di switch |
| `tests/ml/test_promotion_policy.py` | NEW | Test logica upgrade/emergency |

### Configurazione aggiuntiva (MLConfig)
```python
# Live accuracy tracking
live_accuracy_min_samples: int = Field(default=200)   # campioni prima che la % conti
live_accuracy_switch_below: float = Field(default=0.50)  # emergency switch sotto 50%

# Model watcher
model_watcher_enabled: bool = Field(default=False)
model_watcher_poll_sec: int = Field(default=300)      # check ogni 5 minuti
```

---

## Ordine di sviluppo consigliato

```
Phase 1 — Session Gate (½ giornata, basso rischio)
  → Modifica isolata, nessuna dipendenza esterna
  → Test immediati

Phase 2 — Live Accuracy Tracker (1 giornata)
  → Schema SQLite + classe tracker + integrazione in MLModule/BaseRunner
  → Test con barre sintetiche

Phase 3 — Model Watcher + Promotion Policy (1 giornata)
  → Dipende da Phase 2 (usa live_accuracy per le regole di switch)
  → Test con doppio snapshot simulato
```

---

## Decisioni architetturali (confermate)

| # | Domanda | Decisione |
|---|---|---|
| 1 | Il session gate blocca anche la chiusura posizioni già aperte? | No — close/exit sempre permesso indipendentemente dall'orario |
| 2 | Lo switch modello avviene con posizioni aperte? | No — aspetta sempre la chiusura di tutte le posizioni |
| 3 | Notifiche Telegram? | Sì — su apertura/chiusura posizione, PnL, switch modello, emergency alert |
| 4 | Trigger retraining background | Configurabile via env: ogni X ore **oppure** ogni X nuove barre (scegliere uno dei due) |
| 5 | La live accuracy conta solo BUY/SELL o anche HOLD? | Solo BUY/SELL — HOLD non è verificabile con lo stesso criterio ATR |

### Configurazione retraining (env)
```env
# Modalità trigger (scegliere una):
ML_RETRAIN_TRIGGER=hours          # "hours" oppure "bars"
ML_RETRAIN_EVERY_HOURS=6          # usato se trigger=hours
ML_RETRAIN_EVERY_BARS=5000        # usato se trigger=bars

# Switch policy
ML_LIVE_ACCURACY_MIN_SAMPLES=200  # campioni prima che la % conti per lo switch
ML_LIVE_ACCURACY_SWITCH_BELOW=0.50  # emergency switch se accuracy scende sotto questa soglia

# Model watcher
ML_MODEL_WATCHER_ENABLED=false
ML_MODEL_WATCHER_POLL_SEC=120     # quanto spesso il watcher controlla nuovi snapshot

# Holdout richiesto al candidato per essere promosso
ML_CANDIDATE_MIN_HOLDOUT=0.52
```

### Notifiche Telegram (eventi)
| Evento | Messaggio |
|---|---|
| Posizione aperta | Symbol, direzione, lot size, SL, TP, prezzo ingresso |
| Posizione chiusa | Symbol, PnL realizzato, durata, motivo chiusura |
| PnL giornaliero | Riepilogo a fine sessione (o ogni N ore) |
| Switch modello | Vecchio vs nuovo modello, holdout accuracy, live accuracy |
| Emergency (accuracy < threshold) | Alert con statistiche live correnti |
| Retraining completato | Risultato, holdout accuracy, se è candidato o no |
