# ML Module Evolution — Piano PR Incrementale

## Contesto

Il modulo ML attuale (`src/metatrade/ml/`) produce:

- classificazione 3 classi BUY / HOLD / SELL
- confidence = max class probability
- labeling sul close a t+5 con soglia ATR-scaled

L'obiettivo è evolvere il modulo per produrre:

1. **Probabilità calibrate** P(BUY) / P(HOLD) / P(SELL)
2. **Score continuo** di qualità economica attesa (expected value in R-multiple)

---

## Decisione architetturale

### Scelta: Classificatore + Regressore separato (opzione A)

Motivazione:

- Il classificatore esistente ha già interfaccia stabile — non si tocca il nucleo di training
- Il regressore (EV score) si addestra su un target diverso (R-multiple) e può fallire senza impattare il classificatore
- La calibrazione si applica sopra al classificatore esistente senza modificare i pesi del modello
- Ogni componente è testabile e sostituibile in modo indipendente
- Meta-labeling (opzione C) richiede confidenza nel classificatore base già alta; non è il caso attuale

---

## Tabella PR

| # | Nome PR | Obiettivo | File modificati | File nuovi | Classi introdotte | Test da aggiungere | Backward compat | Rischio tecnico | Criterio di merge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `ml/contracts-hardening` | Contratti dati espliciti, separazione artifacts/inference, metadata schema feature, logging migliorato | `classifier.py`, `registry.py`, `module.py` | `contracts.py`, `artifacts.py` | `MlPrediction`, `MlModelArtifacts`, `MlFeatureSchema`, `ArtifactStore` | `test_contracts.py`, `test_artifacts.py` | Piena — `AnalysisSignal` invariato | Basso | Tutti i test esistenti passano; nuovi test ≥ 90% coverage |
| 2 | `ml/calibration` | Probabilità calibrate BUY/HOLD/SELL, `MlPrediction` come DTO primario, calibratore persistito nel registry | `classifier.py`, `module.py`, `registry.py`, `config.py` | `calibration.py`, `prediction.py` | `ProbabilityCalibrator`, `MlPrediction` (con `p_buy/p_hold/p_sell/confidence_margin`) | `test_calibration.py`, `test_prediction.py` | Piena — `direction` e `confidence` invariati, nuovi campi in metadata | Medio — calibrazione può shiftare probabilità | ECE < 0.05 su holdout; esistenti test passano |
| 3 | `ml/expected-value` | Secondo output continuo: EV score in R-multiple, trainer separato, persistito nel registry | `config.py`, `registry.py`, `module.py` | `targets.py`, `trainers.py` | `ContinuousTargetBuilder`, `MlContinuousTarget`, `ExpectedValueTrainer` | `test_targets.py`, `test_trainers.py` | Piena — EV score solo in `AnalysisSignal.metadata` | Medio-alto — rischio leakage, doppio artefatto | R² > 0.05 su OOS; nessun leakage rilevato |
| 4 | `ml/labeling-v2` | Labeling path-aware (triple barrier), spread/cost proxy, walk-forward evaluation rigorosa | `labels.py`, `walk_forward.py`, `config.py` | `evaluation.py`, `labeling/triple_barrier.py` | `TripleBarrierLabeler`, `WalkForwardEvaluator`, `MlEvaluationReport` | `test_evaluation.py`, `test_labels_v2.py` | Piena — `label_bars()` invariata, `label_bars_v2()` additive | Medio — nuova distribuzione classi da documentare | Report OOS generato; distribuzione classi documentata |
| 5 | `ml/consensus-integration` | Esporre `confidence_margin` e `expected_value_score` come campi first-class, enrichment opzionale del segnale | `module.py`, `config.py` | `signal_enricher.py` | `MlSignalEnricher` | `test_signal_enricher.py`, integration test | Piena — logica consensus invariata | Basso — solo metadata additive | Consensus voting invariato; metadata enriched; tutti i test passano |

---

## PR 1 — Checklist Operativa Dettagliata

**Branch:** `feature/ml-contracts-hardening`

**Obiettivo:** introdurre contratti dati espliciti, separare artifacts da inference, aggiungere metadata schema delle feature, migliorare logging. Il comportamento esterno rimane identico.

---

### Step 1 — Setup e analisi baseline

- [ ] Esegui `make test` e verifica che tutti i test esistenti passino (baseline verde)
- [ ] Esegui `make typecheck` e annota gli errori mypy pre-esistenti nel modulo `ml/`
- [ ] Leggi e comprendi completamente:
  - `src/metatrade/ml/classifier.py` — in particolare `predict()`, `serialize()`, `deserialize()`
  - `src/metatrade/ml/registry.py` — `ModelSnapshot`, `_write_to_disk()`, `load_from_disk()`
  - `src/metatrade/ml/module.py` — `MLModule.analyse()` e i casi di fallback
  - `src/metatrade/ml/features.py` — `FeatureVector`, `feature_names()`, `MIN_FEATURE_BARS`
- [ ] Identifica tutti i posti dove i tipi di output del classificatore sono usati come tuple `(direction, confidence)` e dove `ModelSnapshot` viene costruito

---

### Step 2 — Creare `src/metatrade/ml/contracts.py`

- [ ] Crea il file con i seguenti dataclass frozen:

```python
# src/metatrade/ml/contracts.py

@dataclass(frozen=True)
class MlPrediction:
    """Output del classificatore per una singola osservazione."""
    direction: int          # 1=BUY, -1=SELL, 0=HOLD
    confidence: float       # max class probability
    raw_direction: int      # direction prima del remap XGBoost
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class MlFeatureSchema:
    """Descrizione del vettore feature usato per training/inference."""
    feature_names: tuple[str, ...]
    feature_count: int
    min_bars: int
    vector_class: str       # nome della classe FeatureVector usata

@dataclass(frozen=True)
class MlModelArtifacts:
    """Bundle di tutto ciò che serve per caricare e usare un modello."""
    version: str
    symbol: str
    created_at: float
    feature_schema: MlFeatureSchema
    model_bytes: bytes
    metrics: dict[str, object]  # accuracy, n_samples, feature_importances, class_distribution
    tags: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class MlEvaluationReport:
    """Metriche di valutazione di un fold o holdout."""
    fold_id: int
    n_samples: int
    accuracy: float
    buy_precision: float
    sell_precision: float
    hold_precision: float
    buy_recall: float
    sell_recall: float
    class_distribution: dict[int, int]
```

- [ ] Aggiungi type hints completi e docstring utili
- [ ] Non aggiungere dipendenze da altri moduli del package (solo stdlib)
- [ ] Esegui `make typecheck` sul file

---

### Step 3 — Creare `src/metatrade/ml/artifacts.py`

- [ ] Crea `ArtifactStore` — responsabilità: serializzare/deserializzare `MlModelArtifacts` su disco in modo atomico

```python
# src/metatrade/ml/artifacts.py

class ArtifactStore:
    """Persist e carica MlModelArtifacts su disco."""

    def __init__(self, registry_dir: str | Path) -> None: ...

    def save(self, artifacts: MlModelArtifacts) -> Path:
        """Scrive atomicamente .pkl (model_bytes) + .json (metadata) su disco."""
        ...

    def load(self, symbol: str, version: str) -> MlModelArtifacts | None:
        """Carica artifacts da disco. Ritorna None se non esiste."""
        ...

    def list_versions(self, symbol: str) -> list[str]:
        """Lista versioni disponibili su disco per un simbolo, newest first."""
        ...
```

- [ ] Il `.json` deve contenere tutto ciò che è in `MlModelArtifacts` tranne `model_bytes`
- [ ] Usa `os.replace()` per write atomico (già pattern del registry)
- [ ] Gestisci `OSError` e `json.JSONDecodeError` con logging strutturato, non eccezioni propagate
- [ ] Il `.pkl` contiene solo `model_bytes` (già bytes — nessun doppio pickle)

---

### Step 4 — Aggiornare `classifier.py`

- [ ] Aggiungi metodo `predict_raw(feature_vector) -> MlPrediction` che restituisce `MlPrediction` invece di `tuple[int, float]`
  - mantieni `predict()` invariato per backward compat
  - `predict_raw()` estrae `raw_direction` prima del remap e lo espone
- [ ] Aggiungi `feature_schema() -> MlFeatureSchema` che ritorna lo schema delle feature usate nell'ultimo `fit()`
- [ ] Aggiorna `ClassifierMetrics.to_dict() -> dict[str, object]` per facilitare serializzazione in `MlModelArtifacts`
- [ ] Aggiorna `serialize()` / `deserialize()` per includere `_feature_names` e `_inv_label_remap` nei bytes serializzati
  - usa `{"model": model_bytes, "feature_names": [...], "inv_label_remap": {...}}` come dizionario pickle, non solo il modello grezzo
  - mantieni backward compat leggendo anche il vecchio formato (solo il modello sklearn)

---

### Step 5 — Aggiornare `registry.py`

- [ ] Aggiungi campo `feature_schema: MlFeatureSchema | None = None` a `ModelSnapshot`
- [ ] Aggiorna `register()` per accettare e salvare `feature_schema`
- [ ] Aggiorna `_write_to_disk()` per usare `ArtifactStore` invece della logica inline
- [ ] Aggiorna `load_from_disk()` per usare `ArtifactStore.load()` e ricostruire `ModelSnapshot` con tutti i metadati (non più placeholder con accuracy=0.0)
- [ ] Mantieni l'API pubblica invariata: `register()`, `get_active()`, `promote()`, `clear()`

---

### Step 6 — Aggiornare `module.py`

- [ ] Sostituisci `snapshot.classifier.predict(fv)` con `snapshot.classifier.predict_raw(fv)` dove disponibile
- [ ] Espandi `metadata` nell'`AnalysisSignal` ritornato per includere:

```python
metadata={
    "model_version": snapshot.version,
    "raw_direction": prediction.raw_direction,
    "confidence": prediction.confidence,
    "model_available": True,
    "features_ok": True,
    "feature_count": len(fv.to_list()),
}
```

- [ ] Aggiungi log strutturato per ogni predizione (a livello DEBUG, non INFO):

```python
log.debug("ml_prediction", direction=direction.value, confidence=confidence, module=self.module_id)
```

- [ ] Mantieni esattamente lo stesso comportamento per HOLD fallback

---

### Step 7 — Scrivere i test

**`tests/ml/test_contracts.py`**

- [ ] Verifica che `MlPrediction` sia frozen (raise `FrozenInstanceError` se si tenta modifica)
- [ ] Verifica che `MlFeatureSchema` con `feature_count != len(feature_names)` sia rilevato (validazione manuale o `__post_init__`)
- [ ] Verifica serializzazione JSON di `MlModelArtifacts` (tutti i campi eccetto `model_bytes` devono essere JSON-serializzabili)
- [ ] Verifica che `MlEvaluationReport` con buy_precision fuori `[0,1]` sollevi `ValueError` se si aggiunge validazione

**`tests/ml/test_artifacts.py`**

- [ ] `ArtifactStore.save()` → `ArtifactStore.load()` round-trip con verifica campi
- [ ] `load()` con versione inesistente ritorna `None`
- [ ] `save()` è atomico: simula interruzione tra `.pkl.tmp` e `.pkl`, verifica che `load()` non trovi dati corrotti
- [ ] `list_versions()` ritorna solo versioni con sia `.pkl` che `.json` presenti
- [ ] `list_versions()` ordine newest-first

**`tests/ml/test_classifier_contracts.py`** (estensione di test esistenti)

- [ ] `predict_raw()` ritorna `MlPrediction` con campi corretti
- [ ] `serialize()` → `deserialize()` preserva `_feature_names` e `_inv_label_remap`
- [ ] `feature_schema()` prima di `fit()` solleva `RuntimeError`
- [ ] `feature_schema()` dopo `fit()` ha `feature_count == len(feature_names)`

---

### Step 8 — Aggiornare `__init__.py`

- [ ] Aggiungi export di `MlPrediction`, `MlModelArtifacts`, `MlFeatureSchema`, `MlEvaluationReport`, `ArtifactStore` dal package `ml`

---

### Step 9 — Quality gates

- [ ] `make test` — tutti i test esistenti passano
- [ ] `make test-fast` sui soli test nuovi — passa
- [ ] `make typecheck` — nessun errore nuovo introdotto
- [ ] `make lint` — nessun warning nuovo
- [ ] Coverage `tests/ml/` ≥ 90%
- [ ] Nessun `assert` in produzione — usa eccezioni della gerarchia `MetaTradeError` o builtin appropriati
- [ ] Nessun `print()` — solo `log.debug/info/warning`
- [ ] Tutti i valori monetari rimangono `Decimal` (non introdurre `float` per prezzi)

---

### Step 10 — Verifica backward compat

- [ ] Esegui una sessione paper in dry-run e verifica che `MLModule.analyse()` emetta `AnalysisSignal` con stessa struttura di prima
- [ ] Verifica che il `ConsensusEngine` non sia impattato: i campi `direction` e `confidence` su `AnalysisSignal` sono invariati
- [ ] Verifica che `ModelRegistry.load_from_disk()` carichi correttamente modelli salvati prima di questa PR (test di compatibilità con file fixture `.pkl` e `.json` del vecchio formato)

---

### Step 11 — PR Description template

```markdown
## ML/Contracts — Hardening interfacce e contratti dati (PR 1/5)

### Cosa cambia

- Nuovo `contracts.py`: MlPrediction, MlModelArtifacts, MlFeatureSchema, MlEvaluationReport
- Nuovo `artifacts.py`: ArtifactStore con save/load atomico
- `classifier.py`: predict_raw(), feature_schema(), serialize preserva feature_names
- `registry.py`: delega disk I/O ad ArtifactStore, preserva feature_schema
- `module.py`: usa predict_raw(), metadata arricchito, logging strutturato

### Cosa NON cambia

- Interfaccia AnalysisSignal — stessa struttura per il ConsensusEngine
- Comportamento HOLD fallback — identico
- API pubblica ModelRegistry — register/get_active/promote/clear invariati

### Test

- test_contracts.py (nuovo)
- test_artifacts.py (nuovo)
- test_classifier_contracts.py (esteso)
- Tutti i test pre-esistenti passano

### Criterio di merge

- Coverage ml/ >= 90%
- mypy strict senza nuovi errori
- Nessuna regressione nel backtest runner
```

---

## Note architetturali per PR successive

**PR 2 (calibrazione)** potrà aggiungere `ProbabilityCalibrator` come artefatto opzionale in `MlModelArtifacts`, senza cambiare la struttura del registry.

**PR 3 (EV score)** aggiungerà un secondo slot `ev_model_bytes: bytes | None` in `MlModelArtifacts` — backward compat garantita da `None` default.

**PR 4 (labeling v2)** introdurrà `label_bars_v2()` come funzione aggiuntiva in `labels.py` — `label_bars()` invariata.

**PR 5 (consensus integration)** è puramente additive su `AnalysisSignal.metadata` — il voting del `ConsensusEngine` non cambia.
