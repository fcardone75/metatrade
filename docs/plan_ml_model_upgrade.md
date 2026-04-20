# Plan: ML Model Upgrade

## Obiettivo

Migliorare l'accuratezza del classificatore ML sostituendo `HistGradientBoostingClassifier`
con un backend più performante (LightGBM), estendendo il feature set e ampliando
lo spazio di ricerca dell'auto-tune agli iperparametri del modello.

**Problema attuale:** holdout 41–44% su M1, target 58% irraggiungibile con il setup corrente.  
**Causa principale:** HistGBM + 27 features generiche + auto-tune limitato a 2 dimensioni.

---

## Situazione attuale

| Componente | Stato |
|---|---|
| Modello | `HistGradientBoostingClassifier` (scikit-learn) |
| Features | 27 — returns, EMA, RSI, ATR, candele, volume, ora/giorno, ADX, Donchian, Bollinger, Hurst |
| Auto-tune | `forward_bars` × `atr_mult` (2 iperparametri, max 20 trial) |
| Backend | Fisso, non configurabile |
| Holdout M1 | 41–44% (target: 58%) |

---

## Step 1 — Backend pluggabile

**File:** `src/metatrade/ml/classifier.py`

Rendere il backend selezionabile tramite variabile d'ambiente **o** argomento CLI,
senza cambiare l'API pubblica di `MLClassifier`.

### Selezione del backend — precedenza

```
CLI --backend  >  env ML_BACKEND  >  default "histgbm"
```

Il valore CLI sovrascrive sempre l'env, così lo stesso `.env` può definire un default
e ogni run di `train.py` può farne override puntuale.

### Backend supportati

| Valore | Implementazione | Note |
|---|---|---|
| `histgbm` | `HistGradientBoostingClassifier` (sklearn) | default, sempre disponibile |
| `lightgbm` | `LGBMClassifier` (lightgbm) | preferito — leaf-wise, più veloce |
| `xgboost` | `XGBClassifier` (xgboost) | alternativa robusta |

### Argomento CLI in `train.py`

```
python scripts/train.py --source mt5 --symbol EURUSD --backend lightgbm
python scripts/train.py --source mt5 --symbol EURUSD --backend xgboost
python scripts/train.py --source mt5 --symbol EURUSD --backend histgbm
```

Omettendo `--backend`, si usa il valore di `ML_BACKEND` (o `histgbm` se assente).

### Variabile d'ambiente `.env`

```
ML_BACKEND=lightgbm   # histgbm | lightgbm | xgboost
```

### Il backend viene salvato nel modello
Il tag `backend` viene scritto nei metadati del `ModelSnapshot` al momento del salvataggio,
così `ModelRegistry` sa con quale backend è stato trainato ogni versione.

### Comportamento fallback
```
--backend lightgbm → lightgbm non installato → log.warning + fallback a histgbm
```

### Iperparametri nuovi (comuni a tutti i backend)
- `num_leaves` (LightGBM) / `max_leaves` (XGBoost) — complessità foglie
- `learning_rate` — step size boosting
- `min_child_samples` (LightGBM) / `min_child_weight` (XGBoost) — regolarizzazione

---

## Step 2 — Feature set esteso

**File:** `src/metatrade/ml/features.py`

Aggiungere features che migliorano la capacità del modello di catturare contesto
di momentum multi-periodo e struttura di mercato.

### Features da aggiungere (27 → ~35)

| Feature | Calcolo | Motivazione |
|---|---|---|
| `returns_20` | `close[t] / close[t-20] - 1` | Trend intermedio |
| `returns_30` | `close[t] / close[t-30] - 1` | Trend lungo |
| `macd_signal` | MACD signal line normalizzata | Momentum convergenza/divergenza |
| `macd_hist` | MACD histogram normalizzato | Accelerazione momentum |
| `stoch_k` | Stochastic %K | Posizione nel range recente |
| `stoch_d` | Stochastic %D | Media di %K, conferma |
| `rsi5` | RSI(5) scaled [0,1] | RSI veloce, reattivo |
| `rsi21` | RSI(21) scaled [0,1] | RSI lento, trend |

### Invarianti da rispettare
- Tutte le features devono essere dimensionless (return, ratio, scaled)
- `MIN_FEATURE_BARS` va aggiornato (attuale: 55 → stimato: 65)
- `feature_names()` e `to_list()` devono rimanere sincronizzati
- I modelli esistenti su disco diventano incompatibili → versionamento gestito da `ModelRegistry`

---

## Step 3 — Auto-tune esteso

**File:** `scripts/train.py`

Espandere la grid search da 2 a 4–5 dimensioni includendo iperparametri del modello.

### Grid attuale
```python
forward_bars × atr_threshold_mult
```

### Grid proposta
```python
forward_bars × atr_threshold_mult × learning_rate × num_leaves
```

### Vincoli
- Budget fisso: `ML_AUTO_TUNE_MAX_TRIALS` (default 20, invariato)
- La grid viene campionata con strategia a priorità (combinazioni più promettenti prima)
- Cache dei trial già eseguiti mantenuta (già implementata)
- `training_progress.json` già riporta `trials_done/max_trials` — nessuna modifica

---

## Step 4 — Config

**File:** `src/metatrade/ml/config.py`

```python
# Backend
ML_BACKEND: str = "lightgbm"          # histgbm | lightgbm | xgboost

# Iperparametri nuovi
ML_NUM_LEAVES: int = 31               # LightGBM/XGBoost
ML_LEARNING_RATE: float = 0.05        # tutti i backend
ML_MIN_CHILD_SAMPLES: int = 20        # LightGBM (min_child_weight per XGBoost)
```

I campi `max_iter` e `max_depth` restano per compatibilità (usati da HistGBM e come
fallback per gli altri backend).

---

## Ordine di sviluppo

```
Step 1: Config         src/metatrade/ml/config.py
Step 2: Classifier     src/metatrade/ml/classifier.py
Step 3: Features       src/metatrade/ml/features.py
Step 4: Auto-tune      scripts/train.py
Step 5: Test           tests/ml/
```

---

## Cosa NON cambia

| Componente | Motivazione |
|---|---|
| `walk_forward.py` | Agnostico rispetto al backend |
| `registry.py` / `ModelRegistry` | Serializzazione pickle agnostica |
| `module.py` | Usa solo `MLClassifier.predict()` |
| Pipeline runner | Nessun accoppiamento al backend |
| `labels.py` | Labelling invariato |
| Telegram / telemetry | Nessun cambiamento |

---

## Dipendenze opzionali

```
pip install lightgbm    # raccomandato
pip install xgboost     # alternativa
```

Entrambe opzionali: il sistema funziona sempre con scikit-learn come fallback.
Aggiungere a `pyproject.toml` come extra opzionale:

```toml
[project.optional-dependencies]
ml-full = ["lightgbm>=4.0", "xgboost>=2.0"]
```

---

## Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Modelli esistenti incompatibili dopo feature extension | `ModelRegistry` gestisce versioni separate — vecchio modello resta attivo fino a nuovo training |
| LightGBM non disponibile su Windows MT5 server | Fallback automatico a HistGBM |
| Overfitting su M1 con più features | Regolarizzazione più forte via `min_child_samples`, holdout validation invariata |
| Auto-tune più lento con grid 4D | Budget trial fisso, cache dei trial già eseguiti |
| Regressione accuracy su M15/H1 | Walk-forward validation invariata, test separati per timeframe |
