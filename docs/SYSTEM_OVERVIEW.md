# MetaTrade — Come funziona il sistema

> Documento tecnico completo: pipeline, algoritmi, parametri e formule.

---

## Indice

1. [Panoramica architetturale](#1-panoramica-architetturale)
2. [Pipeline decisionale completa](#2-pipeline-decisionale-completa)
3. [I moduli di analisi tecnica](#3-i-moduli-di-analisi-tecnica)
4. [Consensus Engine](#4-consensus-engine)
5. [Adaptive Threshold Manager](#5-adaptive-threshold-manager)
6. [ML Classifier](#6-ml-classifier)
7. [Risk Manager](#7-risk-manager)
8. [Exit Engine](#8-exit-engine)
9. [Intermarket Engine](#9-intermarket-engine)
10. [Broker Adapters](#10-broker-adapters)
11. [Runner modes (Backtest / Paper / Live)](#11-runner-modes)
12. [Esempio di ciclo completo](#12-esempio-di-ciclo-completo)
13. [Tabella parametri di default](#13-tabella-parametri-di-default)

---

## 1. Panoramica architetturale

MetaTrade è un sistema di trading algoritmico modulare per MetaTrader 5. La sua filosofia è che nessun singolo indicatore è affidabile: il sistema combina 14 moduli tecnici, un classificatore ML, un motore di consenso pesato adattivo, un gestore del rischio con kill switch multi-livello e un motore di uscita con 7 regole a reputazione variabile.

**Stack tecnologico**: Python 3.12+, MT5, DuckDB, SQLite, FastAPI, scikit-learn / XGBoost / CatBoost / LightGBM.

**Simboli attivi**: EURUSD (primario), estensibile a qualsiasi simbolo MT5.

**Timeframe attivi**: M1, M5, M15, M30.

```
Componenti principali
─────────────────────────────────────────────────────────────
14 Moduli tecnici          → AnalysisSignal (direction, confidence)
AdaptiveThresholdManager   → filtra i segnali deboli per modulo
ConsensusEngine            → voto pesato adattivo → BUY / SELL / HOLD
IntermarketEngine          → correlazione cross-pair, esposizione valuta
RiskManager                → sizing, SL, kill switch, spread filter
ExitEngine                 → 7 regole di uscita con reputazione EMA
ML Classifier              → modello gradient boosting (4 backend)
IBrokerAdapter             → MT5 live | PaperBroker | Backtest sim
TelemetryStore             → SQLite, dashboard FastAPI
```

---

## 2. Pipeline decisionale completa

Ogni barra chiusa attraversa questa pipeline in ordine sequenziale:

```
BAR OHLCV (M1/M5/M15/M30 @ timestamp_utc)
        │
        ▼
[1] MODULI TECNICI (14 moduli in parallelo)
        │  → AnalysisSignal(direction, confidence ∈ [0,1], reason, metadata)
        ▼
[2] ADAPTIVE THRESHOLD FILTER (per-modulo)
        │  → scarta segnali con confidence < soglia_modulo
        ▼
[3] CONSENSUS ENGINE (DynamicVote)
        │  → buy_pct, sell_pct, hold_pct
        │  → ConsensusResult(direction, aggregate_confidence)
        │  → actionable se max(buy_pct, sell_pct) ≥ 60%
        ▼
[4] ML CLASSIFIER (opzionale)
        │  → AnalysisSignal aggiuntivo con pred_class e pred_confidence
        ▼
[5] INTERMARKET ENGINE
        │  → verifica correlazioni, esposizione valuta, cluster drawdown
        │  → IntermarketDecision(approved, risk_multiplier)
        ▼
[6] RISK MANAGER
        │  → kill switch check
        │  → pre-trade checks (spread, margin, cooldown)
        │  → position sizing (fixed-fractional + vol-scaling)
        │  → SL via Chandelier Exit (entry ± ATR(14) × 2.0)
        │  → TP a 2:1 R:R rispetto allo SL
        │  → RiskDecision(approved, lot_size, sl, tp)
        ▼
[7] BROKER ADAPTER
        │  → MT5Adapter (live) | PaperBroker | BacktestSim
        │  → ordine inviato con SL e TP
        ▼
[8] POSIZIONE APERTA → per ogni barra successiva:
        │
        ├─ EXIT ENGINE (7 regole + reputazione)
        │    → ExitDecision(HOLD | CLOSE_PARTIAL | CLOSE_FULL)
        │
        └─ FEEDBACK LOOP (su chiusura)
             → aggiorna pesi moduli (DynamicVote)
             → aggiorna soglie moduli (AdaptiveThreshold)
             → aggiorna reputazione regole (ExitEngine)
```

---

## 3. I moduli di analisi tecnica

Ogni modulo implementa `ITechnicalModule.analyse(bars) → AnalysisSignal`. Lavorano in parallelo su ogni barra.

---

### 3.1 RSI Module

**Indicatore**: RSI a periodo 14.

**Logica**:
- RSI < 30 (oversold) → **BUY**
- RSI > 70 (overbought) → **SELL**
- RSI ∈ [30, 70] → **HOLD**

**Confidence** (formula depth-based):
```
depth_buy  = (30 − RSI) / 30          se RSI < 30
depth_sell = (RSI − 70) / 30          se RSI > 70
confidence = min(0.90, 0.50 + depth × 0.50)
HOLD:      confidence = 0.60
```

---

### 3.2 Adaptive RSI Module

Evoluzione del RSI classico: le soglie oversold/overbought si adattano dinamicamente al regime di mercato misurato dall'ADX.

**Soglie adattive**:
| ADX | Regime | Oversold | Overbought |
|-----|--------|----------|------------|
| < 20 | Ranging | 30 | 70 |
| 20–35 | Transizione | interpolazione lineare | interpolazione lineare |
| > 35 | Trending | 20 | 80 |

**Interpolazione**: `t = (ADX − 20) / 15`, thresholds = start + t × (end − start).

**Rationale**: In trend forte, il RSI rimane in zona estrema a lungo; soglie più strette evitano falsi segnali di mean-reversion.

**Confidence cap**: 0.88.

---

### 3.3 MACD Module

**Indicatori**: EMA(12) − EMA(26) = MACD line; EMA(MACD, 9) = signal line; histogram = MACD − signal.

**Segnali** (crossover):
- MACD crosses above signal → **BUY**
- MACD crosses below signal → **SELL**
- No crossover → **HOLD**

**Confidence**:
```
confidence = min(0.85, 0.50 + |histogram| × 500)
HOLD: confidence = 0.55
```

Minimo barre richieste: 36 (26 + 9 + 1).

---

### 3.4 ATR Module

**Scopo**: Misura la volatilità corrente. Non è direzionale — emette sempre HOLD ma espone metadata usati da RiskManager e ExitEngine.

**Metadata output**:
```
atr:     ATR(14) corrente
avg_atr: media mobile ATR su 50 barre
sl_buy:  entry − ATR × 2.0   (SL suggerito per BUY)
sl_sell: entry + ATR × 2.0   (SL suggerito per SELL)
```

**Confidence**: 0.75 se alta volatilità (ATR > 1.5 × avg_ATR), altrimenti 0.60.

---

### 3.5 ADX Module

**Indicatori**: ADX(14), +DI(14), −DI(14).

**Soglia trend**: ADX ≥ 25.

**Segnali**:
- ADX ≥ 25 AND +DI > −DI → **BUY**
- ADX ≥ 25 AND −DI > +DI → **SELL**
- ADX < 25 → **HOLD** (mercato senza trend)

**Confidence**:
```
confidence = min(0.90, 0.55 + (ADX − 25) / 100)
HOLD:       confidence = 0.50
```

---

### 3.6 Bollinger Bands Module

**Indicatori**: BB(20, 2σ) = MA(20) ± 2 × StdDev(20).

**Logica** (mean-reversion):
- Chiusura precedente ≤ lower band AND chiusura corrente > lower band → **BUY** (rimbalzo oversold)
- Chiusura precedente ≥ upper band AND chiusura corrente < upper band → **SELL** (rifiuto overbought)
- Prezzo dentro le bande → **HOLD**

**Confidence**: `min(0.85, 0.55 + depth_ratio × 0.30)`
dove `depth_ratio` = profondità relativa del tocco alla banda.

---

### 3.7 EMA Crossover Module (v2: slow = HMA)

**Indicatori**: EMA(9) fast, HMA(21) slow (Hull Moving Average riduce il lag ~50% vs EMA classica).

**Segnali** (crossover):
- EMA(9) crosses above HMA(21) → **BUY**
- EMA(9) crosses below HMA(21) → **SELL**
- No crossover → **HOLD**

**Confidence**:
```
gap_pct    = |EMA(9) − HMA(21)| / prezzo
confidence = min(0.85, gap_pct / 0.001)
```

Minimo barre: ~30 (21 + √21 + 2).

---

### 3.8 Stochastic RSI Module

**Indicatori**: StochRSI = Stoch(RSI(14), 14) smoothed con K(3) e D(3).

**Zone**: oversold < 0.20, overbought > 0.80.

**Segnali** (crossover in zona estrema):
- K crosses above D AND min(K, D) < 0.20 → **BUY** (bullish dalla zona oversold)
- K crosses below D AND max(K, D) > 0.80 → **SELL** (bearish dalla zona overbought)
- Altrimenti → **HOLD**

**Confidence**: `min(0.85, 0.55 + depth_ratio × 0.30)`

Minimo barre: 38.

---

### 3.9 Keltner Squeeze Module

**Indicatori**:
- Bollinger(20, 2σ)
- Keltner Channel: EMA(20) ± ATR(10) × 1.5

**Logica** (volatility compression → breakout):
- BB interamente dentro KC → **squeeze** (HOLD, mercato in attesa)
- Close > KC upper → **BUY** (breakout rialzista post-squeeze)
- Close < KC lower → **SELL** (breakout ribassista post-squeeze)

**Confidence**: `min(0.85, 0.60 + depth × 0.25)`

---

### 3.10 Donchian Breakout Module

**Indicatori**: Donchian(20) = [min_low, max_high] delle ultime 20 barre.

**Segnali** (trend-following classico, stile Turtle):
- Close > previous upper → **BUY** (nuovo massimo a 20 barre)
- Close < previous lower → **SELL** (nuovo minimo a 20 barre)
- Dentro il range → **HOLD**

**Confidence**: `min(0.82, 0.60 + depth × 0.20)`

Il confronto è fatto su `bars[-2]` (barra precedente) per evitare look-ahead bias.

---

### 3.11 Market Regime Module

Il modulo più composito: classifica il regime corrente e orienta l'interpretazione di tutti gli altri.

**Indicatori**: ADX(14), ATR ratio (corrente / media), Esponente di Hurst su finestra 100.

**Regimi**:
| Condizione | Regime | Segnale |
|------------|--------|---------|
| ADX ≥ 25 AND +DI > −DI AND ATR_ratio ≤ 2.0 | TRENDING_UP | BUY |
| ADX ≥ 25 AND −DI > +DI AND ATR_ratio ≤ 2.0 | TRENDING_DOWN | SELL |
| ADX < 25 AND ATR_ratio ≤ 2.0 | RANGING | HOLD |
| ATR_ratio > 2.0 (override) | VOLATILE | HOLD |

**Confidence base**: `min(0.88, 0.55 + (ADX − 25) / 100)`

**Hurst Exponent adjustment**:
```
H > 0.55 (persistente / trending):
    regime trending → +0.04, regime ranging → −0.04

H < 0.45 (anti-persistente / mean-reverting):
    regime ranging → +0.04, regime trending → −0.04

0.45 ≤ H ≤ 0.55 (random walk): nessun aggiustamento
```

---

### 3.12 Pivot Points Module

**Formula** (metodo floor, ricavata dai dati precedenti della sessione):
```
P  = (H + L + C) / 3
R1 = 2P − L        S1 = 2P − H
R2 = P + (H − L)   S2 = P − (H − L)
```

**Touch distance**: `ATR(14) × 0.5` (adattiva alla volatilità).

**Segnali**:
- Prezzo vicino a S1/S2 → **BUY** (supporto)
- Prezzo vicino a R1/R2 → **SELL** (resistenza)
- Altrimenti → **HOLD**

**Confidence**: S1=0.62, S2=0.72, R1=0.62, R2=0.72 + proximity bonus fino a +0.10.

---

### 3.13 Multi-Timeframe Module

**Logica**: analizza il trend su timeframe superiori e lo usa come filtro direzionale.

**Resample**: M15 → H1 (factor 4), M15 → H4 (factor 16).

**Segnali** (EMA fast/slow su HTF):
- EMA(9) > EMA(21) su HTF → **BUY** (uptrend superiore)
- EMA(9) < EMA(21) su HTF → **SELL** (downtrend superiore)
- EMA piatte → **HOLD**

**Confidence**: `min(0.85, 0.60 + depth × 0.25)`

---

### 3.14 Swing Level Module

**Rilevamento swing** (lookback = 5 barre default):
- Swing high: `high[i] ≥ high[i±j]` per j ∈ [1, lookback]
- Swing low: `low[i] ≤ low[i±j]` per j ∈ [1, lookback]

**Touch distance**: `ATR(14) × 0.5`

**Segnali** (supporto/resistenza dinamici):
- Prezzo vicino a swing low → **BUY**
- Prezzo vicino a swing high → **SELL**
- Altrimenti → **HOLD**

**Confidence**: `min(0.82, 0.58 + proximity × 0.22)`

Conserva gli ultimi 5 livelli swing.

---

### 3.15 Seasonality Module

**Logica**: qualità della sessione di trading + momentum a breve termine.

**Qualità sessione** (basata su UTC):
| Finestra | Qualità | Comportamento |
|----------|---------|---------------|
| 00–07, 21+, Ven >15:00, weekend | POOR | HOLD sempre |
| 07–13, 17–21 Lun–Gio | GOOD | BUY/SELL solo se momentum forte |
| 13–17 Mar–Gio (London/NY overlap) | EXCELLENT | BUY/SELL anche con momentum moderato |

**Momentum**: `(close[-1] − close[-4]) / close[-4]`, soglia 0.0003 (≈3 pips EURUSD).

---

### 3.16 News Calendar Module

**Funzione**: blocca i trade nelle finestre ad alto rischio di eventi macroeconomici.

**Eventi hardcoded**: FOMC (19:00 UTC), BCE (13:15 UTC), NFP (primo venerdì del mese, 13:30 UTC).

**Finestra di esclusione**: ±60 minuti dall'evento.

**Segnale**: sempre HOLD nelle finestre di rischio (confidence 0.50).

**Opzionale**: refresh live via API Finnhub.

---

## 4. Consensus Engine

Aggrega i segnali dei moduli (filtrati dalle soglie adattive) in una decisione unica.

### 4.1 Tre schemi di voto

**SimpleVote** (pesi uniformi):
```
score(direction) = Σ confidence(s)  per s.direction = direction
total = Σ score(direction)
pct(direction) = score(direction) / total
```

**WeightedVote** (pesi fissi per modulo, configurati manualmente).

**DynamicVote** (default — pesi adattivi basati sull'accuracy storica):
```
accuracy_ema[mod] = α × outcome_score + (1−α) × accuracy_ema[mod]
    α = 0.10,  inizializzato a 0.50

weight[mod] = max(min_weight, base_weight × 2 × accuracy_ema[mod])
    min_weight = 0.10,  base_weight = 1.0
```

**Effetto dei pesi dinamici**:
- Modulo sempre corretto → accuracy → 1.0 → weight → 2.0 (doppio peso)
- Modulo sempre sbagliato → accuracy → 0.0 → weight → 0.10 (quasi irrilevante)
- Modulo sconosciuto → accuracy = 0.50 → weight = 1.0 (neutro)

### 4.2 Decisione finale

```
buy_pct  = weighted_score(BUY)  / total_weighted_score
sell_pct = weighted_score(SELL) / total_weighted_score

Actionable:
  buy_pct  ≥ 0.60  →  ConsensusResult(direction=BUY)
  sell_pct ≥ 0.60  →  ConsensusResult(direction=SELL)
  altrimenti       →  ConsensusResult(direction=HOLD)

aggregate_confidence = media pesata delle confidence dei voti vincenti
```

La soglia del 60% è configurabile via `RUNNER_CONSENSUS_THRESHOLD`.

---

## 5. Adaptive Threshold Manager

Filtra i segnali **prima** del consenso: ogni modulo ha una soglia di confidence minima che evolve con le sue performance.

### 5.1 Regola di aggiornamento

```
delta     = −alpha × (outcome_score − 0.5) × 2
new_thr   = clip(old_thr + delta,  min=0.45,  max=0.90)
alpha     = 0.01  (1% step per valutazione)
```

**Interpretazione**:
- Modulo corretto (score=1.0): delta = −0.01 → soglia scende → include anche segnali con confidence moderata
- Modulo neutro (score=0.5): delta = 0 → soglia stabile
- Modulo sbagliato (score=0.0): delta = +0.01 → soglia sale → il modulo viene ascoltato solo con alta conviction

**Soglia default**: 0.60. Range: [0.45, 0.90].

La soglia per modulo viene persistita in SQLite e sopravvive ai riavvii.

---

## 6. ML Classifier

### 6.1 Labeling (generazione target)

```python
label_bars(bars, forward_bars=5, atr_threshold_mult=0.8–1.5):
    future_return = (close[i + forward_bars] − close[i]) / close[i]
    threshold     = ATR(14)[i] / close[i] × atr_threshold_mult

    label = +1   se future_return > +threshold   (BUY)
    label = −1   se future_return < −threshold   (SELL)
    label =  0   altrimenti                       (HOLD)

    Ultime forward_bars barre: label = None (dati futuri insufficienti)
```

L'ATR adatta automaticamente la soglia alla volatilità: in mercati calmi bastano movimenti più piccoli per essere classificati BUY/SELL; in mercati volatili serve un movimento più ampio.

### 6.2 Feature engineering

Il `FeatureVector` include circa 40 feature:

| Categoria | Feature |
|-----------|---------|
| Prezzo | OHLC, OHLC ratios (shadow, body) |
| Return | 1-bar, 5-bar, 20-bar |
| Volatilità | ATR(14), realized_vol(20) |
| Momentum | RSI(14), MACD histogram, Stochastic K/D |
| Trend | EMA(9), EMA(21), distanza da MA |
| Struttura | ADX(14), +DI, −DI |
| Regime | classe regime da MarketRegimeModule |
| Temporale | ora del giorno, giorno della settimana |

### 6.3 Walk-forward training

Per evitare look-ahead bias, il training usa una finestra scorrevole:

```
train_window = 2000 barre  (configurabile)
test_window  = 500  barre
step         = 250  barre

Iterazione:
  fold 0: train [0:2000],   test [2000:2500]
  fold 1: train [0:2250],   test [2250:2750]
  fold 2: train [0:2500],   test [2500:3000]
  ...
  holdout: ultime 20% barre (valutazione finale out-of-sample)
```

Il modello migliore (miglior holdout accuracy) viene promosso nel registry.

### 6.4 Auto-tune

Griglia di iperparametri testata automaticamente:

| Parametro | Valori testati |
|-----------|---------------|
| forward_bars | 5, 10, 15, 20, 25 |
| atr_threshold_mult | 0.4, 0.5, 0.6, 0.7, 0.8 |

Ogni combinazione produce un walk-forward completo. Il trial migliore (per signal precision su BUY+SELL) viene selezionato.

**Adaptive loop**: se nessun trial supera la soglia target, ripete con complessità ridotta (depth=4→3→2→1, iter=150→100→75→50→30) fino a 20 tentativi totali.

### 6.5 Backend supportati

| Backend | Libreria | Classe weight | GPU |
|---------|----------|--------------|-----|
| `histgbm` | scikit-learn HistGradientBoosting | `class_weight="balanced"` | No |
| `lightgbm` | LightGBM | nativo | Sì (CUDA) |
| `xgboost` | XGBoost | sample_weight | Sì (CUDA) |
| `catboost` | CatBoost | `auto_class_weights="Balanced"` | Sì (CUDA) |

> **Nota XGBoost**: richiede label non negative. Internamente il sistema rimappa −1/0/1 → 0/1/2 in `fit()` e ripristina i valori originali in `predict()`.

### 6.6 Integrazione nel flusso

Il classificatore ML produce un `AnalysisSignal` aggiuntivo che entra nel Consensus Engine con il suo peso dinamico. Non bypassa il consenso: è un modulo tra gli altri.

```python
direction, confidence = classifier.predict(feature_vector)
# direction ∈ {-1, 0, +1}
# confidence = max(predict_proba(X))   → probabilità della classe vincente
```

---

## 7. Risk Manager

### 7.1 Architettura

```
RiskManager
├── KillSwitchManager    (4 livelli di escalation)
├── PreTradeChecker      (spread, margin, cooldown, posizioni aperte)
└── PositionSizer        (fixed-fractional + volatility scaling)
```

### 7.2 Kill Switch (4 livelli)

| Livello | Effetto | Reset |
|---------|---------|-------|
| NONE | Nessuna restrizione | — |
| TRADE_GATE | Blocca il prossimo singolo trade | Auto-reset dopo |
| SESSION_GATE | Blocca tutti i nuovi trade della sessione | Manuale |
| EMERGENCY_HALT | Chiude le posizioni aperte e ferma il sistema | Manuale |
| HARD_KILL | Override operatore (richiede `force_reset()`) | Solo manuale |

**Auto-kill**: se il drawdown intraday supera il 5% del balance, scatta automaticamente `SESSION_GATE`.

I livelli possono solo aumentare o resettare a NONE; non si può retrocedere a un livello intermedio.

### 7.3 Pre-trade checks

Prima di ogni ordine il sistema verifica:
- Kill switch non in GATE/HALT/HARD
- Spread corrente < `spread_filter_pips` (default 3.0 pip su M1, 2.0 su H1)
- Free margin sufficiente per la posizione richiesta
- Max posizioni aperte non superato (default 3 globale)
- Cooldown trascorso dall'ultimo trade (default 3 candele)

### 7.4 Stop-Loss: Chandelier Exit

```
sl_distance = ATR(14) × 2.0

BUY:  sl_price = entry_price − sl_distance
SELL: sl_price = entry_price + sl_distance
```

Il moltiplicatore 2.0 dell'ATR è la costante chiave: abbastanza lontano da non essere colpito da rumore, abbastanza vicino da limitare la perdita massima.

### 7.5 Take-Profit: 2:1 Risk/Reward

```
tp_distance = sl_distance × risk_reward_ratio   (default 2.0)

BUY:  tp_price = entry_price + tp_distance
SELL: tp_price = entry_price − tp_distance
```

Per un trade con SL a 36 pip, il TP è a 72 pip.

### 7.6 Position Sizing

**Passo 1 — Fixed-fractional**:
```
risk_amount  = balance × max_risk_pct           (default 1%)
sl_pips      = |entry − sl_price| / pip_size
lot_size     = risk_amount / (sl_pips × pip_value_per_lot)
```

**Passo 2 — Volatility scaling** (opzionale):
```
current_atr_pips = ATR(14) / pip_size
vol_mult         = min(1.0, target_atr_pips / current_atr_pips)
vol_mult         = max(vol_scaling_min_mult, vol_mult)   (floor 30%)
lot_size         = lot_size × vol_mult
```

**Passo 3 — Arrotondamento**:
```
lot_size = floor(lot_size / lot_step) × lot_step
lot_size = clip(lot_size, min_lot=0.01, max_lot=10.0)
```

**Passo 4 — Intermarket multiplier**:
```
lot_size = lot_size × intermarket_risk_multiplier   (range 0.3–1.5)
```

**Esempio** (EURUSD, conto USD 10.000, rischio 1%):
```
risk_amount = 10.000 × 0.01 = 100 USD
sl          = entry − ATR×2 → 20 pip
lot_size    = 100 / (20 × 10) = 0.50 lotti

Se ATR = 25 pip, target = 20:
  vol_mult = 20/25 = 0.80
  lot_size = 0.50 × 0.80 = 0.40 lotti
```

### 7.7 Parametri di default

| Parametro | Valore |
|-----------|--------|
| `max_risk_pct` | 0.01 (1%) |
| `risk_reward_ratio` | 2.0 |
| `atr_period` | 14 |
| `atr_multiplier` | 2.0 |
| `spread_filter_pips` | 3.0 (M1), 2.0 (H1) |
| `signal_cooldown_bars` | 3 |
| `auto_kill_drawdown_pct` | 0.05 (5%) |
| `vol_target_atr_pips` | 20 |
| `vol_scaling_min_mult` | 0.30 |

---

## 8. Exit Engine

### 8.1 Architettura

```
ExitEngine (orchestratore)
├── 7 IExitRule (valutate ogni barra su ogni posizione aperta)
│   ├── TrailingStopRule
│   ├── BreakEvenRule
│   ├── TimeExitRule
│   ├── SetupInvalidationRule
│   ├── VolatilityExitRule
│   ├── GiveBackRule
│   └── PartialExitRule
└── ReputationModel (pesi per regola, aggiornati su ogni chiusura)
```

### 8.2 Voto aggregato

Ogni regola emette un voto:
```
EXIT_FULL    → vote_value = 1.0
EXIT_PARTIAL → vote_value = 0.5
HOLD         → vote_value = 0.0
```

Formula aggregata:
```
score = Σ(w_i × vote_value_i × confidence_i) / Σ(w_i)  × 100

score ≥ 60   → CLOSE_FULL    (default close_full_threshold)
score ≥ 35   → CLOSE_PARTIAL (default close_partial_threshold)
score <  35  → HOLD
```

**Perché i pesi non vengono normalizzati a 100**: se normalizzati, una sola regola con peso basso che vota EXIT otterrebbe peso totale. Con pesi raw, una regola con reputazione bassa (peso 15) contribuisce poco anche se è l'unica a votare EXIT — comportamento più conservativo.

### 8.3 Le 7 regole

**TrailingStopRule**
```
Modalità atr (default):
  sl_trail = max_favorable_price − ATR(14) × atr_mult   (BUY)
  sl_trail = min_favorable_price + ATR(14) × atr_mult   (SELL)
  atr_mult = 2.0

Se price ≤ sl_trail (BUY) o price ≥ sl_trail (SELL):
  → EXIT_FULL, confidence = 0.95
Altrimenti:
  → suggest_sl = sl_trail (SL aggiornato per ratcheting)
```

**BreakEvenRule**
```
Se unrealized_pips ≥ trigger_pips (default 15):
  → HOLD, ma suggest_sl = entry ± buffer_pips (default 2)
  // Non chiude; sposta solo lo SL in zona profitto minimo garantito
```

**TimeExitRule**
```
Se bars_held ≥ max_holding_bars (default 48 barre al TF configurato):
  → EXIT_FULL, confidence = 0.85

Se bar.hour_utc ≥ session_end_hour (default 21):
  → EXIT_FULL, confidence = 0.80
```

**SetupInvalidationRule**
```
Riconosce che la tesi del trade è venuta meno:
  BUY:  close < entry − 2×ATR  (prezzo sotto il livello di ingresso con margine)
  SELL: close > entry + 2×ATR

→ EXIT_FULL, confidence = 0.90
```

**VolatilityExitRule**
```
Se ATR_corrente > ATR_ingresso × atr_expansion_mult (default 2.0):
  // Regime di volatilità cambiato: rischio di slippage elevato
  → EXIT_FULL, confidence = 0.75
```

**GiveBackRule**
```
Protegge il profitto maturato:
  Se unrealized_pnl < peak_pnl × (1 − give_back_pct)  (default 40%)
  // Prezzo ha restituito il 40% del massimo profitto floating
  → EXIT_FULL, confidence = 0.85
```

**PartialExitRule**
```
Chiusura parziale a livelli predefiniti:
  Level 1: +20 pip → chiude 33% della posizione
  Level 2: +40 pip → chiude un altro 33%
  // Il 34% restante va a target o viene gestito dalle altre regole
→ EXIT_PARTIAL, close_pct dalla config
```

### 8.4 Modello di reputazione

**Pesi**: `w_i ∈ [5, 95]`, inizializzati a 50 (cold start neutro).

**Score post-chiusura** (calcolato per ogni regola):
```
Trade perdente (final_pnl ≤ 0):
  score = sigmoid(theoretical_pnl_at_signal × 0.05)
  // Uscita prima che la perdita peggiorasse → score > 0.5

Trade vincente (final_pnl > 0):
  Se theoretical_pnl ≥ final_pnl:    score = 0.70  (buona uscita)
  Se 0 < theoretical_pnl < final_pnl: score = 0.50 × (theoretical/final)  (uscita prematura)
  Se theoretical_pnl ≤ 0:            score = 0.20  (uscita in perdita su trade vincente)

HOLD per tutto il trade:
  final_pnl > 0 → score = 0.60
  final_pnl ≤ 0 → score = 0.35
```

**Aggiornamento peso**:
```
w_new = clip(α × score × 100 + (1−α) × w_old,  min=5,  max=95)
α = 0.15 (learning rate — converge in ~20 trade)
```

**Cold start** (n < 10 trade):
```
effective_weight = (n/10) × w_learned + (1 − n/10) × 50
// Blending verso il prior neutro finché i dati sono scarsi
```

**Decay temporale** (inattività > 7 giorni):
```
w_decayed = w + 0.05 × (50 − w)   per ogni giorno inattivo
// Evita che una regola domini dopo un periodo di mercato diverso
```

La reputazione è **per simbolo** se ci sono abbastanza trade; altrimenti usa la reputazione globale.

---

## 9. Intermarket Engine

### 9.1 Scopo

Evita che trade apparentemente indipendenti rappresentino in realtà la stessa scommessa macro (es. long EURUSD + long GBPUSD + short USDCHF = tre trade long USD concentrati).

### 9.2 Componenti

**RollingCorrelationService**: matrice di correlazione rolling tra tutti i simboli abilitati, calcolata su rendimenti logaritmici, finestra di 200 barre, aggiornata ogni barra.

**CurrencyExposureService**: traduce ogni posizione in esposizione netta per valuta:
- Long EURUSD = +EUR, −USD
- Short GBPJPY = −GBP, +JPY

**RiskClusterBuilder**: raggruppa simboli con correlazione assoluta > 0.75 nello stesso cluster di rischio.

### 9.3 Regole decisionali

| Condizione | Azione |
|------------|--------|
| `abs(corr) ≥ 0.80` | `risk_multiplier = 0.50` |
| `0.60 ≤ abs(corr) < 0.80` | `risk_multiplier = 0.75` |
| `abs(corr) < 0.60` | `risk_multiplier = 1.00` |
| Esposizione netta valuta > limite | Veto del trade |
| Cluster in drawdown > 3% daily | Veto per tutti i trade del cluster |
| Trade duplica idea già aperta | Veto o scaling |

### 9.4 Output: IntermarketDecision

```python
IntermarketDecision(
    approved           = True / False,
    risk_multiplier    = Decimal("0.75"),      # applicato al lot_size
    exposure_delta_by_ccy = {"EUR": +1.0, "USD": -1.0},
    correlated_positions  = ["GBPUSD_long_01"],
    reason             = "high correlation EURUSD↔GBPUSD",
    warnings           = ("USD net exposure 1.8/2.0",),
)
```

---

## 10. Broker Adapters

### 10.1 Interfaccia comune (`IBrokerAdapter`)

```python
connect() / disconnect() / is_connected()
send_order(order)         → SUBMITTED + broker_order_id
cancel_order(order_id)
get_account()             → AccountState(balance, equity, margin_free)
get_current_price(symbol) → (bid, ask)
modify_sl_tp(ticket, new_sl, new_tp)   # solo MT5
get_open_positions(symbol)
```

### 10.2 MT5Adapter (live e paper account MT5)

Wrapper attorno alle API MetaTrader5. Usa `mt5.order_send()` con:
- `TRADE_ACTION_DEAL` per ordini a mercato
- `TRADE_ACTION_SLTP` per modificare SL/TP di posizioni aperte
- `magic_number` per distinguere gli ordini del bot da quelli manuali

Il `retcode=10009` (DONE) indica fill avvenuto; `10010` (PARTIAL) viene gestito.

### 10.3 PaperBrokerAdapter (simulazione in-process)

Simula fill senza connessione reale:
- Fill al bar successivo: BUY = ask = mid + spread/2, SELL = bid = mid − spread/2
- Tiene traccia di balance, equity, margin in memoria
- Supporta commissioni configurabili per lotto

### 10.4 BacktestSim (implicito nel BacktestRunner)

Il fill è al prezzo di apertura della barra successiva più uno slippage casuale (uniforme, sempre avverso). SL e TP vengono controllati sui prezzi high/low di ogni barra successiva.

---

## 11. Runner modes

### 11.1 BacktestRunner

**Input**: lista storica di `Bar` da CSV o DuckDB.

**Differenze vs live**:
- Nessuna connessione broker
- Fill simulato (barra successiva + slippage)
- SL/TP controllati sui prezzi H/L di ogni barra
- ExitEngine opera in-process su ogni barra
- Deterministico (seed RNG fisso, default 42)
- Output: `BacktestResult` con equity curve, drawdown, trade log

### 11.2 PaperRunner

**Input**: feed di barre live (MT5 o feed esterno) ma senza soldi reali.

**Differenze vs live**:
- PaperBrokerAdapter invece di MT5Adapter
- Account simulato in-memory (non persistente tra sessioni)
- SL/TP monitorati dall'ExitEngine in-process (MT5 non li vede)
- Utile per validare la strategia prima del live

### 11.3 LiveRunner

**Input**: feed di barre live da MT5.

**Caratteristiche aggiuntive**:
- SL e TP inviati direttamente al server MT5 (vengono eseguiti anche se il bot è offline)
- ExitEngine opera in aggiunta per regole avanzate (trailing, giveback, time exit...)
- OrderManager garantisce idempotenza (no double-submit)
- Kill switch verificato prima di ogni ordine
- Alert Telegram su trade, veto, errori

---

## 12. Esempio di ciclo completo

```
Timestamp: 14:00 UTC  |  Simbolo: EURUSD  |  ATR(14) = 0.00180  |  Entry = 1.08500

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODULI TECNICI:
  adaptive_rsi: BUY  conf=0.72  (ADX=28, trending, RSI non overbought)
  macd:         BUY  conf=0.68  (crossover positivo, histogram=+0.00012)
  adx:          BUY  conf=0.70  (ADX=28 > 25, +DI > −DI)
  market_regime:BUY  conf=0.76  (TRENDING_UP, Hurst=0.58 → +0.04)
  ema_crossover:BUY  conf=0.74  (EMA9 crossed above HMA21)
  multi_tf_h4:  BUY  conf=0.72  (H4 uptrend confermato)
  seasonality:  BUY  conf=0.71  (13-17 UTC, excellent session)
  stoch_rsi:    SELL conf=0.65  (K < D in zona overbought)
  bollinger:    HOLD conf=0.50
  ...altri HOLD...

ADAPTIVE THRESHOLD FILTER:
  stoch_rsi soglia = 0.68  →  0.65 < 0.68  →  SCARTATO
  adaptive_rsi soglia = 0.64  →  0.72 ≥ 0.64  →  PASSA
  ...
  Accettati: 7 segnali BUY, 0 SELL, 1 HOLD

CONSENSUS (DynamicVote):
  buy_pct  = 5.84 / 6.59 = 88.6%  ≥ 60%  → BUY actionable ✓
  aggregate_confidence = 0.722

INTERMARKET:
  Già aperto: GBPUSD long (corr = 0.88 con EURUSD)
  risk_multiplier = 0.85  (alta correlazione)

RISK MANAGER:
  spread = 1.5 pip  <  3.0 pip  ✓
  kill switch = NONE  ✓
  sl  = 1.08500 − 0.00180×2 = 1.08140   (36 pip)
  tp  = 1.08500 + 0.0036×2  = 1.09220   (72 pip, 2:1 R:R)
  lot = (10.000 × 0.01) / (36 × 10) × 0.85 = 0.40 lotti

ORDINE INVIATO:
  BUY 0.40 EURUSD @ 1.08500  SL=1.08140  TP=1.09220

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GESTIONE POSIZIONE (bar successive):

  Bar +5:  BreakEvenRule   → suggest_sl = 1.08530 (entry + 2 pip)
  Bar +12: TrailingStopRule → sl ratchetato a 1.08650
  Bar +18: PartialExitRule  → chiude 33% a 1.09000 (+50 pip parziale)
  Bar +24: TP colpito        → chiude 67% rimanente a 1.09220 (+72 pip)

  P&L totale: 0.40 lot × 64 pip (media ponderata) × 10 = +256 USD (+2.56%)

FEEDBACK LOOP:
  market_regime: accuracy_ema = 0.1×1.0 + 0.9×0.75 = 0.775  → peso ↑
  stoch_rsi (filtrato): nessun aggiornamento
  TrailingStopRule: peso +  (ha contribuito con sl ratchetato)
  PartialExitRule: peso +   (ha catturato profitto parziale)
  Adaptive threshold:
    ema_crossover soglia: 0.58 → 0.571  (corretto, soglia scende)
```

---

## 13. Tabella parametri di default

| Componente | Parametro | Valore |
|-----------|-----------|--------|
| **RSI** | period | 14 |
| | oversold / overbought | 30 / 70 |
| **ADX** | period | 14 |
| | trend threshold | 25 |
| **ATR** | period | 14 |
| | sl_multiplier | 2.0 |
| **EMA Crossover** | fast | EMA(9) |
| | slow | HMA(21) |
| **Bollinger** | period / std | 20 / 2.0 |
| **Donchian** | period | 20 |
| **Keltner** | ema period | 20 |
| | atr period / mult | 10 / 1.5 |
| **Stoch RSI** | rsi / stoch period | 14 / 14 |
| | K / D smooth | 3 / 3 |
| | oversold / overbought | 0.20 / 0.80 |
| **Market Regime** | adx threshold | 25 |
| | high_vol_mult | 2.0 |
| | hurst window | 100 |
| **News Calendar** | window | ±60 min |
| **Consensus** | threshold | 60% |
| | mode | DynamicVote |
| **DynamicVote** | alpha (EMA) | 0.10 |
| | min weight | 0.10 |
| **Adaptive Threshold** | default | 0.60 |
| | alpha | 0.01 |
| | range | [0.45, 0.90] |
| **Risk** | max_risk_pct | 1% |
| | risk_reward_ratio | 2.0 |
| | spread_filter | 3.0 pip |
| | cooldown | 3 candele |
| | daily drawdown kill | 5% |
| | vol_target_atr | 20 pip |
| | vol_min_mult | 0.30 |
| **Exit Engine** | close_full score | ≥ 60 |
| | close_partial score | ≥ 35 |
| **TrailingStop** | atr_mult | 2.0 |
| **BreakEven** | trigger | 15 pip |
| | buffer | 2 pip |
| **TimeExit** | max_holding_bars | 48 |
| | session end UTC | 21:00 |
| **GiveBack** | give_back_pct | 40% |
| **PartialExit** | level 1 | +20 pip → chiude 33% |
| | level 2 | +40 pip → chiude 33% |
| **Reputation** | learning_rate | 0.15 |
| | initial weight | 50 |
| | range | [5, 95] |
| | cold start | 10 trade |
| | decay_days | 7 |
| **ML Labels** | forward_bars | 5 |
| | atr_threshold_mult | 0.8 (auto-tune) |
| **Walk-Forward** | train_window | 2000 barre |
| | test_window | 500 barre |
| | step | 250 barre |
| | holdout_fraction | 20% |
| **Intermarket** | strong corr threshold | 0.80 → ×0.50 |
| | medium corr threshold | 0.60 → ×0.75 |
| | max net exposure / ccy | 2.0 |
| | cluster daily dd | 3% |
