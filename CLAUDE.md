# MetaTrade — Project Reference for Claude

**Stack:** Python 3.12+, MetaTrader 5, SQLite/DuckDB, FastAPI, structlog, pydantic-settings  
**Repo:** `metatrade` · entry points in `scripts/` · package root `src/metatrade/`  
**Data source:** MT5 broker (live/paper) or CSV files (backtest)  
**Instruments:** EURUSD (primary), extensible to any MT5 symbol  
**Timeframes active:** M1, M5, M15, M30

---

## Architettura generale

```
Market Data Feed (MT5 / CSV)
        │  Bar chiuse (OHLCV)
        ▼
   BaseRunner.process_bar()
        │
        ├─ 1. ITechnicalModule × N  →  AnalysisSignal[]
        │      (technical_analysis/modules/)
        │
        ├─ 2. AdaptiveThresholdManager  →  filtra segnali sotto soglia per modulo
        │      (consensus/adaptive_threshold.py)
        │
        ├─ 3. ConsensusEngine  →  ConsensusResult  (BUY / SELL / HOLD + score)
        │      (consensus/engine.py)
        │
        ├─ 4. RiskManager  →  RiskDecision  (approved + lot_size / veto)
        │      (risk/manager.py)
        │
        └─ 5. IBrokerAdapter  →  ordine inviato al broker
               (broker/mt5_adapter.py  |  broker/paper_broker.py)

Posizioni aperte  ──►  ExitEngine.evaluate()  (exit_engine/engine.py)  [NUOVO]
```

---

## Mappa moduli

| Package | Responsabilità |
|---|---|
| `core/contracts/` | Tipi immutabili: Bar, Tick, Order, Position, AccountState, AnalysisSignal, ConsensusResult, RiskDecision, Kill Switch |
| `core/enums.py` | SignalDirection, OrderSide, PositionSide, Timeframe, KillSwitchLevel, ConsensusMode |
| `core/event_bus.py` | Async pub/sub interno (EventBus + EventType constants) |
| `core/config_base.py` | BaseConfig (pydantic-settings, .env, frozen) |
| `core/errors.py` | Gerarchia eccezioni: MetaTradeError → ConfigurationError, DataError, ExecutionError, RiskVetoError, ModuleError… |
| `market_data/` | Collettori (MT5Collector, CsvCollector), store DuckDB/SQLite, HistoricalFeed, MockFeed, gap detector, normalizer |
| `technical_analysis/indicators/` | Funzioni pure: EMA, HMA, RSI, MACD, ATR, ADX, Bollinger, Donchian, Keltner, Stochastic, Hurst |
| `technical_analysis/modules/` | 14 moduli: EMA crossover, MultiTF, ADX, Donchian breakout, Market Regime, Keltner Squeeze, Adaptive RSI, Bollinger, Pivot Points, Swing Level, Seasonality, News Calendar, Volatility Regime, Stochastic RSI |
| `technical_analysis/interface.py` | `ITechnicalModule` (module_id, min_bars, analyse()) |
| `consensus/` | ConsensusEngine, 3 motori di voto (SimpleVote, WeightedVote, DynamicVote), MarketAccuracyTracker |
| `consensus/adaptive_threshold.py` | **Soglie adattive per modulo** — si abbassano se giusto, si alzano se sbagliato; persiste su DB |
| `risk/` | RiskManager, PositionSizer (fixed-fractional + vol-scaling), PreTradeChecker, KillSwitchManager, CorrelationFilter |
| `broker/` | IBrokerAdapter, MT5Adapter (pragma: no cover), PaperBrokerAdapter |
| `execution/` | OrderManager (idempotency guard, lifecycle tracking) |
| `runner/` | BaseRunner, BacktestRunner, PaperRunner, LiveRunner, ModuleBuilder, RunnerConfig |
| `ml/` | Feature engineering, Random Forest classifier, walk-forward validation, ModelRegistry |
| `intermarket/` | Correlazioni, lead-lag, feature builder per strumenti correlati |
| `alerting/` | TelegramAlerter (notifiche trade) |
| `observability/` | FastAPI dashboard, TelemetryStore (SQLite), MT5RuntimeReader |
| `exit_engine/` | **[NUOVO]** Motore uscite modulare con reputazione adattiva — vedi sezione dedicata |

---

## Decisioni di design consolidate

### Segnali e consensus
- Ogni `ITechnicalModule.analyse()` restituisce un `AnalysisSignal` con `direction`, `confidence ∈ [0,1]` e `reason`.
- Il `ConsensusEngine` aggrega con **DYNAMIC_VOTE** (default): pesi per modulo aggiustati via EMA sull'accuracy storica.
- Soglia di consenso: **60%** del voto pesato per aprire una posizione.
- Le **soglie adattive** (`AdaptiveThresholdManager`) filtrano i segnali prima del consensus: un modulo troppo impreciso deve avere alta conviction per essere ascoltato.

### Risk management
- **1% di rischio** per trade (configurabile via `RISK_MAX_RISK_PCT`).
- Stop-loss via **Chandelier Exit**: `entry ± ATR(14) × 2.0` — adattivo alla volatilità.
- Take-profit a **2:1 R:R** rispetto allo stop-loss.
- Kill switch a 4 livelli (TRADE_GATE → SESSION_GATE → EMERGENCY_HALT → HARD_KILL).
- Spread filter: blocco se spread > 3 pip.
- Cooldown: 3 candele dopo ogni trade approvato.
- Daily circuit breaker: blocco se perdita giornaliera > 5%.

### Entry
- BUY e SELL entrambi supportati.
- Una sola posizione contemporaneamente per default (`max_open_positions=3` globale, configurable).
- Nessuna restrizione di direzione (long e short).

### Exit (modalità)
| Modalità | Come avviene l'uscita |
|---|---|
| Backtest | `BacktestRunner` monitora SL/TP/trailing stop barra per barra + ExitEngine |
| Paper | `PaperRunner` + `ExitEngine` (SL/TP simulati in-process) |
| Live (MT5) | SL/TP inviati al server MT5 + ExitEngine aggiuntivo per regole avanzate |

### Persistenza
- **DuckDB**: storico candele (OHLCV) per simbolo/timeframe.
- **SQLite** (`TelemetryStore`): sessioni, decisioni, ordini, training runs, artefatti ML, soglie adattive, **reputazione regole di uscita**.
- Separazione netta: la logica di trading non dipende dallo storage.

---

## Exit Engine — Architettura (NUOVO)

### Problema
Stop-loss e take-profit fissi sono insufficienti in mercati con regime variabile. Servono regole di uscita multiple con pesi che evolvono nel tempo in base alle performance.

### Approccio: Weighted Vote con Reputazione EMA

Ogni regola produce un **voto**:
- `EXIT_FULL` → 1.0
- `EXIT_PARTIAL` → 0.5
- `HOLD` → 0.0

**Punteggio aggregato:**
```
score = Σ(w_i × vote_i) / Σ(w_i) × 100
```
- `score >= 60` → chiudi tutto
- `score >= 35` → chiudi parzialmente
- `score < 35` → tieni aperto

**Perché non normalizzo i pesi a 100:**  
Se normalizzi, una sola regola che spara "EXIT" ottiene peso totale anche se ha reputazione bassa. Con i pesi raw [0,100], una regola poco fidata (peso 15) contribuisce poco anche se è l'unica a votare EXIT — comportamento più conservativo e sicuro.

### Regole supportate

| ID regola | Trigger |
|---|---|
| `trailing_stop` | SL che si ratchetta nella direzione favorevole (fisso % o ATR-based) |
| `break_even` | Sposta SL al prezzo di ingresso dopo `X` pip di profitto |
| `time_exit` | Max holding time; opzionale chiusura a fine sessione NY |
| `setup_invalidation` | Rottura livello, incrocio MA, chiusura oltre zona di supporto/resistenza |
| `volatility_exit` | ATR corrente >> ATR al momento dell'ingresso (regime cambiato) |
| `give_back` | Profitto floating sceso di `X%` rispetto al massimo favorevole |
| `partial_exit` | Chiusura parziale a livelli di profitto predefiniti |

### Modello di Reputazione

**Pesi:** `w_i(t) ∈ [5, 95]`, inizializzati a 50 (cold start neutro).

**Update dopo chiusura trade:**
```
contribution_score ∈ [0, 1]
  - Regola ha segnalato EXIT prima che il trade peggiori → score alto
  - Regola ha segnalato EXIT troppo presto (profitto c'era ancora) → score basso
  - Regola in HOLD per tutto: score proporzionale al PnL finale (se pos→ credito; se neg→ penalità)

w_new = clip(α × score × 100 + (1−α) × w_old, min=5, max=95)
α = 0.15 (learning rate — converge in ~20 trade)
```

**Cold start (n < 10 trade):**
```
effective_weight = (n × w + (10−n) × 50) / 10
```
Bilancia verso il prior neutro finché non ci sono dati sufficienti.

**Decay temporale:** Se una regola non viene valutata per > `decay_days` giorni, il peso decade del 5% verso 50 per giorno di inattività (previene dominanza da regole su mercati passati diversi dall'attuale).

**Reputazione per simbolo:** Supportata opzionalmente. Se non ci sono dati symbol-specific, usa reputazione globale.

### Schema configurazione (YAML)
```yaml
exit_engine:
  enabled: true
  close_full_threshold: 60    # score >= 60 → chiudi tutto
  close_partial_threshold: 35 # score >= 35 → chiudi parzialmente
  reputation:
    learning_rate: 0.15
    min_weight: 5
    max_weight: 95
    cold_start_trades: 10
    decay_days: 7
  rules:
    trailing_stop:
      enabled: true
      mode: atr          # atr | fixed_pct | fixed_pips
      atr_period: 14
      atr_mult: 2.0
      fixed_pct: null
      fixed_pips: null
    break_even:
      enabled: true
      trigger_pips: 15   # attiva dopo 15 pip di profitto
      buffer_pips: 2     # SL a entry + 2 pip (non esattamente break-even)
    time_exit:
      enabled: true
      max_holding_bars: 48    # max 48 barre al timeframe configurato
      close_on_session_end: true
      session_end_hour_utc: 21
    setup_invalidation:
      enabled: true
      ma_cross_fast: 9
      ma_cross_slow: 21
      invalidate_on_ma_cross: true
      price_level: null        # override manuale (es. 1.08500)
    volatility_exit:
      enabled: true
      atr_period: 14
      atr_expansion_mult: 2.0  # esci se ATR > 2x ATR all'ingresso
    give_back:
      enabled: true
      give_back_pct: 0.40      # chiudi se profit floats cala del 40% dal picco
    partial_exit:
      enabled: true
      levels:
        - pips: 20
          close_pct: 0.33
        - pips: 40
          close_pct: 0.33
```

### File del modulo

```
src/metatrade/exit_engine/
├── __init__.py
├── contracts.py          # PositionContext, ExitSignal, ExitDecision, ExitAction
├── config.py             # ExitEngineConfig (pydantic, carica da YAML/.env)
├── reputation.py         # ReputationModel
├── persistence.py        # ReputationStore — tabella SQLite rule_reputation
├── engine.py             # ExitEngine (orchestratore)
└── rules/
    ├── __init__.py
    ├── base.py            # IExitRule (interfaccia)
    ├── trailing_stop.py
    ├── break_even.py
    ├── time_exit.py
    ├── setup_invalidation.py
    ├── volatility_exit.py
    ├── give_back.py
    └── partial_exit.py

tests/exit_engine/
├── __init__.py
├── conftest.py
├── test_reputation.py
├── test_engine.py
└── rules/
    ├── __init__.py
    └── test_*.py  (uno per regola)
```

---

## Workflow di sviluppo

```bash
make install          # pip install -e ".[dev]"
make test             # pytest con coverage >= 90%
make test-fast        # pytest -x --no-cov (iterazione rapida)
make lint             # ruff check + format --check
make lint-fix         # ruff fix automatico
make typecheck        # mypy strict

# Paper trading
make paper SYMBOL=EURUSD TIMEFRAME=M15

# Live (REAL MONEY — sempre testare su paper prima)
make live-confirmed SYMBOL=EURUSD TIMEFRAME=M15

# Backtest
make backtest CSV_FILE=data/EURUSD_M15.csv SYMBOL=EURUSD TIMEFRAME=M15 BALANCE=10000

# Dashboard
make run-dashboard    # FastAPI su http://localhost:8000
```

### Convenzioni codice
- Tutti i prezzi: `Decimal` (mai `float` per valori monetari).
- Timestamp: sempre UTC-aware (`datetime` con `tzinfo=timezone.utc`).
- Oggetti di dominio: `@dataclass(frozen=True)` — immutabilità per default.
- Config: eredita da `BaseConfig` (pydantic-settings), prefisso env var per modulo.
- Logging: `structlog` con `get_logger(__name__)`, keyword args strutturati.
- Nessun `assert` in produzione; usa le eccezioni della gerarchia `MetaTradeError`.
- Test coverage minima: 90%.

### Variabili d'ambiente chiave
```
MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_PATH
RISK_MAX_RISK_PCT=0.01
RUNNER_CONSENSUS_THRESHOLD=0.60
RUNNER_MAX_SPREAD_PIPS=3.0
RUNNER_SIGNAL_COOLDOWN_BARS=3
CONSENSUS_THRESHOLD=0.62
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
```

---

## Trade-off e rischi noti

| Area | Rischio | Mitigazione |
|---|---|---|
| Soglie adattive | Overfitting su serie storiche brevi | Cold start neutro (50), clamp [0.45, 0.90] |
| Reputazione regole | Una regola domina dopo pochi trade favorevoli | Clamp pesi [5, 95], decay temporale, cold start con prior 50 |
| Trailing stop | Viene colpito su spike temporanei | ATR-based (non fisso), moltiplicatore ≥ 2.0 |
| Paper vs live | Paper non simula SL/TP hit tra candele | ExitEngine lo fa in-process; live delega a MT5 server |
| Dati mancanti | Feed interrotto → candele saltate | GapDetector in market_data, fallback a dati cached |
| Concorrenza | Runner async + DB SQLite | WAL mode abilitato, single event loop |
| Determinismo | Backtest riproducibile | PRNG con seed fisso (default 42) |
