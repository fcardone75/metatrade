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
| `intermarket/` | Correlazioni rolling, currency exposure netta, cluster di rischio, lead-lag, regime intermarket, feature builder per strumenti correlati, portfolio constraints cross-pair |
| `alerting/` | TelegramAlerter (notifiche trade) |
| `observability/` | FastAPI dashboard, TelemetryStore (SQLite), MT5RuntimeReader |
| `exit_engine/` | **[NUOVO]** Motore uscite modulare con reputazione adattiva — vedi sezione dedicata |

---

## Decisioni di design consolidate

## Estensione Intermarket e Correlation-Aware Portfolio (NUOVO)

### Obiettivo
Il sistema non deve più ragionare solo sulla singola coppia/timeframe, ma anche sul **contesto cross-pair**. Nel forex, più segnali apparentemente distinti possono in realtà rappresentare la stessa scommessa macro su una valuta dominante (es. USD forte o debole). Per questo il motore deve stimare correlazioni, esposizione per valuta e rischio aggregato prima di autorizzare nuovi trade.

### Principi operativi
- Le correlazioni **non sono segnali di ingresso primari**: sono un filtro di portafoglio, sizing e de-duplicazione del rischio.
- Le correlazioni devono essere **rolling e regime-aware**: mai hardcoded come verità fissa.
- Il sistema deve distinguere tra:
  - **correlazione tra coppie** (es. EURUSD ↔ GBPUSD)
  - **esposizione per valuta** (es. long EURUSD = short USD + long EUR)
  - **conflitto tra segnali** su coppie che condividono una o più valute.
- Il layer intermarket deve stare **tra ConsensusEngine e RiskManager**, così da influenzare approvazione, veto e lot sizing senza sporcare la logica dei moduli tecnici.

### Nuovo flusso decisionale
```text
Market Data Feed (MT5 / CSV)
        │  Bar chiuse (OHLCV)
        ▼
   BaseRunner.process_bar()
        │
        ├─ 1. ITechnicalModule × N  →  AnalysisSignal[]
        ├─ 2. AdaptiveThresholdManager  →  filtra segnali sotto soglia
        ├─ 3. ConsensusEngine  →  ConsensusResult
        ├─ 4. IntermarketEngine  →  IntermarketDecision
        │      - rolling correlation matrix
        │      - currency exposure map
        │      - cluster risk / duplicate idea detection
        │      - lead-lag / confirmation features
        │
        ├─ 5. RiskManager  →  RiskDecision finale
        │      - approvazione / veto
        │      - lot sizing corretto per rischio correlato
        │
        └─ 6. IBrokerAdapter  →  ordine inviato al broker
```

### Scopo dell'IntermarketEngine
L'`IntermarketEngine` ha quattro responsabilità:

1. **Correlation filter**  
   Calcola una matrice di correlazione rolling tra le coppie abilitate, su rendimenti logaritmici o percentuali, per timeframe configurabile.

2. **Currency exposure accounting**  
   Traduce ogni posizione e ogni segnale in esposizione netta per valuta base/quote. Esempio:
   - long `EURUSD` = `+EUR`, `-USD`
   - short `GBPJPY` = `-GBP`, `+JPY`

3. **Duplicate-risk detection**  
   Se più trade rappresentano la stessa idea economica, il motore li accorpa come rischio condiviso. Esempio:
   - long `EURUSD`
   - long `GBPUSD`
   - short `USDCHF`
   possono essere trattati come esposizione concentrata contro USD.

4. **Intermarket features**  
   Espone feature opzionali ai moduli tecnici/ML, per esempio:
   - forza relativa delle valute
   - dispersione intra-basket
   - divergenza tra coppie correlate
   - lead-lag tra strumenti
   - conferma cross-pair del regime corrente

### Nuovi contratti di dominio
Aggiungere i seguenti concetti al dominio:

- `CurrencyCode` — enum/string value object (`EUR`, `USD`, `GBP`, `JPY`, `CHF`, `AUD`, `NZD`, `CAD`)
- `CurrencyExposure` — esposizione netta per valuta e intensità normalizzata
- `PairCorrelation` — correlazione rolling tra due simboli con finestra, timeframe e timestamp
- `IntermarketSnapshot` — stato corrente di correlazioni, esposizioni, cluster di rischio e regime
- `IntermarketDecision` — output dell'IntermarketEngine con:
  - `approved: bool`
  - `risk_multiplier: Decimal`
  - `exposure_delta_by_ccy: dict[CurrencyCode, Decimal]`
  - `correlated_positions: list[str]`
  - `reason: str`
  - `warnings: tuple[str, ...]`

### Nuovi package / file
```text
src/metatrade/intermarket/
├── __init__.py
├── contracts.py          # CurrencyExposure, PairCorrelation, IntermarketSnapshot, IntermarketDecision
├── config.py             # IntermarketConfig
├── engine.py             # IntermarketEngine
├── correlation.py        # RollingCorrelationService
├── exposure.py           # CurrencyExposureService
├── clustering.py         # RiskClusterBuilder
├── lead_lag.py           # LeadLagAnalyzer (opzionale)
├── regime.py             # IntermarketRegimeDetector (opzionale)
└── persistence.py        # storage metriche intermarket

tests/intermarket/
├── __init__.py
├── test_correlation.py
├── test_exposure.py
├── test_engine.py
└── test_clustering.py
```

### Configurazione proposta (YAML)
```yaml
intermarket:
  enabled: true
  symbols:
    - EURUSD
    - GBPUSD
    - USDCHF
    - USDJPY
    - AUDUSD
    - NZDUSD
    - USDCAD
    - EURGBP
    - EURJPY
    - GBPJPY
  correlation:
    returns_mode: log               # log | pct
    lookback_bars: 200
    timeframe: M15
    min_overlap_bars: 150
    recalc_every_bars: 1
    strong_threshold: 0.80
    medium_threshold: 0.60
  exposure:
    max_net_exposure_per_currency: 2.0
    max_gross_exposure_per_currency: 3.0
    normalize_by_volatility: true
  portfolio:
    max_correlated_positions_per_cluster: 2
    duplicate_idea_abs_corr: 0.75
    block_opposite_signals_on_shared_base: true
    block_new_trade_if_cluster_dd_exceeded: true
    cluster_daily_drawdown_pct: 0.03
  risk_adjustment:
    scale_down_medium_corr: 0.75
    scale_down_high_corr: 0.50
    veto_if_same_idea_and_same_direction: true
    veto_if_currency_exposure_limit_exceeded: true
  features:
    enable_relative_strength: true
    enable_dispersion: true
    enable_lead_lag: false
    enable_regime_flags: true
```

### Regole decisionali iniziali
Regole semplici, conservative e implementabili subito:

1. **Veto per eccesso di esposizione valuta**  
   Se il nuovo trade farebbe superare `max_net_exposure_per_currency`, il trade viene rifiutato.

2. **Riduzione size per rischio correlato**  
   Se il trade è correlato a posizioni già aperte:
   - `abs(corr) >= 0.80` → applica `risk_multiplier = 0.50`
   - `0.60 <= abs(corr) < 0.80` → applica `risk_multiplier = 0.75`
   - sotto `0.60` → nessuna penalità

3. **Blocca trade duplicati**  
   Se una nuova operazione rappresenta la stessa idea economica di un trade già aperto e ha direzione coerente con esso, il sistema può aprire solo se non supera i limiti di cluster e di esposizione. In alternativa, veto diretto.

4. **Blocca conflitti inutili**  
   Se il sistema è già fortemente esposto long su una valuta, un trade nuovo che ne aumenta ancora il rischio deve essere scalato o rifiutato; se invece riduce rischio aggregato, può essere favorito.

5. **Cluster drawdown protection**  
   Le coppie fortemente correlate appartengono allo stesso cluster di rischio. Se il cluster supera il limite di drawdown giornaliero, nessun nuovo trade del cluster viene autorizzato.

### Multi-pair trading: decisione architetturale
Il sistema deve poter lavorare su **più coppie contemporaneamente**, ma non come insieme di strategie indipendenti e scollegate. L'architettura corretta è:

- analisi per simbolo/timeframe
- consensus locale per simbolo
- controllo intermarket globale
- risk manager globale di portafoglio
- execution controllata da vincoli cross-pair

Questa scelta permette di:
- aumentare il numero di opportunità
- migliorare la robustezza del sistema
- sfruttare informazione cross-pair come contesto
- evitare falsa diversificazione

### Uso delle altre coppie come feature informative
Ha senso analizzare anche i modelli delle altre coppie, ma in due modi distinti:

1. **Come strumenti tradabili**  
   Il sistema può generare segnali e aprire trade su più coppie abilitate.

2. **Come strumenti informativi**  
   Anche se una coppia non è tradata in quel momento, può contribuire come feature per capire regime, forza relativa e coerenza del movimento.

Esempi:
- `EURUSD` può essere rafforzato se `GBPUSD` e `AUDUSD` confermano debolezza USD.
- un breakout su `USDJPY` può essere reso meno credibile se il basket USD non conferma.
- una divergenza tra `EURUSD` e `GBPUSD` può segnalare rumore locale o rotazione di forza relativa.

### Integrazione con il RiskManager
Il `RiskManager` resta l'autorità finale sul trade, ma deve ricevere `IntermarketDecision` come input. In particolare:
- `lot_size_final = lot_size_base × risk_multiplier`
- se `approved = false` nell'IntermarketDecision, il trade viene vetoed anche se il consenso locale era BUY/SELL
- il motivo del veto deve essere persistito in telemetry

### Persistenza e osservabilità
Persistire almeno:
- matrice rolling di correlazione per finestra/timeframe
- esposizione netta e lorda per valuta
- cluster di rischio correnti
- veto/riduzioni size causati dall'IntermarketEngine
- PnL per cluster e per valuta

Metriche dashboard consigliate:
- heatmap correlazioni tra coppie abilitate
- esposizione netta per valuta
- trade bloccati per duplicate-risk
- size ridotte per correlazione
- drawdown per cluster di rischio

### Trade-off e rischi noti (intermarket)
| Area | Rischio | Mitigazione |
|---|---|---|
| Correlazioni rolling | Instabili tra regimi di mercato | Finestre rolling + recalc frequente + soglie conservative |
| Troppe coppie | Complessità e overtrading | Universo iniziale ristretto (6-10 pair major) |
| Falsa precisione | Correlazione non implica causalità | Usare come filtro rischio, non come segnale primario |
| Dati asincroni | Barre non allineate tra simboli | Normalizer + min_overlap_bars |
| Esposizione FX | Rischio reale nascosto da pair diverse | CurrencyExposureService centralizzato |

### Raccomandazione di rollout
Implementare in 3 fasi:

1. **Phase 1 — Risk only**  
   Correlazioni rolling + currency exposure + veto/riduzione size.

2. **Phase 2 — Portfolio intelligence**  
   Cluster di rischio, drawdown per cluster, dashboard e metriche.

3. **Phase 3 — Signal enrichment**  
   Feature intermarket per moduli tecnici e modello ML.

Questa progressione riduce il rischio di overengineering e mantiene il comportamento del sistema spiegabile.

### Estensione SL/TP Intelligente e Policy Selection (NUOVO)

#### Obiettivo
Il sistema non deve limitarsi a uno `stop-loss` e a un `take-profit` fissi o derivati da una sola formula. Deve poter scegliere in modo **contestuale, spiegabile e data-driven** un **profilo iniziale di uscita** e poi adattare la gestione dell'uscita in modo dinamico mentre il trade evolve.

L'approccio raccomandato **non** è:
- predire direttamente due numeri continui (`SL`, `TP`) come output black-box
- bloccare sempre il trade con un `TP` rigido all'ingresso

L'approccio corretto è:
- generare un insieme finito di candidati plausibili di **exit profile**
- selezionare il profilo iniziale migliore con modello ML o fallback deterministico
- demandare all'`ExitEngine` la gestione dinamica del trade durante la sua vita

#### Principio architetturale
Il problema va modellato come:

**"Dato il contesto al tempo `t0`, quale profilo iniziale di uscita massimizza l'outcome atteso corretto per il rischio?"**

non come:

**"Predici direttamente il numero ottimo di pip per SL e TP e lasciali fissi fino alla chiusura"**

Questo approccio è più robusto, spiegabile, testabile e molto meno soggetto a overfitting.

#### Posizionamento nel flusso decisionale
Il selettore del profilo di uscita deve intervenire **dopo** il consenso locale/intermarket e **prima** dell'invio dell'ordine al broker. L'`ExitEngine` resta poi responsabile della gestione dinamica post-entry.

```text
Market Data Feed (MT5 / CSV)
        │
        ▼
   BaseRunner.process_bar()
        │
        ├─ 1. ITechnicalModule × N  → AnalysisSignal[]
        ├─ 2. AdaptiveThresholdManager
        ├─ 3. ConsensusEngine → ConsensusResult
        ├─ 4. IntermarketEngine → IntermarketDecision
        ├─ 5. RiskManager → RiskDecision
        ├─ 6. ExitProfileCandidateGenerator → ExitProfileCandidate[]
        ├─ 7. ExitProfileSelector → SelectedExitProfile
        ├─ 8. IBrokerAdapter → ordine con SL iniziale e TP/target mode coerenti
        └─ 9. ExitEngine.evaluate() → gestione dinamica post-entry



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
INTERMARKET_ENABLED=true
INTERMARKET_SYMBOLS=EURUSD,GBPUSD,USDCHF,USDJPY,AUDUSD,NZDUSD,USDCAD,EURGBP,EURJPY,GBPJPY
INTERMARKET_CORR_LOOKBACK_BARS=200
INTERMARKET_CORR_TIMEFRAME=M15
INTERMARKET_STRONG_THRESHOLD=0.80
INTERMARKET_MEDIUM_THRESHOLD=0.60
INTERMARKET_MAX_NET_EXPOSURE_PER_CURRENCY=2.0
INTERMARKET_MAX_GROSS_EXPOSURE_PER_CURRENCY=3.0
INTERMARKET_MAX_CORRELATED_POSITIONS_PER_CLUSTER=2
INTERMARKET_DUPLICATE_IDEA_ABS_CORR=0.75
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
| Correlazioni cross-pair | Falsa diversificazione e rischio duplicato | IntermarketEngine + exposure map + cluster limits |
| Multi-pair execution | Overtrading e conflitti tra idee | Veto per duplicate-risk, risk multiplier e cap per cluster |
