.PHONY: help install install-worker install-live start-worker test test-fast lint lint-fix typecheck clean \
        train train-csv train-mt5 train-mt5-mtf fetch-massive backfill-massive-mongo backfill-massive-batches train-massive \
        backtest backtest-no-ml backtest-mt5 \
        walk-forward walk-forward-no-ml walk-forward-mt5 \
        paper paper-no-ml \
        live live-confirmed live-no-ml \
        run-dashboard stop-services \
        kill kill-emergency reset-kill \
        venv-ensure

# ── Virtualenv: stesso effetto di "source .venv/bin/activate" per ogni ricetta ─
# Make avvia una shell nuova per ogni riga; esportare PATH e VIRTUAL_ENV qui
# garantisce che python/pip/pytest nelle ricette usino il venv se esiste.
VENV     ?= .venv
VENV_ABS := $(abspath $(VENV))
ifneq ($(wildcard $(VENV)/bin/python),)
  export PATH := $(VENV_ABS)/bin:$(PATH)
  export VIRTUAL_ENV := $(VENV_ABS)
endif

# ── Configurable defaults (override on the command line) ──────────────────────
SYMBOL    ?= EURUSD
TIMEFRAME ?= H1
BARS      ?= 30000
CSV_FILE  ?= data/$(SYMBOL)_$(TIMEFRAME).csv
MODEL_DIR ?= data/models
BALANCE   ?= 10000

# Paper / live runners (optional; see targets paper, live-confirmed)
WARMUP_BARS          ?= 200
POLL_INTERVAL        ?=
MODEL_VERSION        ?=
# Imposta NO_D1=1 o NO_NEWS=1 per abilitare i flag corrispondenti
NO_D1                ?=
NO_NEWS              ?=
FINNHUB_KEY          ?=
MAX_RISK_PCT         ?= 0.01
DAILY_LOSS_LIMIT_PCT ?= 0.05
KILL_DRAWDOWN_PCT    ?= 0.10
MAX_OPEN_POSITIONS   ?= 1
# Argomenti extra verso run_live.py / run_paper.py (es. --magic-number 20240602)
LIVE_EXTRA           ?=
PAPER_EXTRA          ?=

# ─────────────────────────────────────────────────────────────────────────────

venv-ensure:   ## Verifica .venv e mostra l'interprete Python usato dalle ricette make
	@test -f "$(VENV)/bin/activate" || { \
		printf '%s\n' "Manca $(VENV). Crea con: python3 -m venv $(VENV) && make install"; \
		exit 1; \
	}
	@py="$$(command -v python)"; \
	if [ "$$py" = "$(VENV_ABS)/bin/python" ]; then \
		printf '%s\n' "OK: ricette make usano il venv ($$py)"; \
	else \
		printf '%s\n' "Attenzione: python=$$py (atteso $(VENV_ABS)/bin/python)"; \
		exit 1; \
	fi

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Configurable variables (pass on command line):"
	@echo "    SYMBOL=$(SYMBOL)  TIMEFRAME=$(TIMEFRAME)  BARS=$(BARS)"
	@echo "    CSV_FILE=$(CSV_FILE)  MODEL_DIR=$(MODEL_DIR)  BALANCE=$(BALANCE)"
	@echo "    FOLDS=$(FOLDS)  TRAIN_PCT=$(TRAIN_PCT)  TIMEFRAMES=$(TIMEFRAMES)  PROMOTE_TF=$(PROMOTE_TF)"
	@echo "    Paper/live: WARMUP_BARS=$(WARMUP_BARS)  POLL_INTERVAL=$(POLL_INTERVAL)  MODEL_VERSION=$(MODEL_VERSION)"
	@echo "    NO_D1=$(NO_D1)  NO_NEWS=$(NO_NEWS)  MAX_RISK_PCT=$(MAX_RISK_PCT)  LIVE_EXTRA=...  PAPER_EXTRA=..."

# ── Development ───────────────────────────────────────────────────────────────

install:   ## Install package + dev dependencies
	pip install -e ".[dev]"

install-worker:   ## Install all dependencies needed on the ML worker (Linux)
	pip install -e ".[worker,dev]"

install-live:   ## Install all dependencies needed on the live master (Windows)
	pip install -e ".[live,distributed,dev]"

start-worker: venv-ensure ## clear Python cache, reinstall, then start the ML worker daemon
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	python scripts/train_worker.py

test:   ## Run full test suite with coverage
	pytest

test-fast:   ## Run tests without coverage (faster iteration)
	pytest -x --no-cov

lint:   ## Check code style (ruff)
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:   ## Auto-fix code style issues
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:   ## Run static type checking (mypy)
	mypy src/

clean:   ## Remove build artefacts, caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

# ── Training ──────────────────────────────────────────────────────────────────

train:   ## Train ML model from CSV  (CSV_FILE=data/EURUSD_H1.csv SYMBOL=EURUSD)
	python scripts/train.py \
		--source csv \
		--file $(CSV_FILE) \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR)

train-csv:   ## Same as train (explicit alias)
	$(MAKE) train

train-mt5:   ## Train ML model fetching data live from MT5  (SYMBOL=EURUSD BARS=30000)
	python scripts/train.py \
		--source mt5 \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--bars $(BARS) \
		--model-dir $(MODEL_DIR)

TIMEFRAMES ?= M5,M15,M30

train-mt5-mtf:   ## Train su M5, M15, M30 da MT5 (TIMEFRAMES= override, PROMOTE_TF= timeframe attivo)
	python scripts/train.py \
		--source mt5 \
		--symbol $(SYMBOL) \
		--timeframes $(TIMEFRAMES) \
		--bars $(BARS) \
		--model-dir $(MODEL_DIR) \
		$(if $(PROMOTE_TF),--promote-timeframe $(PROMOTE_TF),)

# REFRESH=1 forza riscarico Massive (fetch e train-massive)
REFRESH ?=
BACKFILL_FROM ?=
BACKFILL_TO ?=
SYMBOL_LIMIT ?=
SYMBOL_OFFSET ?=
BATCH_SIZE ?= 10
MAX_BATCHES ?=
RUN_ID ?=
RESUME ?=

fetch-massive:   ## Scarica OHLCV forex da Massive in data/massive (serve MASSIVE_API_KEY in .env)
	python scripts/fetch_massive_bars.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--bars $(BARS) \
		$(if $(filter 1,$(REFRESH)),--refresh,)

backfill-massive-mongo:   ## Scarica storico Massive e salva barre idempotenti su Mongo (SYMBOLS=... TIMEFRAMES=...)
	python scripts/backfill_massive_mongo.py \
		--env-file .env.worker \
		--symbols $(or $(SYMBOLS),$(SYMBOL)) \
		--timeframes $(or $(TIMEFRAMES),$(TIMEFRAME)) \
		--bars $(BARS) \
		$(if $(BACKFILL_FROM),--from $(BACKFILL_FROM),) \
		$(if $(BACKFILL_TO),--to $(BACKFILL_TO),) \
		$(if $(SYMBOL_LIMIT),--symbol-limit $(SYMBOL_LIMIT),) \
		$(if $(SYMBOL_OFFSET),--symbol-offset $(SYMBOL_OFFSET),)

backfill-massive-batches:   ## Esegue backfill Massive->Mongo a batch riprendibili (tutti i TF di default)
	python scripts/run_massive_backfill_batches.py \
		--env-file .env.worker \
		--timeframes $(or $(TIMEFRAMES),M1,M5,M15,M30,H1,H4,D1) \
		--from $(or $(BACKFILL_FROM),oldest) \
		--batch-size $(BATCH_SIZE) \
		--start-offset $(or $(SYMBOL_OFFSET),0) \
		$(if $(BACKFILL_TO),--to $(BACKFILL_TO),) \
		$(if $(MAX_BATCHES),--max-batches $(MAX_BATCHES),) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) \
		$(if $(filter 1,$(RESUME)),--resume,)

train-massive:   ## Train con dati Massive (cache CSV; REFRESH=1 per riscaricare)
	python scripts/train.py \
		--source massive \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--bars $(BARS) \
		--model-dir $(MODEL_DIR) \
		$(if $(filter 1,$(REFRESH)),--massive-refresh,)

# ── Backtesting ───────────────────────────────────────────────────────────────

FOLDS     ?= 5
TRAIN_PCT ?= 0.70

backtest:   ## Backtest on CSV data  (CSV_FILE=... SYMBOL=... BALANCE=10000)
	python scripts/backtest.py \
		--source csv \
		--file $(CSV_FILE) \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--initial-balance $(BALANCE) \
		--model-dir $(MODEL_DIR)

backtest-no-ml:   ## Backtest with technical indicators only (no ML)
	python scripts/backtest.py \
		--source csv \
		--file $(CSV_FILE) \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--initial-balance $(BALANCE) \
		--no-ml

backtest-mt5:   ## Backtest fetching data live from MT5
	python scripts/backtest.py \
		--source mt5 \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--bars $(BARS) \
		--initial-balance $(BALANCE) \
		--model-dir $(MODEL_DIR)

# ── Walk-forward validation ───────────────────────────────────────────────────

walk-forward:   ## Walk-forward validation on CSV  (CSV_FILE=... SYMBOL=... FOLDS=5)
	python scripts/walk_forward_validation.py \
		--source csv \
		--file $(CSV_FILE) \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--folds $(FOLDS) \
		--train-pct $(TRAIN_PCT) \
		--initial-balance $(BALANCE) \
		--model-dir $(MODEL_DIR)

walk-forward-no-ml:   ## Walk-forward validation — TA indicators only, no ML
	python scripts/walk_forward_validation.py \
		--source csv \
		--file $(CSV_FILE) \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--folds $(FOLDS) \
		--train-pct $(TRAIN_PCT) \
		--initial-balance $(BALANCE) \
		--no-ml

walk-forward-mt5:   ## Walk-forward validation fetching data live from MT5
	python scripts/walk_forward_validation.py \
		--source mt5 \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--bars $(BARS) \
		--folds $(FOLDS) \
		--train-pct $(TRAIN_PCT) \
		--initial-balance $(BALANCE) \
		--model-dir $(MODEL_DIR)

# ── Paper trading ─────────────────────────────────────────────────────────────

paper:   ## Paper trading — live MT5 data, simulated fills
	python scripts/run_paper.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR) \
		--warmup-bars $(WARMUP_BARS) \
		--max-risk-pct $(MAX_RISK_PCT) \
		$(if $(strip $(POLL_INTERVAL)),--poll-interval $(POLL_INTERVAL),) \
		$(if $(strip $(MODEL_VERSION)),--model-version $(MODEL_VERSION),) \
		$(if $(filter 1,$(NO_D1)),--no-d1,) \
		$(if $(filter 1,$(NO_NEWS)),--no-news,) \
		$(if $(strip $(FINNHUB_KEY)),--finnhub-key $(FINNHUB_KEY),) \
		$(PAPER_EXTRA)

paper-no-ml:   ## Paper trading with technical indicators only (no ML)
	python scripts/run_paper.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR) \
		--warmup-bars $(WARMUP_BARS) \
		--max-risk-pct $(MAX_RISK_PCT) \
		$(if $(strip $(POLL_INTERVAL)),--poll-interval $(POLL_INTERVAL),) \
		$(if $(filter 1,$(NO_D1)),--no-d1,) \
		$(if $(filter 1,$(NO_NEWS)),--no-news,) \
		$(if $(strip $(FINNHUB_KEY)),--finnhub-key $(FINNHUB_KEY),) \
		$(PAPER_EXTRA) \
		--no-ml

# ── Live trading ──────────────────────────────────────────────────────────────

live:   ## Show live trading instructions (does NOT start trading)
	@echo ""
	@echo "  WARNING: LIVE TRADING places REAL orders with REAL money."
	@echo ""
	@echo "  Esempi (stesso timeframe del modello addestrato):"
	@echo "    make live-confirmed SYMBOL=EURUSD TIMEFRAME=M1 MODEL_VERSION=v20260414_190200_M1 NO_D1=1 POLL_INTERVAL=10"
	@echo "    make live-no-ml SYMBOL=EURUSD TIMEFRAME=H1 NO_D1=1"
	@echo ""
	@echo "  Variabili: WARMUP_BARS, POLL_INTERVAL, MODEL_VERSION, NO_D1=1, NO_NEWS=1,"
	@echo "    FINNHUB_KEY, MAX_RISK_PCT, DAILY_LOSS_LIMIT_PCT, KILL_DRAWDOWN_PCT,"
	@echo "    MAX_OPEN_POSITIONS, LIVE_EXTRA (argomenti extra verso run_live.py)"
	@echo ""
	@echo "  Prima prova: make paper (stesse variabili, tranne LIVE_EXTRA -> PAPER_EXTRA)"
	@echo ""

live-confirmed:   ## Start LIVE trading — REAL MONEY (use with care)
	python scripts/run_live.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR) \
		--warmup-bars $(WARMUP_BARS) \
		--max-risk-pct $(MAX_RISK_PCT) \
		--daily-loss-limit-pct $(DAILY_LOSS_LIMIT_PCT) \
		--kill-drawdown-pct $(KILL_DRAWDOWN_PCT) \
		--max-open-positions $(MAX_OPEN_POSITIONS) \
		$(if $(strip $(POLL_INTERVAL)),--poll-interval $(POLL_INTERVAL),) \
		$(if $(strip $(MODEL_VERSION)),--model-version $(MODEL_VERSION),) \
		$(if $(filter 1,$(NO_D1)),--no-d1,) \
		$(if $(filter 1,$(NO_NEWS)),--no-news,) \
		$(if $(strip $(FINNHUB_KEY)),--finnhub-key $(FINNHUB_KEY),) \
		$(LIVE_EXTRA) \
		--confirm

live-no-ml:   ## Start LIVE trading with technical indicators only — REAL MONEY
	python scripts/run_live.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR) \
		--warmup-bars $(WARMUP_BARS) \
		--max-risk-pct $(MAX_RISK_PCT) \
		--daily-loss-limit-pct $(DAILY_LOSS_LIMIT_PCT) \
		--kill-drawdown-pct $(KILL_DRAWDOWN_PCT) \
		--max-open-positions $(MAX_OPEN_POSITIONS) \
		$(if $(strip $(POLL_INTERVAL)),--poll-interval $(POLL_INTERVAL),) \
		$(if $(filter 1,$(NO_D1)),--no-d1,) \
		$(if $(filter 1,$(NO_NEWS)),--no-news,) \
		$(if $(strip $(FINNHUB_KEY)),--finnhub-key $(FINNHUB_KEY),) \
		$(LIVE_EXTRA) \
		--no-ml \
		--confirm

# ── Dashboard ─────────────────────────────────────────────────────────────────

run-dashboard: venv-ensure  ## Start FastAPI observability dashboard (OBSERVABILITY_PORT in .env, default 8080)
	python scripts/run_dashboard.py

# Suggerimento Windows: liberare la porta prima di run-dashboard
stop-services:   ## Print commands to stop dashboard + traders (run in PowerShell)
	@echo "PowerShell (dashboard su porta 8080 + processi run_live/run_paper/run_dashboard):"
	@echo "  Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $$_.OwningProcess -Force -ErrorAction SilentlyContinue }"
	@echo "  Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $$_.CommandLine -match 'run_live|run_paper|run_dashboard' } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force }"
	@echo ""
	@echo "Poi: make run-dashboard   oppure   make live-confirmed ..."

# ── Kill switch ───────────────────────────────────────────────────────────────

kill:   ## Activate SESSION_GATE — blocks all new trades until reset (cross-platform)
	python scripts/kill_switch_cli.py activate --level 2

kill-emergency:   ## Activate EMERGENCY_HALT — close positions and stop system
	python scripts/kill_switch_cli.py activate --level 3

reset-kill:   ## Reset kill switch — re-enables order flow
	python scripts/kill_switch_cli.py reset
