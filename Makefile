.PHONY: help install test test-fast lint lint-fix typecheck clean \
        train train-csv train-mt5 train-mt5-mtf \
        backtest backtest-no-ml backtest-mt5 \
        walk-forward walk-forward-no-ml walk-forward-mt5 \
        paper paper-no-ml \
        live live-confirmed live-no-ml \
        run-dashboard \
        kill kill-emergency reset-kill

# ── Configurable defaults (override on the command line) ──────────────────────
SYMBOL    ?= EURUSD
TIMEFRAME ?= H1
BARS      ?= 30000
CSV_FILE  ?= data/$(SYMBOL)_$(TIMEFRAME).csv
MODEL_DIR ?= data/models
BALANCE   ?= 10000

# ─────────────────────────────────────────────────────────────────────────────

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Configurable variables (pass on command line):"
	@echo "    SYMBOL=$(SYMBOL)  TIMEFRAME=$(TIMEFRAME)  BARS=$(BARS)"
	@echo "    CSV_FILE=$(CSV_FILE)  MODEL_DIR=$(MODEL_DIR)  BALANCE=$(BALANCE)"
	@echo "    FOLDS=$(FOLDS)  TRAIN_PCT=$(TRAIN_PCT)  TIMEFRAMES=$(TIMEFRAMES)"

# ── Development ───────────────────────────────────────────────────────────────

install:   ## Install package + dev dependencies
	pip install -e ".[dev]"

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

paper:   ## Paper trading — live MT5 data, simulated fills  (SYMBOL=EURUSD)
	python scripts/run_paper.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR)

paper-no-ml:   ## Paper trading with technical indicators only (no ML)
	python scripts/run_paper.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--no-ml

# ── Live trading ──────────────────────────────────────────────────────────────

live:   ## Show live trading instructions (does NOT start trading)
	@echo ""
	@echo "  WARNING: LIVE TRADING places REAL orders with REAL money."
	@echo ""
	@echo "  To start live trading, use one of:"
	@echo "    make live-confirmed SYMBOL=$(SYMBOL) TIMEFRAME=$(TIMEFRAME)"
	@echo "    make live-no-ml     SYMBOL=$(SYMBOL) TIMEFRAME=$(TIMEFRAME)"
	@echo ""
	@echo "  Always test on paper first: make paper"
	@echo ""

live-confirmed:   ## Start LIVE trading — REAL MONEY (use with care)
	python scripts/run_live.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--model-dir $(MODEL_DIR) \
		--confirm

live-no-ml:   ## Start LIVE trading with technical indicators only — REAL MONEY
	python scripts/run_live.py \
		--symbol $(SYMBOL) \
		--timeframe $(TIMEFRAME) \
		--no-ml \
		--confirm

# ── Dashboard ─────────────────────────────────────────────────────────────────

run-dashboard:   ## Start FastAPI observability dashboard at http://localhost:8000
	python scripts/run_dashboard.py

# ── Kill switch ───────────────────────────────────────────────────────────────

kill:   ## Activate SESSION_GATE — blocks all new trades until reset (cross-platform)
	python scripts/kill_switch_cli.py activate --level 2

kill-emergency:   ## Activate EMERGENCY_HALT — close positions and stop system
	python scripts/kill_switch_cli.py activate --level 3

reset-kill:   ## Reset kill switch — re-enables order flow
	python scripts/kill_switch_cli.py reset
