# Piano Telegram — Notifiche e Comandi

## Obiettivo

Telegram è l'unico punto di controllo del bot mentre è in esecuzione.
Due direzioni:
- **Bot → Utente**: notifiche proattive su eventi rilevanti
- **Utente → Bot**: comandi per chiedere status o inviare istruzioni

---

## Parte 1 — Notifiche in uscita

### Principio di design

| Priorità | Frequenza max | Regola |
|---|---|---|
| 🔴 Critical | Immediata | Sempre inviata, qualunque orario |
| 🟠 High | Immediata | Inviata sempre durante la sessione |
| 🟡 Medium | 1 per evento | Inviata, ma con throttle se ripetitiva |
| ⚪ Low | Configurabile | Off di default, opt-in via config |

---

### Catalogo notifiche

#### Trade Lifecycle

| ID | Priorità | Quando | Contenuto |
|---|---|---|---|
| `TRADE_OPENED` | 🟠 | Ogni apertura posizione | Symbol, side, lot size, entry, SL, TP, ATR, modello attivo |
| `TRADE_CLOSED` | 🟠 | Ogni chiusura posizione | Symbol, side, PnL (pips + valuta), durata barre, motivo chiusura, balance post-trade |
| `TRADE_SL_MOVED` | ⚪ | SL spostato a breakeven o trailing | Nuovo livello SL, profitto protetto |
| `TRADE_PARTIAL_CLOSE` | 🟡 | Chiusura parziale | Quota chiusa, PnL parziale, posizione residua |

**Formato `TRADE_OPENED`:**
```
🟢 LONG aperto — EURUSD
  Lots:    0.10
  Entry:   1.08523
  SL:      1.08300  (−22.3 pip)
  TP:      1.08970  (+44.7 pip)  R:R 2.0
  ATR(14): 0.00180
  Modello: v20260418_1430_M1_at  (holdout 55.7%)
  Ora:     10:32 UTC
```

**Formato `TRADE_CLOSED`:**
```
✅ LONG chiuso — EURUSD  +18.4 pip  (+$18.40)
  Entry:   1.08523  →  Exit: 1.08707
  Durata:  23 barre  (23 min)
  Motivo:  trailing_stop
  Balance: $10 018.40  (sessione: +$18.40)
```

---

#### Sessione e Giornata

| ID | Priorità | Quando | Contenuto |
|---|---|---|---|
| `SESSION_START` | 🟡 | Runner avviato | Modalità, symbol, timeframe, modello attivo, balance iniziale |
| `SESSION_END` | 🟡 | Runner fermato | Riepilogo sessione: trade, PnL, win rate, balance finale |
| `DAILY_SUMMARY` | 🟡 | 21:00 UTC (fine NY) o on-demand | Trade del giorno, PnL, win rate, drawdown max, balance |
| `DAILY_LOSS_LIMIT` | 🔴 | Soglia perdita giornaliera raggiunta | Perdita attuale, soglia, runner in pausa |

**Formato `DAILY_SUMMARY`:**
```
📊 Riepilogo giornaliero — EURUSD M1
  Trade:       12 (7 win / 5 loss)
  Win rate:    58.3%
  PnL:         +$47.20
  PnL pips:    +47.2 pip
  Max DD:      −$12.30 (−0.12%)
  Balance:     $10 047.20
  Modello:     v20260418_1430  live_acc=54.1% (87 campioni)
```

---

#### Modello ML

| ID | Priorità | Quando | Contenuto |
|---|---|---|---|
| `RETRAIN_STARTED` | ⚪ | Scheduler lancia train.py | Trigger (ore/barre), parametri usati |
| `RETRAIN_COMPLETE` | 🟡 | train.py termina con successo | Nuova versione, holdout acc, se è candidato |
| `RETRAIN_FAILED` | 🟠 | train.py termina con errore | Motivo, ultimo modello ancora attivo |
| `MODEL_SWITCHED` | 🟠 | ModelWatcher esegue lo switch | Vecchio vs nuovo modello, holdout e live accuracy |
| `MODEL_ACCURACY_WARN` | 🟠 | Live accuracy < soglia (50%) | Accuracy attuale, campioni valutati, soglia |
| `MODEL_ACCURACY_CHECKPOINT` | ⚪ | Ogni 50 previsioni valutate | Accuracy corrente, campioni, affidabilità |

**Formato `MODEL_SWITCHED`:**
```
🔄 Modello aggiornato — EURUSD
  Precedente: v20260415_0900  (holdout 53.1%,  live 49.2%  |  203 campioni)
  Nuovo:      v20260418_1430  (holdout 55.7%)
  Trigger:    upgrade (holdout migliore)
  Ora switch: 14:32 UTC  (0 posizioni aperte)
```

**Formato `MODEL_ACCURACY_WARN`:**
```
⚠️ Accuracy modello degradata — EURUSD
  Modello:   v20260418_1430
  Live acc:  47.3%  (soglia: 50%)
  Campioni:  214  ✅ affidabile
  Azione:    in attesa di candidato migliore
```

---

#### Sistema e Risk

| ID | Priorità | Quando | Contenuto |
|---|---|---|---|
| `DRAWDOWN_ALERT` | 🔴 | DD > soglia config (default 5%) | Equity, peak, DD %, modalità recovery attiva |
| `KILL_SWITCH` | 🔴 | Kill switch attivato (qualsiasi livello) | Livello, motivo, chi lo ha attivato |
| `SPREAD_BLOCKED` | ⚪ | Spread > max (opt-in, molto rumoroso) | Spread attuale vs max configurato |
| `MT5_RECONNECT` | 🟡 | Connessione MT5 persa e ripristinata | Downtime, bars saltate |
| `SYSTEM_ERROR` | 🔴 | Eccezione non gestita nel loop | Modulo, stacktrace troncato |

---

## Parte 2 — Comandi in entrata (User → Bot)

### Architettura

Il bot attuale è **solo in uscita** (fire-and-forget HTTP POST). Per ricevere comandi serve un **receiver a polling** che giri in background:

```
TelegramCommandReceiver (thread daemon)
  └─ ogni 3 sec: GET /getUpdates?offset=N
       └─ per ogni update:
            - verifica chat_id autorizzato
            - parsa il comando
            - invoca il callback registrato
            - risponde con sendMessage
```

Sicurezza: solo messaggi con `chat_id == config.chat_id` vengono processati. Tutto il resto viene silenziosamente ignorato.

---

### Catalogo comandi

| Comando | Descrizione | Risposta |
|---|---|---|
| `/status` | Stato generale del sistema | Running/stopped, modalità, symbol, TF, balance, equity, posizioni aperte, uptime |
| `/positions` | Posizioni aperte in dettaglio | Per ogni posizione: side, entry, unrealized PnL in pip e USD, durata barre |
| `/balance` | Snapshot account | Balance, equity, free margin, margine usato, DD corrente |
| `/model` | Info modello attivo | Versione, holdout acc, live acc, campioni valutati, affidabilità |
| `/accuracy` | Accuracy live dettagliata | n_evaluated, n_correct, n_pending, CI 95%, is_reliable |
| `/daily` | Riepilogo giornata on-demand | Stesso formato di `DAILY_SUMMARY` |
| `/retrain` | Avvia retraining manuale | Conferma avvio, parametri usati |
| `/pause` | Pausa nuove entry (exit continuano) | Conferma, ora di pausa |
| `/resume` | Riprende entry dopo `/pause` | Conferma, durata pausa |
| `/stop` | Stop completo (richiede conferma) | "Invia /stop confirm per confermare" |
| `/stop confirm` | Conferma stop | Kill switch attivato, posizioni gestite normalmente |
| `/help` | Lista comandi | Lista formattata |

---

### Formato `/status`
```
🤖 MetaTrade — EURUSD M1  [PAPER]
  Balance:   $10 023.40
  Equity:    $10 031.20  (+$7.80 unreal.)
  Posizioni: 1 aperta
  Uptime:    2h 14m
  Ultimo trade: 12 min fa
  Modello:   v20260418_1430  live=54.1%
  Session:   🟢 London+NY  (07:00–21:00 UTC)
  Ore:       14:47 UTC
```

### Formato `/positions`
```
📋 Posizioni aperte (1)

1. EURUSD LONG  0.10 lots
   Entry:  1.08523  (14:32 UTC)
   Attuale: 1.08690
   PnL:    +16.7 pip  (+$16.70)
   Durata: 15 barre
   SL:     1.08300  TP: 1.08970
```

### Formato `/model`
```
🧠 Modello attivo — EURUSD M1
  Versione:    v20260418_1430_M1_at
  Holdout acc: 55.7%  (training)
  Live acc:    54.1%  (87/214 corretti)
  Affidabile:  ✅  (≥200 campioni)
  Candidato:   nessuno in attesa
  Ultimo train: 4h 12m fa
```

---

## Parte 3 — Architettura dei nuovi file

### File da creare / modificare

```
src/metatrade/alerting/
├── config.py            MODIFY  — aggiunge throttle_sec, daily_summary_hour_utc,
│                                  notify_on_* flag per categoria
├── telegram_alerter.py  MODIFY  — nuovi metodi: alert_trade_closed_full,
│                                  alert_session_start/end, alert_retrain_started/failed,
│                                  alert_daily_summary_full, throttle interno
├── command_receiver.py  NEW     — TelegramCommandReceiver (thread daemon, long-poll)
└── command_handler.py   NEW     — CommandHandler: registry dei callback per ogni comando

tests/alerting/
├── test_telegram_alerter.py   NEW   — test nuovi metodi (mock HTTP)
└── test_command_handler.py    NEW   — test routing comandi, sicurezza chat_id
```

### Modifiche al runner

`BaseRunner` riceve un `CommandHandler` opzionale e registra i callback:
```python
# In __init__:
if command_handler is not None:
    command_handler.register("/status",    self._cmd_status)
    command_handler.register("/positions", self._cmd_positions)
    command_handler.register("/balance",   self._cmd_balance)
    command_handler.register("/pause",     self._cmd_pause)
    command_handler.register("/resume",    self._cmd_resume)
    command_handler.register("/stop",      self._cmd_stop)
```

### AlertConfig — nuove opzioni

```python
@dataclass
class AlertConfig:
    # ... esistenti ...

    # Throttle: non inviare lo stesso tipo di alert più di una volta ogni N sec
    throttle_sec: int = 60            # default: max 1 alert identico al minuto

    # Ora UTC in cui inviare il daily summary automatico (None = disabilitato)
    daily_summary_hour_utc: int | None = 21

    # Flag per categoria (tutto on di default tranne le voci noiose)
    notify_trade_opened: bool = True
    notify_trade_closed: bool = True
    notify_trade_sl_moved: bool = False    # opt-in
    notify_session_start: bool = True
    notify_session_end: bool = True
    notify_daily_summary: bool = True
    notify_model_events: bool = True
    notify_accuracy_checkpoint: bool = False   # opt-in
    notify_spread_blocked: bool = False        # opt-in (molto rumoroso)
    notify_system_errors: bool = True
```

---

## Parte 4 — Ordine di sviluppo

```
Step 1 — Nuovi metodi TelegramAlerter (½ giornata)
  - alert_trade_closed_full (con entry/exit/pips/durata/balance)
  - alert_session_start / alert_session_end
  - alert_retrain_started / alert_retrain_failed
  - alert_daily_summary_full
  - throttle interno (dict ultimo invio per tipo)
  - Aggiornamento AlertConfig con nuove opzioni

Step 2 — Chiamate mancanti nel runner (½ giornata)
  - BaseRunner: alert_trade_opened già c'è, mancano session_start/end
  - ExitEngine: alert_trade_closed_full quando chiude una posizione
  - RetrainScheduler: alert_retrain_started al lancio, alert_retrain_failed se exit code != 0
  - alert_daily_summary al raggiungimento di daily_summary_hour_utc

Step 3 — CommandReceiver (1 giornata)
  - TelegramCommandReceiver: thread daemon, long-poll getUpdates
  - Filtro chat_id, parsing comando, dispatch al CommandHandler
  - Gestione /stop confirm (stato temporaneo)
  - Test con mock HTTP

Step 4 — CommandHandler + callback nel runner (1 giornata)
  - CommandHandler: registro comandi → callback
  - Callback nel runner: _cmd_status, _cmd_positions, _cmd_balance,
    _cmd_model, _cmd_accuracy, _cmd_daily, _cmd_pause, _cmd_resume, _cmd_stop
  - /retrain chiama RetrainScheduler._launch() direttamente
  - Test integrazione comando → risposta
```

---

## Variabili d'ambiente nuove

```env
ALERT_BOT_TOKEN=123456:ABC...
ALERT_CHAT_ID=-1001234567890
ALERT_ENABLED=true
ALERT_MIN_PNL_ALERT=0.0
ALERT_THROTTLE_SEC=60
ALERT_DAILY_SUMMARY_HOUR_UTC=21
ALERT_NOTIFY_TRADE_SL_MOVED=false
ALERT_NOTIFY_ACCURACY_CHECKPOINT=false
ALERT_NOTIFY_SPREAD_BLOCKED=false
```
