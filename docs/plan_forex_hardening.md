# Piano Forex Hardening

## Obiettivo

Questo piano prende il progetto nello stato attuale e lo porta verso un sistema:

1. robusto sul piano operativo;
2. coerente tra backtest, paper e live;
3. misurabile sul piano statistico;
4. difficile da rompere in produzione;
5. capace di migliorare le performance senza mentire a se stesso.

Il target non e' "aggiungere altri indicatori". Il target e' eliminare tutte le zone in cui oggi il sistema puo' sovrastimare il proprio edge, sottostimare il rischio o prendere decisioni con metriche incoerenti.

---

## Sintesi secca dello stato attuale

La repo ha una base valida: risk layer, orchestration, MT5 adapter, ML lifecycle, intermarket, exit engine, telemetry e dashboard.

Il problema non e' la mancanza di componenti. Il problema e' che alcune parti chiave non sono ancora allineate tra loro. Questo crea tre rischi grossi:

1. performance apparente migliore di quella reale;
2. risk management meno severo di quanto sembri;
3. decisioni di promotion / sizing / exposure basate su proxy incompleti.

---

## Debolezze principali trovate nel progetto

### P0 - Disallineamento tra simulazione e realta'

- `src/metatrade/broker/paper_broker.py`
  - Il broker paper accetta ordini e genera fill, ma non mantiene davvero posizioni aperte, PnL mark-to-market, margine usato e open positions count.
- `src/metatrade/runner/paper_runner.py`
  - Il runner paper registra il fill, ma non aggiorna uno stato di posizione coerente e non replica il ciclo completo order -> fill -> position -> exit -> pnl.
- `src/metatrade/runner/backtest_runner.py`
  - Il backtest gestisce una sola posizione alla volta, passa sempre `open_positions=0`, `free_margin=balance`, `account_equity=balance`, non usa spread live e non riproduce davvero il comportamento del motore live.

Effetto: paper e backtest oggi non sono ambienti affidabili per stimare la performance live.

### P0 - Risk management meno duro di quanto dichiari

- `src/metatrade/risk/manager.py`
  - Il sizing usa `max(account.equity, account.balance)`. In drawdown flottante questo impedisce al size di ridursi davvero.
- `src/metatrade/risk/pre_trade_checker.py`
  - `daily_loss_limit_pct` e' calcolato contro `account.drawdown_from_equity`, quindi guarda solo la perdita flottante, non la perdita giornaliera realizzata.
- `src/metatrade/risk/manager.py`
  - Anche l'auto-kill si attiva sulla drawdown flottante rispetto al balance corrente, non sulla perdita intraday totale.

Effetto: il sistema puo' continuare a tradare piu' del dovuto dopo una giornata negativa gia' realizzata.

### P0 - Intermarket e portfolio risk non realmente collegati al portafoglio

- `src/metatrade/runner/base.py`
  - L'intermarket usa `_get_open_positions()` ma il base ritorna sempre lista vuota.
  - La valutazione usa un lot nominale fisso `0.01` invece del lot reale candidato.

Effetto: se abiliti il layer intermarket, oggi il controllo del rischio di portafoglio e' parziale e puo' essere fuorviante.

### P0 - ML governance con metriche non omogenee

- `src/metatrade/ml/live_tracker.py`
  - La live accuracy usa un criterio direzionale semplice (`future_close > entry_close` / `<`), diverso dalla logica di labeling del training.
- `src/metatrade/ml/promotion_policy.py`
  - La promotion confronta holdout training e live accuracy che non misurano esattamente la stessa cosa.

Effetto: il modello puo' essere promosso o degradato con un metro diverso da quello con cui e' stato addestrato.

### P1 - Lifecycle di ordine e trade non completamente chiuso

- `src/metatrade/execution/order_manager.py`
  - L'order manager costruisce e sottomette ordini, ma il tracking di fill/position non e' il centro reale del flusso live.
- `src/metatrade/exit_engine/engine.py`
  - `record_outcome()` esiste, ma non e' collegato in modo chiaro al lifecycle di chiusura posizione.
- `src/metatrade/observability/store.py`
  - Il trade journal esiste, ma non e' il ledger unificato che governa apertura, aggiornamento e chiusura di tutti i trade.

Effetto: reputazione delle exit rules, analytics post-trade e audit trail non sono ancora di livello istituzionale.

### P1 - Assunzioni statiche sbagliate per simboli diversi da EURUSD

- `src/metatrade/runner/backtest_runner.py`
  - `_PIP_SIZE = 0.0001` e' hardcoded.
- `src/metatrade/risk/config.py`
  - `pip_value_per_lot` e `pip_digits` sono statici di default.
- `src/metatrade/runner/base.py`
  - Fallback SL a distanza fissa `0.0020`.

Effetto: su JPY pair, cross particolari o broker con contratti diversi, PnL, sizing e stop possono essere sbagliati.

### P1 - Stack moduli molto largo, ma non si vede ancora una disciplina di ablation

- `src/metatrade/runner/module_builder.py`
  - Molti moduli sono attivi di default insieme.

Effetto: rischio di ridondanza, overfitting di consenso, peso eccessivo dato alla complessita' rispetto all'edge misurato.

### P2 - News e dati esogeni ancora troppo rigidi

- `src/metatrade/technical_analysis/modules/news_calendar_module.py`
  - Il calendario hardcoded copre eventi fissati a mano e ha orizzonte temporale limitato.

Effetto: il filtro news puo' degradare nel tempo o perdere eventi importanti non previsti.

### P2 - Ambiente di test non riproducibile qui e ora

- In questa sessione non e' stato possibile eseguire `pytest` perche' `pytest` non e' installato nell'ambiente corrente.

Effetto: il progetto ha molti test, ma manca la garanzia pratica che un checkout pulito sia subito verificabile nello stesso ambiente operativo.

---

## Principio guida

Prima si corregge la verita' del sistema, poi si ottimizza la performance.

Ordine corretto:

1. verita' della simulazione;
2. verita' del rischio;
3. verita' delle metriche ML;
4. verita' dell'audit operativo;
5. solo dopo, ottimizzazione dell'edge.

---

## Roadmap completa

## Fase 0 - Bloccare i falsi positivi del sistema

### Obiettivo

Impedire che backtest, paper e metriche live raccontino una storia migliore di quella reale.

### Da aggiornare

- `src/metatrade/runner/backtest_runner.py`
- `src/metatrade/runner/paper_runner.py`
- `src/metatrade/broker/paper_broker.py`
- `src/metatrade/risk/manager.py`
- `src/metatrade/risk/pre_trade_checker.py`
- `src/metatrade/ml/live_tracker.py`
- `src/metatrade/ml/promotion_policy.py`

### Da aggiungere

- Un contratto unico di position lifecycle condiviso da backtest, paper e live.
- Una struttura di account state che distingua:
  - realized pnl del giorno;
  - unrealized pnl;
  - peak equity di sessione;
  - session start equity;
  - exposure e margin usage reali.

### Da eliminare o sostituire

- Le assunzioni basate su `max(balance, equity)` per sizing e margin reference.
- Il daily loss basato solo sulla perdita flottante.
- La live accuracy direzionale se non e' allineata ai label del training.

### Definition of done

- Un trade chiuso in perdita riduce davvero la capacita' di aprire nuovi trade nella stessa giornata.
- La riduzione size in drawdown si basa su capitale effettivamente rischiabile.
- La live accuracy e l'holdout accuracy sono confrontabili.

---

## Fase 1 - Rendere paper e backtest credibili

### Obiettivo

Fare in modo che paper e backtest siano ambienti di validazione, non demo estetiche.

### Interventi

1. Ricostruire il `PaperBrokerAdapter` come broker simulato vero:
   - registro posizioni aperte;
   - mark-to-market ad ogni barra;
   - used margin realistico;
   - free margin realistico;
   - open positions count reale;
   - commissioni, spread, slippage configurabili;
   - chiusura SL/TP/trailing.

2. Allineare `PaperRunner` al flusso live:
   - usare `OrderManager`;
   - produrre `Fill`;
   - aprire `Position`;
   - aggiornare il journal;
   - chiudere la posizione con lo stesso exit path del live.

3. Rifare il `BacktestRunner` per supportare:
   - piu' posizioni contemporanee;
   - margin constraints;
   - spread dinamico;
   - slippage dipendente da sessione e volatilita';
   - gap handling;
   - stessa logica di stop e trailing del live.

4. Correggere il problema entry/SL:
   - oggi il rischio viene deciso sul close segnale e poi fillato alla barra dopo;
   - il motore deve ricalcolare il rischio effettivo sul prezzo di fill reale.

### File target

- `src/metatrade/broker/paper_broker.py`
- `src/metatrade/runner/paper_runner.py`
- `src/metatrade/runner/backtest_runner.py`
- `src/metatrade/execution/order_manager.py`
- `src/metatrade/core/contracts/position.py`
- `tests/broker/test_paper_broker.py`
- `tests/runner/test_runners.py`
- `tests/runner/test_trailing_slippage_circuit.py`

### Metriche di accettazione

- Reconciliation test: stesso dataset, stessa config, differenza paper vs backtest entro tolleranza definita.
- Il numero posizioni aperte in paper coincide sempre con il broker simulato.
- Il PnL netto del backtest include spread, slippage e commissioni in modo esplicito.

---

## Fase 2 - Hardening del risk engine

### Obiettivo

Far diventare il risk layer il vero gatekeeper, non un filtro incompleto.

### Interventi

1. Ancorare tutti i limiti giornalieri a `session_start_equity`, non al balance corrente.
2. Distinguere tre livelli di perdita:
   - realized daily loss;
   - unrealized open risk;
   - total intraday drawdown.
3. Calcolare il position sizing sul capitale rischiabile reale:
   - equity disponibile;
   - margine disponibile;
   - heat di portafoglio;
   - rischio gia' allocato su posizioni aperte.
4. Aggiungere portfolio heat caps:
   - max total open risk;
   - max risk per currency;
   - max correlated exposure;
   - max same-session losses.
5. Aggiungere regime-aware risk scaling:
   - risk down in high spread / high ATR / news adjacency;
   - risk down after drawdown cluster;
   - no martingale implicito.
6. Aggiungere kill switch multilivello:
   - soft pause;
   - trade gate;
   - session gate;
   - hard kill con motivazione persistita.

### File target

- `src/metatrade/risk/manager.py`
- `src/metatrade/risk/pre_trade_checker.py`
- `src/metatrade/risk/position_sizer.py`
- `src/metatrade/risk/config.py`
- `src/metatrade/risk/kill_switch.py`
- `src/metatrade/runner/base.py`
- `tests/risk/test_risk_manager.py`
- `tests/risk/test_pre_trade_checker.py`
- `tests/risk/test_kill_switch.py`

### Metriche di accettazione

- Dopo una perdita giornaliera oltre soglia, nessuna nuova entry passa.
- Il lot size scende automaticamente quando equity e free margin scendono.
- Il rischio totale aperto non supera mai il cap di portafoglio.

---

## Fase 3 - Rendere serio il portfolio risk

### Obiettivo

Passare da single-symbol bot con accessori intermarket a motore che capisce davvero l'esposizione aggregata.

### Interventi

1. Implementare `_get_open_positions()` in runner concreti.
2. Passare all'`IntermarketEngine` il lot size candidato reale, non un placeholder.
3. Collegare il portfolio risk a:
   - exposure per currency;
   - correlation cluster;
   - same idea / same side throttling;
   - hedging illusion detection.
4. Aggiungere persistence delle snapshot di exposure.
5. Creare dashboard portfolio-level:
   - net exposure;
   - gross exposure;
   - cluster occupancy;
   - open risk by currency.

### File target

- `src/metatrade/runner/base.py`
- `src/metatrade/runner/paper_runner.py`
- `src/metatrade/runner/live_runner.py`
- `src/metatrade/intermarket/engine.py`
- `src/metatrade/intermarket/exposure.py`
- `src/metatrade/intermarket/persistence.py`
- `tests/intermarket/test_engine.py`
- `tests/intermarket/test_exposure.py`

### Metriche di accettazione

- L'intermarket produce decisioni diverse a seconda del portafoglio aperto reale.
- Nessun trade passa se porta l'esposizione oltre i limiti configurati.

---

## Fase 4 - Ricostruire il laboratorio ML in modo coerente con il live

### Obiettivo

Smettere di ottimizzare accuracy e iniziare a ottimizzare edge netto, robustezza e stabilita' fuori campione.

### Interventi

1. Allineare training labels, live tracking e promotion metric.
   - stessa definizione di correttezza;
   - stessa finestra forward;
   - stessa gestione HOLD/no-trade zone;
   - stesse session filter.

2. Sostituire la promotion basata solo su accuracy con una policy multi-metrica:
   - holdout accuracy;
   - expectancy netta dopo costi;
   - max drawdown;
   - trade frequency minima;
   - stability score per fold.

3. Aggiungere ablation framework:
   - contributo reale di ogni modulo;
   - contributo del consenso dinamico;
   - contributo di exit profile;
   - contributo di session/news filter.

4. Aggiungere walk-forward piu' realistico:
   - cost-aware;
   - spread-aware;
   - session-aware;
   - news-aware;
   - purged validation se usi feature con overlap temporale forte.

5. Aggiungere drift monitoring:
   - feature drift;
   - label drift;
   - regime drift;
   - spread regime drift.

6. Segmentare i modelli:
   - per simbolo;
   - per timeframe;
   - eventualmente per regime.

### File target

- `src/metatrade/ml/walk_forward.py`
- `src/metatrade/ml/live_tracker.py`
- `src/metatrade/ml/promotion_policy.py`
- `src/metatrade/ml/model_watcher.py`
- `src/metatrade/ml/classifier.py`
- `src/metatrade/ml/config.py`
- `scripts/train.py`
- `scripts/walk_forward_validation.py`
- `tests/ml/test_live_tracker.py`
- `tests/ml/test_model_watcher.py`
- `tests/ml/test_promotion_policy.py`

### Metriche di accettazione

- Nessun modello viene promosso se migliora accuracy ma peggiora expectancy netta.
- Le metriche live e holdout sono confrontabili senza reinterpretazioni.
- Ogni release modello produce report con breakdown per regime, sessione e costo.

---

## Fase 5 - Migliorare davvero la performance di trading

### Obiettivo

Dopo aver ripulito la verita' del sistema, migliorare il ritorno risk-adjusted.

### Interventi

1. Ridurre il numero di trade mediocri:
   - alzare la soglia su condizioni di spread scadente;
   - evitare sessioni deboli;
   - evitare re-entry troppo ravvicinate;
   - ridurre le entry durante regime instabile.

2. Migliorare la qualita' dell'entry:
   - non usare tutti i moduli sempre accesi;
   - creare bundle di strategia:
     - trend;
     - mean reversion;
     - breakout;
     - no-trade filter.

3. Migliorare la qualita' dell'exit:
   - tracciare MFE/MAE per trade;
   - confrontare TP fisso vs trailing vs partials;
   - adattare exit per regime e spread.

4. Migliorare il consenso:
   - pesare i moduli non solo sulla direzione corretta, ma sull'utilita' economica;
   - ridurre il peso di moduli ridondanti;
   - introdurre confidence calibration.

5. Migliorare il costo medio per trade:
   - filtri di liquidita';
   - no-trade su spread tail;
   - slippage model per sessione;
   - eventuale size throttling vicino a news e roll-over.

### Da aggiungere

- Trade tagging automatico:
  - regime;
  - sessione;
  - setup type;
  - volatility bucket;
  - spread bucket;
  - module stack attiva;
  - model version.

- Report research:
  - expectancy per tag;
  - profit factor per tag;
  - max adverse excursion;
  - hold time distribution;
  - outlier analysis.

### Da eliminare o ridurre

- Moduli che non aggiungono expectancy netta dopo costi.
- Logiche di entry che funzionano solo su un simbolo o un regime.
- Complessita' non difendibile da metriche.

---

## Fase 6 - Dati, contratti strumento e qualita' esecuzione

### Obiettivo

Togliere dal sistema tutte le assunzioni statiche che lo rendono fragile fuori da EURUSD ideale.

### Interventi

1. Derivare dal broker o da metadata centralizzati:
   - pip size;
   - pip digits;
   - point size;
   - pip value per lot;
   - min lot;
   - lot step;
   - max lot;
   - stops level;
   - margin requirements per symbol.

2. Creare un `InstrumentSpecRegistry`.

3. Usare questi dati in:
   - sizing;
   - PnL;
   - backtest fills;
   - stop validation;
   - dashboards e reports.

4. Rimuovere:
   - `_PIP_SIZE = 0.0001`;
   - fallback SL fissi non normalizzati per simbolo;
   - spread costanti hardcoded nel paper runner.

### File target

- `src/metatrade/broker/mt5_adapter.py`
- `src/metatrade/risk/config.py`
- `src/metatrade/risk/position_sizer.py`
- `src/metatrade/runner/backtest_runner.py`
- `src/metatrade/runner/base.py`
- `src/metatrade/core/contracts/market.py`
- `tests/broker/test_mt5_adapter.py`
- `tests/runner/test_min_sl_enforcement.py`

---

## Fase 7 - Osservabilita', audit e controllo operativo

### Obiettivo

Avere un sistema che si puo' governare in tempo reale e spiegare ex post.

### Interventi

1. Fare del `trade_journal` il ledger ufficiale.
2. Collegare ogni trade a:
   - decision snapshot;
   - segnali dei moduli;
   - modello;
   - fill reali;
   - reason di exit;
   - pnl lordo e netto;
   - MFE/MAE;
   - slippage realizzato.
3. Collegare davvero `ExitEngine.record_outcome()` al lifecycle di chiusura.
4. Aggiungere alert operativi per:
   - execution drift;
   - fill rejection cluster;
   - spread anomaly;
   - model degradation;
   - repeated veto reasons.
5. Aggiungere dashboard di controllo:
   - exposure;
   - open risk;
   - realized/unrealized pnl giornaliero;
   - strategy attribution;
   - model attribution;
   - broker execution quality.

### File target

- `src/metatrade/observability/store.py`
- `src/metatrade/observability/api.py`
- `src/metatrade/execution/order_manager.py`
- `src/metatrade/runner/live_runner.py`
- `src/metatrade/exit_engine/engine.py`
- `src/metatrade/observability/static/dashboard.js`
- `src/metatrade/observability/templates/dashboard.html`

---

## Fase 8 - Test, release discipline e sicurezza del rilascio

### Obiettivo

Fare in modo che ogni modifica che tocca rischio, execution o ML sia dimostrabile prima del deploy.

### Interventi

1. Rendere l'ambiente test riproducibile:
   - `dev` environment installabile con un comando;
   - test command standard;
   - CI che esegue i blocchi critici.

2. Creare test di non regressione su:
   - sizing;
   - daily loss;
   - kill switch;
   - multi-position backtest;
   - paper/live parity;
   - promotion policy;
   - intermarket exposure.

3. Aggiungere replay tests su dataset storici noti:
   - sessioni normali;
   - news shock;
   - spread shock;
   - gap;
   - fast trend reversal.

4. Aggiungere release gates:
   - no deploy se fail qualsiasi test P0;
   - no model promotion se mancano metriche OOS complete;
   - no live start se config simbolo incompleta.

---

## Cosa aggiornare, aggiungere o eliminare per area

## Aggiornare subito

- broker paper
- backtest runner
- risk manager
- pre-trade checker
- live accuracy tracker
- promotion policy
- intermarket integration
- trade journal wiring

## Aggiungere subito

- instrument spec registry
- true paper position ledger
- portfolio heat controls
- trade attribution report
- cost-aware validation metrics
- replay test suite

## Eliminare o sostituire subito

- placeholder lot size `0.01` nell'intermarket
- sizing basato su `max(balance, equity)`
- daily loss basato solo su drawdown flottante
- pip assumptions hardcoded
- spread hardcoded nel paper path
- metriche live non allineate ai label del training

---

## Ordine pratico di esecuzione

1. Correggere risk engine e metrica live.
2. Rifare paper broker e paper runner.
3. Rifare il backtest per parity con live/paper.
4. Collegare lifecycle ordine -> fill -> posizione -> chiusura -> journal.
5. Sistemare intermarket con open positions reali.
6. Sistemare instrument specs per multi-symbol serio.
7. Solo dopo: ablation, feature selection, tuning, promotion multi-metrica.

---

## Criteri per dichiarare il sistema pronto al live serio

Il sistema non va considerato pronto finche' non soddisfa tutti questi gate:

- backtest, paper e live condividono le stesse regole di esecuzione;
- il risk engine blocca correttamente su realized daily loss e portfolio heat;
- ogni trade e' tracciato end-to-end nel journal;
- la promotion ML usa metriche coerenti tra training e live;
- i costi reali sono misurati e non stimati grossolanamente;
- il sistema conosce i contratti reali del simbolo che sta tradando;
- esistono replay test per scenari di mercato ostili;
- i moduli attivi sono giustificati da expectancy netta, non da intuizione.

---

## Risultato atteso da questo piano

Se eseguito nell'ordine giusto, questo piano non "promette miracoli". Fa qualcosa di piu' utile:

- rimuove le illusioni statistiche;
- abbassa gli errori operativi;
- riduce il rischio di overfitting;
- migliora il rapporto tra edge e costo reale;
- trasforma il bot in un sistema che sa quando NON deve tradare.

Per un motore forex, e' li' che comincia la vera performance.
