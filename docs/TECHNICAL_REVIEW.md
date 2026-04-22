# MetaTrade — Review Tecnica Spietata

> Prospettiva: quant researcher + ML engineer + risk manager istituzionale + software architect + trader sistematico scettico sull'overfitting.
> Obiettivo: decidere se il sistema è promuovibile a live, da bloccare, o da rifondare.

---

## 1. Executive Summary

Il sistema è **sovra-ingegnerizzato, metodologicamente vulnerabile e non validato**.

Ha idee architetturali interessanti — il modello di reputazione, il voto pesato adattivo, il walk-forward — ma le ha impilate una sopra l'altra senza mai rispondere alla domanda fondamentale: **c'è un edge reale?** Prima di qualsiasi adattamento online, prima di qualsiasi modulo aggiuntivo, bisogna dimostrare che il segnale di base batte il random dopo costi di transazione. Questo non è stato fatto.

I problemi principali sono tre:

1. **Ridondanza estrema tra moduli**: almeno 6 dei 14 moduli misurano la stessa cosa con formule diverse. Il Consensus Engine li somma come se fossero indipendenti. Non lo sono.

2. **Adattamento senza validazione**: ci sono almeno 4 sistemi adattativi sovrapposti (AdaptiveThreshold, DynamicVote, Reputation ExitEngine, ML online). Ognuno impara da segnali rumorosi. Insieme rischiano di imparare il rumore del training set e presentare come adattamento ciò che è overfitting.

3. **Leakage metodologico nel ML**: il processo di auto-tune seleziona il miglior trial su un holdout che è stato visto decine di volte. Questo non è out-of-sample. È selezione guidata.

**Il sistema non è pronto per il live trading.** Può diventarlo, ma richiede una revisione profonda della metodologia di validazione e una drastica semplificazione dell'architettura adattiva.

---

## 2. Problemi architetturali principali

### 2.1 Stack adattivo a 4 livelli non validato

Il sistema ha **quattro strati di adattamento online** sovrapposti:

```
AdaptiveThresholdManager   → soglie per modulo
DynamicVote                → pesi per modulo nel consenso
ML Classifier              → ritraining periodico
ExitEngine ReputationModel → pesi per regola di uscita
```

Nessuno di questi layer è stato validato in isolamento. Non sappiamo se ognuno aggiunge valore marginale o se si compensano a vicenda nascondendo l'assenza di un edge di base.

Un sistema adattivo che non ha un edge di base impara a fittare il passato in modo sempre più efficiente. Il risultato in backtest è buono; il risultato in live è casuale o peggio.

**Regola pratica**: aggiungere adattamento ha senso solo se il sistema fisso già funziona.

### 2.2 Ridondanza tra moduli → falsa diversificazione nel consenso

Il Consensus Engine assume implicita indipendenza tra i segnali. Non c'è penalizzazione per correlazione. Risultato: moduli che misurano la stessa cosa votano come se fossero fonti indipendenti.

Cluster di ridondanza identificati:

| Cluster | Moduli ridondanti |
|---------|-----------------|
| Momentum oscillatori | RSI, Adaptive RSI, Stochastic RSI, MACD |
| Trend/MA | EMA Crossover, Multi-TF, ADX, Market Regime |
| Supporto/Resistenza | Pivot Points, Swing Level, Donchian |
| Volatilità | ATR, Bollinger, Keltner Squeeze |

Avere 4 varianti del momentum oscillatore equivale a votare 4 volte la stessa informazione. Il consenso 60% diventa più facile da raggiungere non perché ci siano più evidenze, ma perché c'è più rumore correlato.

### 2.3 Il ML duplica i moduli tecnici anziché aggiungere informazione

Le feature del FeatureVector includono RSI, MACD, Stochastic, ADX, distanze dalle EMA — le stesse variabili che già producono segnali separati nei 14 moduli. Poi il segnale ML rientra nel Consensus insieme ai moduli tecnici.

Questo è **double counting strutturale**: la stessa informazione di prezzo passa attraverso due percorsi (moduli tecnici → consenso, feature ML → segnale ML → consenso) e viene sommata come se fossero evidenze indipendenti.

### 2.4 Cicli di feedback circolari

Il DynamicVote aggiorna i pesi in base all'accuratezza storica dei moduli. Ma l'accuratezza è misurata rispetto al **risultato del trade**, che dipende da SL/TP, ExitEngine, condizioni di mercato. Un modulo che ha prodotto un segnale corretto ma il trade è stato stoppato da volatilità si vedrà punire il peso nonostante la previsione fosse corretta.

Il ciclo è:
```
peso_modulo → consensus → ordine → trade outcome (funzione di SL/TP/regime)
    ↑                                          │
    └──────── accuracy_ema update ←────────────┘
```

Il feedback mescola qualità del segnale e qualità dell'esecuzione. Non si può imparare niente di pulito da questo.

### 2.5 Componenti non coerenti con i timeframe M1/M5

Il sistema gira su M1/M5/M15/M30. Diversi moduli sono stati progettati concettualmente per H1 o superiori e su M1 producono prevalentemente rumore:

- **Hurst Exponent su 100 barre M1** = 1h40m di dati. L'esponente di Hurst su 100 minuti è statisticamente privo di significato.
- **Pivot Points su M1**: le sessioni giornaliere diventano 1440 barre. La logica di "sessione precedente" diventa complicata e la rilevanza dei pivot diminuisce drasticamente.
- **News Calendar con finestra ±60 min**: su M1 è utile. Su M30 è 2 barre.
- **Seasonality con momentum a 3 barre M1** = 3 minuti di momentum. Triviale.

---

## 3. Review modulo per modulo

| Modulo | Verdetto | Motivazione |
|--------|----------|-------------|
| **RSI** | KEEP WITH CHANGES | Valido, ma ridondante con Adaptive RSI. Tieni uno solo. Su M1 aggiunge rumore. Usa periodo 21 su M1 invece di 14. |
| **Adaptive RSI** | KEEP WITH CHANGES | Concettualmente superiore al RSI fisso perché adatta le soglie al regime. Tieni questo, rimuovi RSI classico. |
| **MACD** | KEEP WITH CHANGES | Ridondante con EMA Crossover: entrambi usano EMA differenziali. Tieni MACD, rimuovi EMA Crossover. Il confidence scaling con `histogram × 500` è arbitrario — serve calibrazione. |
| **ATR Module** | KEEP | Utilità reale come fornitore di metadata (SL, volatilità). Non direzionale. Non introduce rumore. |
| **ADX Module** | REMOVE | Completamente assorbito da Market Regime Module che usa ADX, +DI, −DI e in più Hurst. Duplicato. |
| **Bollinger Bands** | KEEP WITH CHANGES | Valido per mean-reversion. Ma su M1 genera segnali ogni pochi minuti: serve filtro aggiuntivo (es. solo in RANGING regime). |
| **EMA Crossover** | REMOVE | Ridondante con MACD. Entrambi misurano il momentum delle MA. Tieni MACD che ha anche la signal line. |
| **Stochastic RSI** | REMOVE | RSI di un RSI: tripla derivazione del prezzo. Su M1 è quasi rumore puro. Genera segnali frequenti e poco affidabili. |
| **Keltner Squeeze** | REMOVE | Ridondante con Bollinger Bands: entrambi misurano compressione/espansione di volatilità. Il Keltner aggiunge un secondo canale, ma l'informazione è la stessa. |
| **Donchian Breakout** | KEEP WITH CHANGES | Il breakout a 20 barre su M1 = 20 minuti. Troppo breve per essere significativo. Su M15/M30 ha più senso. Usa periodo 50 su M1. |
| **Market Regime** | KEEP | Il modulo più utile. Classifica il regime e condiziona gli altri. Correggi Hurst (usa finestra 200 barre minimo, meglio 500). |
| **Pivot Points** | KEEP WITH CHANGES | Valido come S/R livelli giornalieri, ma su M1 la gestione delle sessioni è delicata. Assicurati di calcolare i pivot sulla sessione corretta, non sulla finestra a 24 barre. |
| **Multi-TF** | KEEP WITH CHANGES | L'idea è giusta — il trend su TF superiori conta. Ma il resampling in-process da M1 è approssimativo. Usa barre M15/M30 reali dalla sorgente dati invece di ricalcolare. |
| **Swing Level** | REMOVE | Ridondante con Pivot Points e Donchian. Tre moduli che identificano supporto/resistenza dinamico. Tieni solo uno. |
| **Seasonality** | REMOVE | Le fasce orarie sono configurazioni empiriche non robuste. Il momentum a 3 minuti è triviale. Inserisci il filtro sessione direttamente nel Runner come pre-condizione, non come modulo di voto. |
| **News Calendar** | REMOVE | Hardcoded con orari statici. Non copre gap imprevedibili (flash news, interventi verbali, dati revisionati). La finestra ±60 min blocca trade in momenti in cui potrebbe esserci il miglior movimento direzionale. Usa invece un semplice filtro spread: se spread > X, non tradare. |

**Dopo questa review**: da 16 moduli a **6 moduli** (Adaptive RSI, MACD, ATR, Bollinger, Donchian, Market Regime + Multi-TF come filtro, non come modulo).

---

## 4. Critica dei sistemi chiave

### 4.A Consensus Engine

**Problema principale: indipendenza assunta, non verificata.**

La formula `buy_pct = weighted_score(BUY) / total` implica che i voti siano statisticamente indipendenti. Con i cluster di ridondanza identificati, la correlazione tra i voti può essere 0.7–0.9. Un consenso del 70% potrebbe derivare da un solo cluster di 4 moduli correlati, non da 4 fonti indipendenti.

**Soglia 60% — arbitraria.**
Non c'è evidenza che 60% sia ottimale o anche solo sensata. Su M1, con 14 moduli rumorosi, questa soglia viene raggiunta spesso per pura correlazione stocastica. Non è stato fatto alcun test per confrontare 50%, 60%, 70%, 75%.

**Il DynamicVote è pericoloso se non validato.**
Il meccanismo EMA dei pesi ha un warm-up lento (α=0.10 → 10 trade per adattarsi). Durante i primi 30-50 trade, i pesi sono essenzialmente casuali rispetto alla loro destinazione finale. Un cambio di regime durante il warm-up può portare a pesi che amplificano moduli inadatti per il nuovo contesto.

**Suggerimento**: Sostituisci DynamicVote con una regressione logistica allenata offline sul segnale aggregato. I pesi fissi appresi su dati storici sono più difendibili di pesi adattivi in tempo reale senza validazione.

### 4.B Adaptive Threshold Manager

**Il meccanismo è statisticamente fragile.**

`delta = −0.01 × (score − 0.5) × 2`

Con α=0.01, ci vuole 1 trade per ridurre la soglia di 0.01. Se un modulo produce 20 segnali corretti di fila (cosa possibile per pura fortuna in un trend forte), la sua soglia scende di 0.20. Poi il regime cambia e il modulo inizia a sbagliare, ma la soglia è già bassa.

**Il problema più grave**: il "score" con cui si aggiorna la soglia è il risultato del trade, non l'accuratezza predittiva del segnale. Un segnale BUY corretto che produce una perdita per un'uscita anticipata dell'ExitEngine abbassa comunque il peso del modulo. Non si può imparare la qualità del segnale da un outcome confuso con l'esecuzione.

**Eliminala per ora.** Usa soglie fisse per confidence (0.60 globale) e non adattarle finché non hai abbastanza dati per fare una stima significativa per modulo (almeno 200 segnali per modulo per timeframe).

### 4.C ML Classifier

Questa è la sezione più critica.

**Problema 1: leakage nel processo di selezione.**

Il processo di auto-tune:
- 20 outer attempts (adaptive loop)
- × 20 inner trials (grid search forward_bars × atr_mult)
- = potenzialmente 400 walk-forward sullo stesso dataset

La selezione del "miglior modello" avviene su un holdout che è stato indirettamente visto 400 volte. Questa non è validazione out-of-sample: è selezione ottimistica da un pool di 400 esperimenti. Il fenomeno è noto come **multiple testing inflation**: con 400 trial, un sistema puramente casuale avrà senza dubbio alcuni holdout > 55%.

**Fix richiesto**: Tieni una finestra di dati completamente bloccata (es. ultimi 6 mesi) che non viene mai toccata dal processo di training/tuning. La selezione del modello avviene su validation. La stima delle performance avviene sull'holdout finale. Un solo utilizzo.

**Problema 2: l'etichettatura è discutibile su M1.**

`label = BUY se return(t→t+5min) > ATR(14) × 0.8`

Su M1, ATR(14) è la volatilità media dei 14 minuti precedenti. Con EURUSD e spread di 1.5 pip, la soglia effettiva del BUY è spesso 2–4 pip. Con un rumore tick-level tipico di 1–2 pip su M1, stai classificando come BUY movimenti che sono al confine del rumore. La percentuale di HOLD sarà dominante (spesso 60–70%), il che significa che un modello che predice sempre HOLD avrà accuracy del 60–70% senza imparare niente.

**Fix**: Allunga forward_bars a 15–30 su M1 (15–30 minuti), oppure lavora su M5 con forward_bars=6–12 (30–60 minuti). Verifica la distribuzione delle label: se HOLD > 60%, il problema è mal formulato.

**Problema 3: double counting con i moduli tecnici.**

Le feature includono RSI, MACD, EMA, ADX. I moduli tecnici producono segnali da RSI, MACD, EMA, ADX. Il segnale ML rientra nel consenso accanto ai moduli tecnici. L'informazione RSI passa due volte: una come feature del modello, una come segnale del modulo RSI. Questo non è diversificazione: è correlazione strutturale mascherata.

**Fix**: Le feature del ML devono essere disgiunte dai moduli tecnici, oppure il ML sostituisce i moduli tecnici — non si affianca.

**Problema 4: calibrazione assente.**

`confidence = max(predict_proba(X))`

Le probabilità di un gradient boosting non calibrato su dati finanziari (rumorosi, non stazionari) sono sistematicamente sovrastimat o understimat. Usarle direttamente come confidence nel voto è metodologicamente sbagliato.

**Fix**: Calibra le probabilità con Platt scaling o isotonic regression su un validation set separato. Valuta ECE (Expected Calibration Error) come metrica.

**Problema 5: la metrica di selezione è sbagliata.**

Usi holdout accuracy (o signal_precision) per selezionare il modello. Entrambe sono metriche di classificazione, non di trading. Un classificatore che predice BUY su movimenti di 2 pip e SELL su movimenti di 2 pip può avere accuracy 55% e Sharpe ratio negativo dopo costi di transazione.

**Fix**: La metrica di selezione deve essere **P&L simulato su holdout dopo costi di transazione**, non accuracy classificatoria. In alternativa: precision@K (tra le previsioni più confident, quante sono corrette?).

**Problema 6: il labeling usa il futuro.**

`ATR(14)[i]` nel labeling usa le 14 barre precedenti alla barra i — OK. Ma `close[i+forward_bars]` usa il futuro — per definizione, nel training è lecito. Il rischio è che le feature includano indicatori che "vedono" implicitamente informazioni future. Verifica che nessun indicatore nel FeatureVector usi informazioni post-timestamp.

**Problema 7: nessuna considerazione sui costi di transazione nel training.**

Il label `BUY se return > ATR×0.8` non considera che entri a ask e esci a bid. Sul M1 EURUSD con spread 1–2 pip e ATR M1 di 3–5 pip, il costo round-trip (2–3 pip) è il 40–60% della soglia di labeling. Stai allenando il modello su movimenti che esistono teoricamente ma non sopravvivono dopo i costi reali.

### 4.D Risk Manager

**SL via Chandelier su M1: troppo stretto e troppo largo simultaneamente.**

ATR(14) su M1 misura la volatilità dei 14 minuti precedenti. Con EURUSD tipicamente a 2–5 pip di ATR M1, uno SL a 2×ATR è 4–10 pip. Questo è:
- Troppo stretto: un normale rimbalzo di mercato può colpire uno SL a 5 pip anche in una direzione favorevole
- Troppo largo: su M1, uno SL a 10 pip implica un R:R di 1:2 con TP a 20 pip, che richiede un movimento di 20 minuti sostenuto nella direzione prevista — improbabile con l'orizzonte temporale del sistema

Il problema è che il SL viene calcolato sull'ATR al momento dell'ingresso, ma la posizione viene gestita minuto per minuto. L'ATR può raddoppiare in 10 minuti su M1 durante una release macro.

**TP fisso 2:1 — rigido e spesso sbagliato.**

Un TP fisso a 2×SL distanza ignora la struttura del mercato: livelli S/R, distanza dal TP successivo, momentum. Se il TP è a 20 pip ma c'è una resistenza a 12 pip, il trade probabilmente non raggiungerà mai il TP. Il mercato non rispetta R:R arbitrari.

**Fix**: Il TP deve essere un candidato iniziale, non un livello fisso. L'ExitEngine dovrebbe gestire l'uscita dinamicamente. Un TP fisso 2:1 non è compatibile con un ExitEngine sofisticato — o usi il TP fisso, o lasci all'ExitEngine il controllo dell'uscita. Fare entrambi crea conflitti.

**Daily drawdown kill al 5% — arbitrario ma non sbagliato.**

Il valore è difendibile come limite operativo. Il problema è che scatta a livello globale, non per simbolo. Se stai tradando EURUSD e GBPUSD e il portfolio perde 5%, smetti su entrambi. Ma se la perdita è concentrata su GBPUSD, stai impedendo a EURUSD di recuperare.

**Cooldown a 3 candele su M1 — praticamente inutile.**

3 minuti di cooldown non impediscono overtrading. Su M1, dopo una perdita, il sistema può riaprire un trade in 3 minuti nella stessa direzione. Il cooldown deve essere adattivo alla volatilità: almeno 1 ATR period (14 barre) di pausa.

**Mancano controlli fondamentali:**
- Nessun limite di perdita giornaliera per simbolo (solo globale)
- Nessun max loss streak (es. blocco dopo 5 trade persi di fila)
- Nessun limite di frequenza di trading (si può aprire 50 trade al giorno su M1 se tutti passano i check)
- Nessuna correlazione tra SL e volatilità implicita (bid-ask spread al momento dell'ingresso)

### 4.E Exit Engine

**Il reputation model impara su dati insufficienti.**

Con cold_start_trades=10 e learning_rate=0.15, dopo 50 trade il peso di una regola ha un intervallo di confidenza enorme. Non hai potere statistico per distinguere una regola buona da una rumorosa. Stai letteralmente imparando da 10 esempi se TrailingStop è meglio di GiveBack.

Il peso si muove da 50 verso qualcosa in 20-30 trade. In 30 trade su M1, hai probabilmente attraversato solo 1-2 giorni di mercato. Non è abbastanza.

**BreakEvenRule non è una regola di uscita — è una modifica dello SL.**

La regola emette HOLD ma modifica la variabile `suggested_sl`. Questo è un side effect mascherato da voto. L'architettura a voto ponderato assume che ogni regola produca un voto pulito (HOLD/PARTIAL/FULL). BreakEvenRule viola questo contratto: vota HOLD ma nel frattempo sposta lo SL. Se questa regola viene rimossa o il suo peso crolla, la logica di break-even scompare silenziosamente.

**GiveBack e PartialExit possono creare comportamenti instabili.**

Scenario: PartialExit chiude il 33% a +20 pip. Il 67% rimanente continua. GiveBack scatta quando il floating profit scende del 40% dal picco — ma il picco era calcolato sull'intera posizione, non sul 67% rimanente. Il calcolo del picco dopo una chiusura parziale deve essere normalizzato. Se non lo è, GiveBack può scattare troppo presto o troppo tardi.

**SetupInvalidation e VolatilityExit si sovrappongono.**

Entrambe chiudono su "il mercato si è mosso troppo contro la mia tesi". La differenza è sottile (prezzo vs ATR). In condizioni di alta volatilità, entrambe sparano quasi simultaneamente, creando ridondanza nel voto.

**Suggerimento concreto**: Riduci l'ExitEngine a 3 regole — TrailingStop, TimeExit, GiveBack. Rimuovi reputazione finché non hai almeno 500 trade per regola. Aggiungi le regole aggiuntive solo con evidenza quantitativa che migliorano il Sharpe.

### 4.F Intermarket Engine

**Su M1, la correlazione rolling a 200 barre è inutile.**

200 barre M1 = 3 ore e 20 minuti. La correlazione tra EURUSD e GBPUSD calcolata su 3 ore di M1 è statisticamente rumorosa: può essere 0.95 in un'ora e 0.20 in quella successiva durante una news release. Usare questa correlazione come moltiplicatore di rischio introduce volatilità nel sizing che non corrisponde a rischio reale.

**La currency exposure è concettualmente corretta ma prematura.**

L'idea di tradurre posizioni in esposizione netta per valuta è corretta. Ma per funzionare correttamente richiede:
1. Tutti i simboli attivi con posizioni aperte in tempo reale
2. Una stima affidabile del pip value per ogni coppia
3. Un aggiornamento coerente tra apertura e chiusura di ogni posizione

Su un sistema M1 che apre e chiude rapidamente, questa contabilità deve essere perfettamente sincronizzata. Il rischio di bug silenzioso (exposure non aggiornata correttamente) è alto.

**Per ora: prematura e non necessaria per EURUSD singolo.**

Se stai tradando solo EURUSD, l'IntermarketEngine non serve. Serve quando gestisci un portafoglio multi-simbolo reale. Tienila come architettura, ma disabilitala finché non hai validato il sistema su un singolo simbolo.

---

## 5. Errori metodologici e rischi di overfitting

### 5.1 Nessuna baseline senza adattamento

Non esiste una versione del sistema a parametri fissi, senza alcun adattamento, su cui misurare l'edge di base. Senza questa baseline, non si può sapere se l'adattamento aggiunge valore o semplicemente maschera l'assenza di edge.

**Prima di tutto il resto**: esegui un backtest con parametri fissi, senza DynamicVote, senza AdaptiveThreshold, senza ML, senza ExitEngine sofisticato. Solo 5-6 moduli con pesi fissi e TrailingStop. Misura Sharpe ratio, max drawdown, win rate. Se questa versione non funziona, aggiungere complessità non la farà funzionare.

### 5.2 Multiple testing senza correzione

Il processo di auto-tune testa 20 combinazioni (forward_bars × atr_mult) × 4 backend = 80 esperimenti. La probabilità che almeno uno di essi sembri buono per pura fortuna è alta. Con α=0.05 e 80 test, ci aspettiamo 4 falsi positivi.

Non c'è nessuna correzione per multiple testing (Bonferroni, Holm, BH). Il "miglior trial" è selezionato senza aggiustare per il numero di confronti.

### 5.3 Regime overfitting nei dati di training

Il dataset EURUSD M1 2025-2026 (periodo del training) corrisponde a un regime specifico: tassi alti, bassa liquidità, movimenti direzionali limitati. Un modello trainato su questi dati potrebbe essere ottimizzato per questo regime. Quando il regime cambia (es. tassi calano, volatilità aumenta), il modello fallisce sistematicamente.

Non c'è test su dati di regime diverso.

### 5.4 I pesi adattativi non sono riproducibili

Il DynamicVote e l'AdaptiveThreshold hanno uno stato interno che dipende dalla sequenza dei trade passati. Questo stato non è salvato in modo robusto (non è incluso nel ModelRegistry). Dopo un riavvio, il sistema riparte con pesi neutri.

Implicazione: **il backtest non è riproducibile in condizioni di live**. Il sistema in live al giorno 100 ha pesi completamente diversi dal backtest al giorno 100 se la sequenza di trade è diversa anche solo parzialmente.

### 5.5 Transaction cost assenti nella selezione del modello

L'holdout accuracy non include spread, slippage, commissioni. Su M1 con segnali frequenti, i costi di transazione possono azzerare un edge del 2-3% su base annua. Un modello con holdout accuracy 54% potrebbe essere perdente netto dopo costi.

### 5.6 Look-ahead bias potenziale nella normalizzazione

Se qualsiasi indicatore nel FeatureVector usa statistiche globali (es. mean/std calcolato sull'intero dataset per la normalizzazione dei feature), questo introduce look-ahead bias. Le statistiche di normalizzazione devono essere calcolate solo sui dati di training, poi applicate al test set.

### 5.7 I pesi del DynamicVote dipendono dal futuro nel backtest

Nel backtest, il DynamicVote al bar t usa accuracy_ema[module] calcolata su trade passati. Ma "trade passati" include la sequenza completa di trade che dipende da decisioni future (es. se un trade è aperto e non ancora chiuso, il suo outcome non è ancora noto). Come viene gestito questo nel backtest? Se il sistema chiama `on_eval` solo a chiusura del trade, i pesi durante la vita di una posizione aperta sono "congelati" — ma questo deve essere esplicitamente verificato.

---

## 6. SE DOVESSI TAGLIARE IL 30% DEL SISTEMA

Nell'ordine:

1. **ADX Module** (duplicato di Market Regime) → rimuovi
2. **EMA Crossover** (duplicato di MACD) → rimuovi
3. **Stochastic RSI** (derivativo del derivativo, rumoroso su M1) → rimuovi
4. **Keltner Squeeze** (ridondante con Bollinger) → rimuovi
5. **Swing Level** (ridondante con Pivot + Donchian) → rimuovi
6. **Seasonality Module** (hardcoded, empirico, non robusto) → rimuovi, sostituisci con filtro sessione nel Runner
7. **News Calendar** (statico, falsa sicurezza) → rimuovi, sostituisci con filtro spread dinamico
8. **AdaptiveThresholdManager** (non validato, introduce instabilità) → rimuovi, usa soglie fisse
9. **ExitEngine ReputationModel** (dati insufficienti per imparare) → rimuovi, usa pesi fissi per le regole
10. **4 backend ML in parallelo** (multiple testing) → scegli uno (LightGBM), valida, poi eventualmente confronta

Risultato: 16 → 6 moduli tecnici, zero adattamento online incontrollato, sistema molto più testabile.

---

## 7. Top 10 cambiamenti ad alto impatto

### #1 — Valida l'edge di base prima di tutto
**Problema**: Non sai se c'è un segnale reale dopo costi di transazione.
**Perché è grave**: Tutto il resto — adattamento, ML, exit engine — è costruito su sabbia se non c'è edge.
**Soluzione**: Backtest con 5 moduli fissi (Adaptive RSI, MACD, ATR, Bollinger, Market Regime), pesi uniformi, trailing stop ATR×2, zero adattamento. Misura Sharpe ratio, Calmar ratio, max drawdown. Se Sharpe < 0.5 dopo costi, fermati.
**Priorità**: CRITICA

### #2 — Correggi il leakage nel processo di selezione ML
**Problema**: 400 walk-forward sullo stesso dataset, holdout visto più volte.
**Perché è grave**: Le performance ML riportate sono ottimisticamente distorte. Il modello in live perderà.
**Soluzione**: Blocca gli ultimi 6 mesi come test set assoluto. Non toccarli mai durante training/tuning. Usa i restanti dati per train/validation. L'holdout finale si valuta una sola volta.
**Priorità**: CRITICA

### #3 — Riduci i moduli a 6 decorrelati
**Problema**: 14 moduli con ridondanza estrema. Il consenso conta la stessa informazione 3-4 volte.
**Perché è grave**: Falsa diversificazione, soglia 60% raggiunta troppo facilmente da cluster correlati.
**Soluzione**: Tieni un modulo per cluster (Adaptive RSI, MACD, ATR, Bollinger, Donchian, Market Regime). Misura correlazione tra segnali dei moduli: se > 0.5, rimuovi il più debole.
**Priorità**: CRITICA

### #4 — Separa le feature ML dai moduli tecnici
**Problema**: Double counting strutturale: RSI entra come feature ML E come segnale modulo.
**Perché è grave**: Il segnale ML non aggiunge informazione indipendente, amplifica solo il rumore dei moduli.
**Soluzione**: Le feature ML devono usare prezzi grezzi, microstructure features (spread, volume), e indicatori non usati dai moduli attivi. Oppure il ML sostituisce tutti i moduli tecnici.
**Priorità**: ALTA

### #5 — Rimuovi il TP fisso 2:1 o rendilo dinamico
**Problema**: Un TP a distanza fissa ignora la struttura del mercato.
**Perché è grave**: Il sistema sceglie se uscire basandosi su una distanza arbitraria, non su informazione di mercato.
**Soluzione**: Rimuovi il TP fisso. Lascia che l'ExitEngine gestisca l'uscita con TrailingStop + GiveBack. Oppure usa come TP il prossimo livello S/R significativo (dai Pivot Points).
**Priorità**: ALTA

### #6 — Aggiungi embargo tra train e test nel walk-forward
**Problema**: Il walk-forward usa barre consecutive: la barra finale del training è la barra prima del test. Su M1, le barre consecutive sono autocorrelate.
**Perché è grave**: Introduce leakage sottile: le ultime barre del training e le prime del test condividono informazione.
**Soluzione**: Inserisci un embargo di `forward_bars` barre tra fine training e inizio test. Su M1 con forward_bars=5, escludi 5 barre tra train e test.
**Priorità**: ALTA

### #7 — Calibra le probabilità del ML
**Problema**: Le predict_proba del gradient boosting non sono probabilità calibrate.
**Perché è grave**: Usarle come confidence nel consenso senza calibrazione distorce i pesi.
**Soluzione**: Implementa Platt scaling (calibrazione isotonica) su un validation set dedicato. Valuta ECE. Usa le probabilità calibrate nel consenso.
**Priorità**: ALTA

### #8 — Misura Sharpe ratio e non solo accuracy come metrica di training
**Problema**: Un classificatore con accuracy 55% può avere Sharpe negativo dopo costi.
**Perché è grave**: Stai ottimizzando la metrica sbagliata. Il modello "migliore" per accuracy potrebbe essere il peggiore per P&L.
**Soluzione**: Nella selezione del modello, simula il P&L sull'holdout con costi di transazione realistici (spread + slippage). Seleziona il modello con Sharpe massimo, non accuracy massima.
**Priorità**: ALTA

### #9 — Rimuovi l'AdaptiveThresholdManager o ridesignalo
**Problema**: Aggiorna le soglie su signal outcome confuso con trade outcome.
**Perché è grave**: Impara la qualità sbagliata. In un trend forte, tutti i moduli sembrano buoni e le soglie crollano; poi il regime cambia e i moduli sopravvalutati producono segnali inaffidabili.
**Soluzione**: Rimuovilo. Se vuoi adattamento per modulo, usa una soglia fissa per 6 mesi, poi rivaluta offline i moduli su dati storici e aggiorna manualmente.
**Priorità**: MEDIA

### #10 — Alloca budget di complessità in modo esplicito
**Problema**: Il sistema ha 14 moduli + 4 layer adattativi + 7 regole exit + ML + intermarket. Non c'è nessun principio che guidi l'aggiunta di componenti.
**Perché è grave**: Ogni nuovo componente aggiunge parametri liberi e riduce la generalizzazione.
**Soluzione**: Stabilisci un budget di parametri liberi massimo (es. 20 iperparametri totali). Per ogni nuovo componente che supera il budget, uno esistente deve essere rimosso o semplificato.
**Priorità**: MEDIA

---

## 8. Proposta MetaTrade v2 semplificata

### Principi guida

- **Meno è più**: ogni componente deve dimostrare il suo valore marginale in isolamento prima di essere integrato.
- **Validazione prima dell'adattamento**: nessun meccanismo adattivo finché il sistema fisso non mostra edge.
- **Separazione netta**: il modello ML è un layer separato dal sistema tecnico, non integrato nel consenso.
- **Riproducibilità**: il backtest deve essere bit-for-bit riproducibile. Nessuno stato nascosto.

### Architettura v2

```
BAR OHLCV (M5 o M15 — non M1 come primario)
    │
    ▼
[FILTRO SESSIONE]
    Blocca: weekend, 00–07 UTC, Ven >18:00 UTC
    Blocca: spread > soglia_volatilità
    (nessun modulo di voto — è un gate binario)
    │
    ▼
[5 MODULI TECNICI — pesi fissi, non adattativi]
    1. Market Regime (ADX + ATR ratio)      → classifica regime
    2. Adaptive RSI                          → momentum oscillatore
    3. MACD                                  → trend/momentum
    4. Bollinger Bands                       → mean-reversion
    5. Donchian Breakout (period 50 su M5)  → trend-following
    │
    ▼
[CONSENSUS — SimpleVote con pesi fissi appresi offline]
    threshold: 65% (più conservativo)
    min_signals: 3/5 concordi
    Nessun DynamicVote — pesi fissi rivisti ogni trimestre offline
    │
    ▼
[FILTRO REGIME]
    In RANGING: accetta solo Adaptive RSI e Bollinger (mean-reversion)
    In TRENDING: accetta solo MACD e Donchian (trend-following)
    In VOLATILE: blocca tutto
    │
    ▼
[RISK MANAGER — semplificato]
    SL: ATR(14) × 2.5 (leggermente più largo di prima)
    TP: dinamico — prossimo livello S/R significativo (Pivot/Donchian)
       oppure trailing stop ATR × 1.5 (se preferisci non usare livelli)
    Sizing: fixed-fractional 0.5% per trade (non 1%) — più conservativo
    Cooldown: 1 ATR period (14 barre al TF attivo)
    Daily kill: 3% drawdown (più conservativo del 5% attuale)
    │
    ▼
[BROKER ADAPTER → ordine]
    │
    ▼
[EXIT ENGINE — 3 regole, pesi fissi]
    1. TrailingStop (ATR × 2.0 — ratchet)
    2. TimeExit (max 24 barre su M5, pari a 2 ore)
    3. GiveBack (40% del peak profit)
    Nessuna reputazione — pesi fissi [0.5, 0.3, 0.2]
    Rivalutazione manuale trimestrale
    │
    ▼
[ML — layer separato, non integrato nel consenso]
    Ruolo: filtro di qualità (meta-labeling), non generatore di segnali
    Logica: "dato che il consensus tecnico dice BUY, il ML stima la
             probabilità che questo specifico BUY sia profittevole"
    Training: features di microstructure + regime, NON gli stessi
              indicatori dei moduli tecnici
    Validazione: final holdout bloccato 6 mesi, mai toccato
    Integrazione: se ML_confidence < 0.55 → forza HOLD
                  se ML_confidence ≥ 0.55 → permetti il trade del consensus
```

### Cosa v2 guadagna

- Zero adattamento online non validato
- Moduli decorrelati con ruolo chiaro
- Validazione riproducibile e rigorosa
- ML come filtro aggiuntivo, non come concorrente dei moduli tecnici
- Sistema spiegabile: ogni trade ha una causa chiara e tracciabile
- Meno di 15 iperparametri totali (vs 50+ in v1)

### Roadmap v2

1. **Mese 1**: Backtest v2 con 3 anni di dati M5. Misura Sharpe netto. Se < 0.5, non procedere.
2. **Mese 2**: Paper trading v2 per 60 giorni. Confronta P&L paper vs backtest out-of-sample.
3. **Mese 3**: Valida il layer ML separatamente. Misura se il meta-labeling riduce il numero di trade perdenti senza eliminare quelli vincenti.
4. **Mese 4**: Live con size minima (0.01 lot). Valida slippage, latenza, comportamento in condizioni reali.
5. **Mese 5–6**: Scala size solo se Sharpe live > 0.8 su 3 mesi di trading reale.

---

## 9. Verdetto finale

**SISTEMA PROMETTENTE MA METODOLOGIA DA RIFONDARE**

L'architettura di base ha alcune idee valide — il modello di reputazione per le regole di uscita, la classificazione del regime come filtro, il walk-forward training. Il problema non è l'idea; è l'esecuzione metodologica e la quantità di adattamento online non validato impilato su un edge che non è stato dimostrato.

**Il rischio concreto in live**: il sistema mostrerà buone performance in backtest perché ha 4 layer adattativi che, sul training set, imparano esattamente la struttura dei dati. In live, questi layer inizialmente non hanno stato (ripartono neutri), poi imparano il rumore dei primi giorni. Il risultato tipico è: le prime 2-4 settimane sono caotiche, poi il sistema converge verso qualcosa — ma non si sa se verso il segnale o verso il rumore del periodo live.

**Cosa fare prima di qualsiasi altra cosa**:

1. Congela il codice. Non aggiungere altro.
2. Esegui il backtest di v2 semplificata su 3 anni di M5 con costi di transazione realistici.
3. Se Sharpe > 0.7: procedi con paper trading.
4. Se Sharpe < 0.7: il problema non è l'implementazione ma il segnale. Ripensa il labeling e i moduli.

Non è un sistema da bloccare definitivamente — è un sistema da non mettere in live finché la metodologia di validazione non è rigida come quella di un fondo istituzionale. Quello che hai costruito è un prototipo di ricerca sofisticato. Non è ancora un sistema di trading.
