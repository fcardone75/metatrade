Agisci come un Principal Quant ML Engineer, Senior Python Architect e refactoring lead.

Devi lavorare su un repository reale chiamato MetaTrade.
Il tuo obiettivo NON è fare teoria, ma progettare e implementare una nuova versione del modulo ML già esistente, rispettando l’architettura attuale del repository e introducendo gradualmente una nuova capacità:

1. output probabilistico BUY / HOLD / SELL calibrato
2. secondo output continuo che rappresenti la qualità economica attesa del setup

ATTENZIONE:
- non devi fare un refactor distruttivo
- non devi rompere la compatibilità con il sistema attuale
- non devi riscrivere tutto da zero senza motivo
- devi lavorare per fasi incrementali
- ogni fase deve lasciare il sistema in stato funzionante
- ogni cambiamento deve essere motivato
- ogni scelta deve essere coerente con il codice già esistente

==================================================
1. CONTESTO REALE DEL REPOSITORY
==================================================

Esistono già questi elementi nel repository:

- src/metatrade/ml/labels.py
  - contiene il labeling attuale con funzione tipo label_bars()
  - oggi usa classificazione 3 classi BUY / HOLD / SELL
  - forward_bars = 5
  - threshold = (ATR14 / close) * atr_threshold_mult
  - HOLD se il return assoluto è sotto soglia
  - NON usa triple barrier
  - NON usa TP/SL
  - NON usa spread/costi
  - NON considera il path intra-barra

- src/metatrade/ml/features.py
  - contiene le feature ML attuali
  - esistono due strutture principali:
    - FeatureVector
    - MultiResFeatureVector
  - single timeframe + contesto M5/M15 per alcuni modelli

- src/metatrade/ml/module.py
  - contiene il modulo ML attuale integrato nel sistema
  - oggi produce direction + confidence
  - se confidence < 0.55 forza HOLD
  - il segnale entra nel ConsensusEngine come voto

- MLConfig
  - configurazione ML esistente
  - include forward_bars, atr_threshold_mult e altri parametri attuali

- ModelRegistry
  - salva snapshot di modelli separati per timeframe e simbolo
  - naming stile ml_m1, ml_m5, ecc.

Vincoli architetturali attuali:
- un modello per timeframe e per simbolo
- il modulo ML è un modulo del consensus, non un motore separato
- il RiskManager non usa ancora la confidence ML per il sizing
- esiste un sistema separato per candidati SL/TP, ma è indipendente dal classificatore ML

==================================================
2. STATO ATTUALE DA CUI PARTIRE
==================================================

Target attuale:
- classificazione 3 classi BUY / HOLD / SELL
- horizon fisso = 5 barre
- soglia ATR-scaled
- nessun target continuo

Feature attuali:
- returns multi-lag
- ema distances / ema cross
- RSI / MACD / Stoch / ADX slope
- ATR rel / std / atr zscore
- candlestick body / wick
- tick volume relative
- time cyclic features
- donchian_pos / bb_pct_b / rsi_divergence / hurst_short
- MultiRes context M5/M15

Limitazioni attuali:
- niente spread come feature
- niente cost proxy
- niente intermarket feature nel vettore ML
- nessun output economico atteso
- confidence = sola max class probability
- labeling limitato al close a t+5

==================================================
3. OBIETTIVO DEL LAVORO
==================================================

Devi evolvere il modulo ML esistente in modo repository-aware.

Il nuovo modulo deve arrivare a produrre:

OUTPUT 1
Probabilità calibrate:
- P(BUY)
- P(HOLD)
- P(SELL)

OUTPUT 2
Uno score continuo di trade quality / expected economic value.

Questo nuovo modulo deve:
- restare compatibile con src/metatrade/ml/module.py
- poter convivere inizialmente con la logica attuale
- salvare e caricare i nuovi artefatti via ModelRegistry
- rispettare il modello “uno per timeframe e simbolo”
- essere introdotto in più fasi, non tutto insieme

==================================================
4. VINCOLI DI IMPLEMENTAZIONE
==================================================

Devi rispettare tutti questi vincoli:

- Python 3.12+
- design modulare
- classi piccole e testabili
- no notebook-centric code
- no giant script
- type hints completi
- docstring utili
- logging strutturato
- fail fast su input invalidi
- niente hardcode inutile
- configurazione centralizzata
- separazione netta tra:
  - labeling
  - feature engineering
  - training
  - calibration
  - evaluation
  - inference
  - persistence
- retrocompatibilità dove possibile
- non rompere le interfacce attuali senza proporre piano di migrazione
- codice leggibile e production-grade
- niente segreti hardcoded
- serializzazione artefatti gestita in modo esplicito
- validazione robusta per serie temporali
- evitare leakage e overfitting

==================================================
5. DECISIONE TECNICA RICHIESTA
==================================================

Devi scegliere e implementare la migliore architettura tra queste:

A. classificatore + regressore separato
B. multi-output model
C. classificatore + meta-labeling
D. altra soluzione migliore, se motivata

Non voglio neutralità.
Scegli la soluzione migliore per questo repository e questo sistema.

==================================================
6. QUELLO CHE DEVI PRODURRE
==================================================

Devi fornire:

1. analisi del codice attuale a livello architetturale
2. proposta di modifica repository-aware
3. elenco file da modificare
4. elenco file nuovi da creare
5. classi nuove da introdurre
6. modifiche puntuali a labels.py, features.py, module.py, MLConfig, ModelRegistry
7. strategia di compatibilità con il sistema attuale
8. implementazione del nuovo training flow
9. implementazione del nuovo inference flow
10. persistenza del nuovo secondo output
11. test minimi
12. piano incrementale a fasi
13. criteri di completamento per ogni fase

==================================================
7. FASI OBBLIGATORIE DA RISPETTARE
==================================================

Devi strutturare il lavoro in queste fasi.
NON saltarle.
NON accorparle in modo confuso.
Ogni fase deve:
- avere obiettivo
- file toccati
- classi introdotte
- compatibilità
- test
- definition of done

------------------------------------------
FASE 1 — ANALISI E HARDENING DEL MODULO ATTUALE
------------------------------------------

Obiettivo:
- consolidare l’attuale classificatore
- rendere più chiari contratti dati, config, training/inference separation
- preparare il terreno al secondo output senza introdurlo ancora

In questa fase devi:
- analizzare src/metatrade/ml/labels.py
- analizzare src/metatrade/ml/features.py
- analizzare src/metatrade/ml/module.py
- analizzare MLConfig
- analizzare ModelRegistry

Poi devi proporre e implementare:
- pulizia delle interfacce
- DTO / dataclass / pydantic model per input-output del modulo ML
- separazione tra artifacts di training e predictor di inference
- persistenza esplicita dei metadata del modello
- metadata schema delle feature
- gestione versione modello
- logging migliore
- test minimi sul flusso attuale

Vincolo:
- il comportamento esterno deve restare il più possibile invariato
- il modulo deve continuare a produrre direction + confidence come oggi

Definition of Done Fase 1:
- il sistema continua a funzionare come prima
- il modulo ML è più pulito e testabile
- esiste una base architetturale pronta per il nuovo output continuo

------------------------------------------
FASE 2 — OUTPUT PROBABILISTICO COMPLETO E CALIBRATO
------------------------------------------

Obiettivo:
- evolvere il classificatore esistente per restituire probabilità calibrate BUY/HOLD/SELL
- mantenere retrocompatibilità con direction + confidence

In questa fase devi:
- introdurre una struttura output tipo MlPrediction o equivalente
- aggiungere:
  - probability_buy
  - probability_hold
  - probability_sell
  - selected_direction
  - confidence
  - confidence_margin
- implementare calibration layer
- scegliere e motivare tecnica di calibration
- aggiornare il predictor
- aggiornare ModelRegistry per salvare anche calibratore + metadata
- fare in modo che module.py continui a poter emettere AnalysisSignal compatibile col consensus attuale

Vincolo:
- nessun secondo output continuo ancora
- retrocompatibilità obbligatoria

Definition of Done Fase 2:
- il classificatore produce probabilità calibrate
- il modulo esistente continua a funzionare
- il consensus può ancora ricevere direction + confidence
- esistono test su probabilità e inferenza

------------------------------------------
FASE 3 — INTRODUZIONE DEL SECONDO OUTPUT CONTINUO
------------------------------------------

Obiettivo:
- introdurre un secondo modello o componente che stimi la qualità economica attesa del setup

In questa fase devi:
- definire il miglior target continuo per questo repository
- confrontare almeno:
  - expected future return
  - expected pips
  - expected value in R
  - TP-before-SL probability
- scegliere una soluzione netta
- implementare il relativo label builder
- creare trainer e predictor del secondo output
- integrare il risultato in un output finale tipo:
  - expected_value_score
  - expected_value_r o metrica equivalente
  - quality_bucket

Vincolo:
- in questa fase il secondo output deve essere disponibile come metadata del modulo ML
- NON deve ancora pilotare il sizing del RiskManager
- il sistema deve restare compatibile con il consensus esistente

Definition of Done Fase 3:
- il modulo ML produce due output
- il secondo output è persistito, caricato e disponibile in inference
- l’output è integrabile come metadata senza rompere il sistema

------------------------------------------
FASE 4 — MIGLIORAMENTO DEL LABELING E VALIDAZIONE
------------------------------------------

Obiettivo:
- migliorare la qualità statistica del modulo ML
- superare i limiti del solo target close@t+5
- rafforzare la metodologia di validazione

In questa fase devi:
- criticare l’attuale labeling in labels.py
- proporre V1 e V2:
  - V1 pragmatica: miglioramento minimo difendibile
  - V2 avanzata: triple barrier o approccio equivalente
- spiegare come introdurre spread/cost proxy nel labeling o nella valutazione
- rivedere horizon e soglie
- progettare walk-forward / expanding validation
- definire embargo corretto
- definire metriche appropriate
- ridurre rischio leakage / selection bias / multiple testing

Vincolo:
- la fase può introdurre nuove classi o file, ma deve spiegare come convivere con il labeling legacy
- niente rottura cieca della pipeline esistente

Definition of Done Fase 4:
- esiste una pipeline di validazione seria
- il labeling è documentato, migliorato e confrontabile col legacy
- sono definite metriche robuste e criteri di promozione

------------------------------------------
FASE 5 — INTEGRAZIONE EVOLUTIVA NEL CONSENSUS
------------------------------------------

Obiettivo:
- usare meglio i nuovi output senza rompere il comportamento attuale

In questa fase devi:
- spiegare come module.py deve esporre il nuovo output
- decidere se expected_value_score deve restare metadata oppure entrare nella logica decisionale
- proporre una migrazione graduale del consensus
- spiegare come usare confidence_margin e expected_value_score
- proporre un piano per futuro uso in:
  - trade filtering
  - ranking
  - priorità segnali
  - eventuale future sizing, ma non implementarlo se non necessario

Vincolo:
- questa fase deve essere compatibile con l’attuale ConsensusEngine
- ogni modifica deve avere fallback semplice

Definition of Done Fase 5:
- il sistema ha un percorso chiaro per usare i nuovi output
- non viene rotto il voto attuale
- la migrazione è graduale e controllabile

==================================================
8. FILE E CLASSI: COSA DEVI PROPORRE IN MODO CONCRETO
==================================================

Devi proporre in modo esplicito:

File esistenti da modificare:
- src/metatrade/ml/labels.py
- src/metatrade/ml/features.py
- src/metatrade/ml/module.py
- file dove vive MLConfig
- file dove vive ModelRegistry

File nuovi suggeriti, se servono, ad esempio:
- src/metatrade/ml/contracts.py
- src/metatrade/ml/prediction.py
- src/metatrade/ml/calibration.py
- src/metatrade/ml/trainers.py
- src/metatrade/ml/evaluation.py
- src/metatrade/ml/targets.py
- src/metatrade/ml/artifacts.py

Non sei obbligato a usare questi nomi, ma devi proporre una struttura concreta e coerente col repository.

Per ogni file devi dire:
- scopo
- classi/funzioni principali
- dipendenze
- perché serve

==================================================
9. REQUISITI SUI CONTRATTI DATI
==================================================

Devi introdurre contratti dati chiari.
Per esempio, o equivalente migliore:

- MlFeatureRow
- MlTrainingDataset
- MlClassificationTarget
- MlContinuousTarget
- MlPrediction
- MlModelArtifacts
- MlCalibrationArtifacts
- MlEvaluationReport

Devono essere tipizzati e serializzabili.

==================================================
10. OUTPUT FINALE DESIDERATO
==================================================

L’output finale della tua risposta deve contenere:

1. Executive Summary
2. Analisi del modulo ML attuale
3. Architettura repository-aware proposta
4. Fase 1
5. Fase 2
6. Fase 3
7. Fase 4
8. Fase 5
9. File da modificare
10. File nuovi da creare
11. Classi / contratti dati
12. Esempi di patch o codice Python
13. Test da scrivere
14. Strategia di migrazione
15. Rischi e limiti
16. Verdetto finale

==================================================
11. STILE DI LAVORO OBBLIGATORIO
==================================================

Lavora come se dovessi aprire una PR seria su questo repository.

Questo significa:
- niente risposte vaghe
- niente teoria astratta senza file/classi
- niente riscrittura totale non richiesta
- ogni proposta deve essere collegata ai path reali
- ogni fase deve essere implementabile da sola
- ogni fase deve lasciare il repository in stato coerente

Ora produci una proposta completa, concreta e orientata a implementazione.





----
Adesso trasforma la proposta in un piano da PR incrementali.

Voglio una tabella con una PR per fase.

Per ogni PR indicami:
- nome PR
- obiettivo
- file modificati
- file nuovi
- classi introdotte
- test da aggiungere
- compatibilità backward
- rischio tecnico
- criterio di merge

Poi, per la PR 1, scrivi anche una checklist operativa dettagliata passo-passo.