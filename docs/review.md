Agisci come un team composto da:

1. un quant researcher molto rigoroso,
2. un ML engineer specializzato in time series finanziarie,
3. un risk manager istituzionale,
4. un software architect senior,
5. un trader sistematico estremamente scettico verso l’overfitting.

Devi fare una review tecnica spietata, concreta e non diplomatica del seguente sistema di trading algoritmico Forex per MT5.

OBIETTIVO:
Voglio che tu mi dica TUTTO ciò che devo:
- cambiare,
- migliorare,
- semplificare,
- eliminare,
- riscrivere,
- validare meglio,
- misurare diversamente,
- rendere più robusto in live.

Non voglio complimenti.
Non voglio una descrizione del sistema.
Non voglio una riscrittura del documento.
Voglio una CRITICA TECNICA PROFONDA, con priorità, rischi, motivazioni e proposte operative.

DEVI ANALIZZARE IL SISTEMA COME SE DOVESSI DECIDERE SE:
- promuoverlo a live trading con soldi reali,
oppure
- bloccarlo perché fragile, sovra-ingegnerizzato, incoerente o illusorio.

### Cosa ti chiedo di fare

Analizza il sistema nei seguenti assi:

## A. Critica architetturale generale
Valuta se l’architettura è:
- troppo complessa,
- ridondante,
- fragile,
- difficile da validare,
- difficile da spiegare ex post,
- troppo piena di euristiche sovrapposte,
- troppo ricca di moduli che introducono rumore invece che edge.

Dimmi chiaramente:
- quali componenti hanno senso,
- quali sono probabilmente inutili,
- quali duplicano altri moduli,
- quali possono introdurre conflitti logici,
- quali sono sospetti di overengineering.

## B. Critica dei moduli tecnici
Per ciascun modulo tecnico:
- valuta se ha senso su EURUSD e in particolare su M1/M5/M15/M30,
- dimmi se è coerente con il regime su cui dovrebbe lavorare,
- dimmi se rischia di essere troppo rumoroso,
- dimmi se genera segnali ridondanti rispetto ad altri moduli,
- dimmi se il suo confidence scoring è arbitrario o sensato,
- dimmi se le soglie sono deboli o mal calibrate,
- dimmi se il modulo andrebbe tenuto, modificato o eliminato.

Per ogni modulo voglio uno di questi esiti:
- KEEP
- KEEP WITH CHANGES
- REMOVE

e una spiegazione tecnica breve ma dura.

## C. Critica del Consensus Engine
Valuta se il Consensus Engine:
- crea vera robustezza oppure falsa robustezza,
- somma moduli correlati come se fossero indipendenti,
- rischia di contare più volte la stessa informazione,
- usa pesi dinamici in modo corretto o pericoloso,
- soffre di feedback loop che rinforzano errori temporanei,
- può diventare instabile nei cambi di regime.

Dimmi:
- se il voto pesato adattivo ha senso,
- se i pesi andrebbero appresi diversamente,
- se servirebbe una penalizzazione per correlazione tra moduli,
- se la soglia 60% è sensata o arbitraria,
- se il consensus andrebbe sostituito con un meta-model.

## D. Critica dell’Adaptive Threshold Manager
Valuta se la logica di aggiornamento delle soglie:
- è statisticamente difendibile,
- è troppo sensibile al rumore,
- rischia di inseguire il passato,
- introduce adattamento utile o overfitting online,
- è troppo lenta o troppo veloce.

Dimmi se:
- va mantenuta,
- va semplificata,
- va sostituita,
- va eliminata del tutto.

## E. Critica del ML Classifier
Valuta in modo severo:
- labeling,
- feature engineering,
- walk-forward,
- auto-tuning,
- backend ML,
- metrica di selezione,
- rischio leakage,
- rischio selection bias,
- rischio multiple testing,
- rischio overfitting da trial multipli,
- rischio di costruire un classificatore senza edge reale.

Voglio che tu mi dica chiaramente:
- se il target BUY / SELL / HOLD è ben definito,
- se forward_bars e atr_threshold_mult sono scelti bene,
- se il problema va formulato come classificazione, ranking, regressione o meta-labeling,
- se le feature sono troppo deboli, troppo correlate o troppo ingenue,
- se il sistema sta probabilmente ottimizzando rumore,
- se usare HistGBM / LightGBM / XGBoost / CatBoost ha davvero senso qui,
- se devo cambiare algoritmo o cambiare approccio.

In particolare dimmi:
- quali metriche devo usare al posto della sola accuracy,
- come cambiare la validazione,
- come evitare leakage e overfitting,
- se devo introdurre “no trade” più aggressivo,
- se devo fare regime-specific models,
- se serve probabilistic calibration,
- se serve meta-labeling.

## F. Critica del Risk Manager
Valuta:
- sizing,
- SL,
- TP,
- RR fisso 2:1,
- ATR multiplier,
- kill switch,
- cooldown,
- spread filter,
- limiti di esposizione.

Dimmi se:
- il risk model è robusto o semplicistico,
- il TP fisso 2:1 è troppo rigido,
- lo SL Chandelier è davvero coerente con l’ingresso,
- il sizing fixed-fractional + vol scaling è sufficiente,
- mancano controlli fondamentali,
- il drawdown kill al 5% è sensato o arbitrario.

Voglio suggerimenti concreti per:
- migliorare il controllo del rischio,
- evitare concentrazione nascosta,
- ridurre il tail risk,
- gestire meglio il live.

## G. Critica dell’Exit Engine
Valuta se l’Exit Engine:
- è realmente utile,
- è troppo complesso,
- è troppo parametrico,
- rischia di adattarsi al rumore,
- ha regole ridondanti o in conflitto.

Per ciascuna exit rule dimmi:
- se ha senso,
- se è misurabile,
- se rischia di peggiorare il sistema,
- se va tenuta o rimossa.

Valuta anche il reputation model:
- è sensato oppure troppo autoreferenziale?
- sta imparando davvero qualcosa o solo reagendo al rumore?
- rischia di creare instabilità?
- è troppo sofisticato rispetto alla qualità del segnale di base?

## H. Critica dell’Intermarket Engine
Valuta se per EURUSD e per il set di simboli previsto:
- l’intermarket engine aggiunge valore reale,
- la correlazione rolling a 200 barre è abbastanza stabile,
- i cluster di rischio sono ben definiti,
- il currency exposure service è corretto concettualmente,
- i veto e i multiplier sono troppo semplici o adeguati.

Dimmi se questa parte è:
- necessaria,
- prematura,
- incompleta,
- utile solo in multi-asset reale,
- inutile nel contesto attuale.

## I. Critica metodologica da quant
Questa sezione è fondamentale.
Voglio che tu cerchi con aggressività tutti i possibili problemi di:
- look-ahead bias,
- data leakage,
- survivorship bias,
- regime overfitting,
- parameter overfitting,
- multiple testing,
- confirmation bias,
- false discovery,
- selection bias sui backtest,
- metriche non adatte al trading reale,
- mismatch tra backtest, paper e live.

Dimmi dove il sistema è più vulnerabile a “sembrare intelligente” senza esserlo davvero.

## J. Critica software e di mantenibilità
Valuta se il sistema è:
- troppo difficile da testare,
- troppo difficile da debuggare,
- troppo poco spiegabile,
- troppo dipendente da parametri artigianali,
- troppo complesso da portare in produzione in modo affidabile.

Dimmi:
- quali parti andrebbero isolate meglio,
- quali interfacce andrebbero semplificate,
- quali moduli dovrebbero essere separati,
- quali log, metriche e audit trail mancano.

## K. Cosa eliminare subito
Fammi una sezione chiamata:

“SE DOVESSI TAGLIARE IL 30% DEL SISTEMA”

in cui mi dici esattamente quali componenti elimineresti per primi per ridurre complessità e overfitting.

## L. Cosa migliorare prima di tutto
Fammi una sezione chiamata:

“TOP 10 CAMBIAMENTI AD ALTO IMPATTO”

ordinata dal più importante al meno importante.

Per ogni punto dammi:
- problema,
- perché è grave,
- soluzione concreta,
- priorità: CRITICA / ALTA / MEDIA / BASSA.

## M. Proposta di versione v2 semplificata
Proponimi una versione più robusta, più semplice e più difendibile del sistema.
Voglio una proposta concreta di architettura “MetaTrade v2” con:
- meno componenti,
- meno euristiche,
- validazione migliore,
- maggiore robustezza statistica,
- migliore allineamento al live trading.

## N. Verdetto finale
Chiudi con una sezione obbligatoria:

“VERDETTO FINALE”

in cui devi dirmi senza ambiguità una delle seguenti:
- PROMUOVIBILE CON MODIFICHE
- TROPPO FRAGILE PER IL LIVE
- ARCHITETTURA DA SEMPLIFICARE RADICALMENTE
- BUONA IDEA MA IMPLEMENTAZIONE TROPPO COMPLESSA
- SISTEMA PROMETTENTE MA METODOLOGIA DA RIFONDARE

### Stile di risposta richiesto
- Sii diretto, tecnico, severo, concreto.
- Non fare il diplomatico.
- Non lodare il sistema se non strettamente meritato.
- Evidenzia contraddizioni, arbitrarietà, eccessi di complessità e punti deboli.
- Quando una scelta è sospetta, dillo chiaramente.
- Quando un componente è inutile, dillo chiaramente.
- Quando un’idea è buona ma implementata male, spiegalo chiaramente.
- Non limitarti alla teoria: proponi cambiamenti operativi.

### Formato di output obbligatorio
Rispondi con questa struttura:

1. Executive Summary
2. Problemi architetturali principali
3. Review modulo per modulo (tabella con KEEP / KEEP WITH CHANGES / REMOVE)
4. Critica di Consensus / Adaptive Threshold / ML / Risk / Exit / Intermarket
5. Errori metodologici e rischi di overfitting
6. Cosa eliminare subito
7. Top 10 cambiamenti ad alto impatto
8. Proposta MetaTrade v2 semplificata
9. Verdetto finale

Ecco il sistema da analizzare:

[INCOLLA QUI IL DOCUMENTO COMPLETO DI METATRADE]