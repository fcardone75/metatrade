# Plan: dati Forex da Massive.com (sostituto / alternativa a MT5)

## Obiettivo

Consentire il training (e in seguito backtest / walk-forward) usando **serie storiche OHLCV** da [Massive REST Forex](https://massive.com/docs/llms.txt) invece di scaricare le barre da **MetaTrader 5**, con:

- rispetto dei **limiti API** (paginazione, dimensione risposta, rate limit);
- **download organizzato** per simbolo, timeframe configurato (es. M1) e finestra temporale;
- **cache su disco**: stessa “chiave” di dataset → **un solo file** riusabile per più run di training (anche multi-timeframe se il design lo permette, vedi sotto);
- formato preferito **CSV** compatibile con `CsvCollector` (colonne `datetime,open,high,low,close,volume`, UTC).

Quando il dataset in cache è già presente per la stessa chiave (simbolo, TF, finestra), **non si riscarica nulla** salvo comando esplicito di refresh (flag CLI / target Makefile): niente TTL automatico.

Riferimenti API:

- [All tickers](https://massive.com/docs/rest/forex/tickers/all-tickers) — `GET /v3/reference/tickers` (filtro `market`, paginazione `next_url`);
- [Custom bars (Forex)](https://massive.com/docs/rest/forex/aggregates/custom-bars.md) — `GET /v2/aggs/ticker/{forexTicker}/range/{multiplier}/{timespan}/{from}/{to}` (per M1: `1` / `minute`).

---

## Stato attuale nel repo

- `scripts/train.py` supporta `--source csv` e `--source mt5`.
- CSV: `CsvCollector` si aspetta il formato documentato in `csv_collector.py` (default `datetime` + OHLCV).
- Multi-timeframe da file: `--file` con segnaposto `data/EURUSD_{TIMEFRAME}.csv` (`resolve_csv_path`).
- MT5: stima `date_from` da `--bars` e minuti per timeframe (`_TF_MINUTES`).

Nessun integrazione Massive oggi: il piano descrive dove agganciarla.

---

## Vincoli Massive (da rispettare nel design)

| Aspetto | Comportamento atteso |
|--------|------------------------|
| Lista ticker | `limit` default 100, **max 1000**; seguire **`next_url`** fino a esaurimento. |
| Aggregati | Parametro `limit` sulla query (max **50000**, default 5000); finestre lunghe → **spezzare** l’intervallo `[from, to]` in chunk (giorni/settimane) e concatenare risultati ordinati per `t`. |
| Ticker forex | Nella doc delle barre compare il formato tipo **`C:EURUSD`**; allineare il simbolo passato agli aggs a quanto restituisce `/v3/reference/tickers` per `market=fx`. |
| Fuso orario | Le barre forex REST sono in **Eastern Time (ET)** secondo la doc; **normalizzare a UTC** in scrittura CSV per coerenza con `CsvCollector` (“parsed as UTC unless timezone”). |
| Assenza barre | Se non ci sono quote in un intervallo, Massive può **non emettere** barra: si applica la **regola buchi** (sezione sotto). |
| Rate limit | Dipende dal piano; per **Currencies Basic** vedi sotto (**5 richieste/minuto** → throttling obbligatorio, niente parallelismo aggressivo). In ogni caso: **retry con backoff**, rispetto **429** / header di retry. |

Da verificare al momento dell’implementazione: base URL esatta, nome header/query per **API key** (la chiave sarà solo in **env**, vedi sotto).

### Piano di riferimento: Currencies Basic ($0/mese)

Allineato a quanto risulta dal tier **Currencies** “Basic” (forex + crypto, non Stocks):

| Vincolo | Valore | Implicazione per il downloader |
|--------|--------|--------------------------------|
| Chiamate API | **5 / minuto** | Intervallo minimo ~**12 s** tra una richiesta e la successiva (meglio **13–15 s** con margine); **una richiesta alla volta** o conteggio globale rigoroso. |
| Storico | **2 anni** | Non richiedere `from`/`to` oltre la finestra consentita; normalizzare `to` a “oggi” e `from` ≥ `to − 2 anni`. |
| Simboli | **All Forex and Crypto Tickers** | Lista ticker coerente con il piano; il fetch barre è per coppia scelta (es. EURUSD). |
| Risoluzione | **Minute aggregates** (e altro nel piano) | M1 tramite endpoint aggregati minuto; chunk lunghi richiedono **molte** chiamate rispettando il rate cap (download lento ma ripetibile + cache). |

---

## Regola buchi tra barre (default implementazione)

Obiettivo: non “inventare” mercato su fine settimana o lunghi stop, ma **stabilizzare** la griglia quando mancano poche candele (tipico buco API / quote sporadiche).

Dopo download: ordinare per timestamp, **deduplicare** stesso `t`.

1. **Micro-buchi** — numero di periodi mancanti **≤ `MASSIVE_GAP_FILL_MAX_BARS`** (default **5**, configurabile via env): inserire candele **sintetiche** piatte: `open = high = low = close = ultimo close` noto, `volume = 0`. Così il training vede una sequenza regolare in barre per piccole lacune.
2. **Macro-buchi** — gap più lungo del soglia: **nessun** riempimento: si passa direttamente alla barra successiva presente (come dopo un weekend o una lunga assenza di quote). Nessuna riga sintetica per Sabato/Domenica o stop multi-ora.

Per timeframes > M1 la “distanza” attesa tra barre è il multiplo di minuti del TF; la stessa logica usa lo **step atteso** in minuti.

Motivazione: allineato a pratiche comuni (flat fill solo su gap piccoli); evita file enormi di minuti piatti e distorsioni su chiusure di sessione.

---

## Architettura proposta

### 1. Layer “provider” Massive

Modulo dedicato (es. `src/metatrade/market_data/providers/massive/`) con:

- **client HTTP** minimo (timeouts, retry, logging strutturato);
- **`list_fx_tickers(...)`** — wrapper su `/v3/reference/tickers` con `market=fx` (o valore esatto accettato dall’API), paginazione completa;
- **`fetch_aggregates_range(...)`** — per un `forexTicker`, `multiplier`, `timespan`, `from`, `to`, gestione chunk + merge + dedup per `t`;
- mapping **timeframe Metatrade** (`M1`, `M5`, …) → coppia `(multiplier, timespan)` per l’endpoint (M1 → `1` + `minute`; H1 → `1` + `hour` se supportato, altrimenti `60` + `minute` — **da confermare** sulla doc / prova reale).

### 2. Cache file e convenzione percorsi

Directory suggerita: `data/massive/` (o `data/cache/massive/`), con nome file **deterministico**, es.:

```text
data/massive/{SYMBOL}_{TIMEFRAME}_{from}_{to}.csv
```

con `from`/`to` in forma `YYYYMMDD` UTC (o ET se si preferisce coerenza con provider, ma allora documentare e convertire sempre allo stesso modo).

**Manifest opzionale** (JSON accanto al CSV): `request_id`, `resultsCount`, hash parametri, timestamp download, per debug e invalidazione.

**Chiave di deduplica (un solo download per più training)**:

- Stesso `symbol` (normalizzato come Massive), stesso timeframe aggregato, stessa finestra `[from, to]` → stesso path.
- Se due comandi chiedono la stessa finestra ma uno con “ultimi N barre” e l’altro con date fisse: definire una **regola unica** (vedi sotto).

### 3. “Ultimi N barre” vs intervallo assoluto

Oggi MT5 usa `--bars` con `now` come fine. Per Massive:

- **Opzione A (consigliata per ripetibilità)**: calcolare `to = now` (UTC), `from` = `to - N * minuti_TF` (con margine per weekend/assenza barre), poi arrotondare a confini “puliti” se serve chunking.
- **Opzione B**: parametri espliciti `--massive-from` / `--massive-to` (ISO date) e `--bars` solo come stima UI.

La cache include nella chiave **from/to effettivi** (o equivalente deterministica). Due run che vogliono **estendere** la storia fino a “oggi” devono usare una finestra diversa in chiave oppure invocare **`--refresh`** / **`make fetch-massive REFRESH=1`** (o nome concordato) per **forzare** il riscarico.

**Policy adottata**: in generale **mai riscaricare** se il file per quella chiave esiste; solo su **comando esplicito** di refresh (nessun TTL automatico).

### 4. Training multi-timeframe e “un solo file”

Due strategie compatibili con “file unico” dove possibile:

1. **Un file per timeframe** (allineato a `EURUSD_{TIMEFRAME}.csv`): per ogni TF si scarica una serie; la cache evita duplicati tra run. Non è un solo file su disco per tutti i TF, ma **ogni serie si scarica una volta**.
2. **Un solo granulare + resampling locale**: scaricare solo **M1** (o il TF più fine richiesto), derivare M5/M15/… in pipeline prima del training. **Un file M1** può alimentare più TF senza nuove chiamate aggs per TF superiore. Costo: CPU/disco maggiore; beneficio: meno chiamate API e coerenza tra TF.

Il piano raccomanda di implementare prima **(1)** per semplicità e parità con il modello attuale CSV multi-file; valutare **(2)** come ottimizzazione se i limiti API lo richiedono.

### 5. Integrazione con `train.py`

Strade possibili (sceglierne una nell’implementazione):

- **`--source massive`**: internamente chiama il fetcher (cache hit/miss), ottiene path CSV, poi riusa `load_from_csv` / stesso codice di `csv`.
- Oppure **`--source csv`** resta l’unico ingresso training e si aggiunge target Makefile / script **`scripts/fetch_massive_bars.py`** che popola `data/...csv` prima di `make train`.

Vantaggio `--source massive`: un solo comando per l’utente. Vantaggio script separato: training resta offline e testabile senza rete.

Si può supportare **entrambi**: script autonomo + thin wrapper in `train.py`.

### 6. Configurazione

**API key**: obbligatoria per il fetch; **solo** tramite variabile d’ambiente (es. nel `.env`, mai in repo):

```env
# Obbligatoria per download Massive
MASSIVE_API_KEY=...

# Opzionale
MASSIVE_BASE_URL=https://api.massive.com
MASSIVE_HTTP_TIMEOUT_SEC=60
# Con Currencies Basic (5 richieste/min) usare 1
MASSIVE_MAX_IN_FLIGHT=1
MASSIVE_GAP_FILL_MAX_BARS=5
```

Timeframe e simbolo restano come oggi (`SYMBOL`, `TIMEFRAME`, `--timeframes`, …).

### 7. Makefile

Aggiungere target tipo:

- `fetch-massive` — popola cache / CSV per `SYMBOL`, `TIMEFRAME`, finestra (e placeholder per multi-TF);
- `train-massive` — dipende da fetch (o usa `--source massive` integrato).

Documentare in `make help`.

### 8. Test e qualità

- Test unitari del client con **responses mockate** (paginazione ticker, chunk aggregati, merge ordinato).
- Test integrazione opzionale (segno `@pytest.mark.integration`) con chiave reale solo in CI segreta.
- Confronto campione: stesso simbolo/periodo limitato vs export manuale (sanity check OHLC).

### 9. Worker senza master (training senza MT5)

Poiché il training **non** dipende più da MT5 per i dati storici Massive, l’intero ciclo **fetch Massive → CSV in cache → `train.py`** può avvenire **sulla macchina worker** (o su qualsiasi host con rete verso Massive e repo), **senza** coinvolgere il master.

- Il master resta utile per live/paper MT5, Telegram, orchestrazione, ecc., ma **non è richiesto** per preparare o eseguire il training su dati Massive.
- Se si usa ancora **MongoDB / GridFS** per job distribuiti: il worker può generare il CSV localmente prima del training, oppure ricevere un CSV pre-caricato da un job; in entrambi i casi l’artefatto è lo stesso file riusabile.

---

## Ordine di implementazione suggerito

1. Client + fetch aggregati con chunk + scrittura CSV + test mock.
2. CLI `fetch_massive_bars.py` + target Makefile + variabili `.env.example` / `CLAUDE.md`.
3. Integrazione `train.py --source massive` (o documentazione “solo csv precaricato” se si preferisce passo intermedio).
4. (Opzionale) Resampling da un solo TF fine.
5. (Opzionale) Allineamento `backtest.py` / `walk_forward_validation.py` con la stessa sorgente.

---

## Decisioni già prese

- **API key**: solo env (`MASSIVE_API_KEY`).
- **Buchi**: regola micro-fill / macro-no-fill nella sezione dedicata.
- **Cache**: nessun riscarico automatico; solo refresh esplicito.
- **Dove gira il training**: può essere **solo worker** (o dev box), senza master, per il percorso Massive.

## Da chiarire in implementazione

1. **Ticker symbol**: mapping effettivo `EURUSD` ↔ formato Massive (es. `C:EURUSD`) per tutti i simboli usati.
2. **Altri tier**: se in futuro si passa a un piano a pagamento, aggiornare rate limit e `MASSIVE_MAX_IN_FLIGHT` di conseguenza (oggi: **Currencies Basic = 5/min**).

---

## Riepilogo

Il flusso desiderato è: **discovery ticker (fx) → download aggregati per TF e finestra con chunk e retry → CSV in cache → training identico al percorso `--source csv`**. Un solo file per combinazione `(provider, simbolo, risoluzione, finestra)` evita download duplicati tra più esperimenti ML; il multi-timeframe può usare più CSV in cache o, in futuro, un solo download a granularità minima con resampling locale.
