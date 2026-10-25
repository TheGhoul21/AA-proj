# Un programma interpretativo per CNN e pattern matching

Proposta di lavoro — 25 ottobre 2026.

**Inquadramento concordato:** progetto per l'esame di Algoritmi Avanzati,
in dialogo con Foundations of Neural Networks. Il centro è il confronto
tra soluzione classica e soluzione neurale del pattern matching:
specifica, rappresentazione, preprocessing, riuso dei confronti e correttezza.
Le analisi interpretative servono a spiegare questo confronto.

Il primo passo implementato è descritto in
[`algorithmic-comparison-first-results.md`](algorithmic-comparison-first-results.md):
baseline CNN esatta costruita a mano, valutazione esaustiva, checkpoint e
confronto degli scanner a parità di matching esatto.

Il secondo passo è l'[intervento sulla distribuzione dei quasi-match](distribution-intervention-results.md):
il pattern resta fisso, cambiano le frequenze dei negativi difficili, e si
verifica quale informazione dei margini può guidare l'ordine dei confronti.

Questo documento nasce dalla lettura del codice e del paper del progetto,
del piano di ampliamento del 24 ottobre e dei workflow di interpretabilità
di `../../weight-trajectories`. Non riporta nuovi addestramenti: distingue
deduzioni verificabili dal codice, piccoli controesempi controllati e
ipotesi da investigare. I numeri già presenti nel paper non sono stati
nuovamente riprodotti in questa ricognizione.

## La domanda che tiene insieme il progetto

**Quando una rete impara a riconoscere un pattern, quale struttura della
soluzione rappresenta, come la acquisisce e quale parte di quella struttura
si può trasformare in un algoritmo corretto?**

Il parallelo utile con `weight-trajectories` è il percorso dalle prestazioni
alle rappresentazioni, dalla loro evoluzione agli interventi che ne verificano
la funzione. Qui possiamo aggiungere un controllo particolarmente forte:
con alfabeto di quattro simboli e pattern lungo cinque, le finestre possibili
sono soltanto 1.024. Possiamo studiare l'intera funzione appresa.

Occorre distinguere tre oggetti: il classificatore di finestre, il predittore
di tabelle di preprocessing e lo scanner che usa quelle tabelle. Risolvere
uno di questi compiti non dimostra di avere appreso gli altri.

## Primo audit: cosa possiamo già affermare

### 1. L'additività del filtro singolo precede l'apprendimento

In `model.py`, il logit è per costruzione
`b + sum_i w[i, W[i]]`. La riscrittura in penalità vale anche con pesi casuali.
L'esito da spiegare sperimentalmente è il recupero del pattern e la forma
delle differenze tra pesi, non l'emergere dell'additività.

Inoltre, aggiungere una costante `a[i]` a tutti i pesi della riga `i` e
sottrarre `sum_i a[i]` dal bias lascia invariati tutti i logits. Il segno
assoluto dei pesi non identifica quindi un meccanismo unico. Penalità,
margini e risposte alle sostituzioni sono oggetti più adatti al confronto
tra addestramenti. Prima del recupero corretto del pattern, distinguere
penalità rispetto all'argmax e differenze rispetto al pattern bersaglio:
queste ultime possono essere negative.

### 2. Il quasi-collasso del modello ReLU è una domanda sperimentale

Il modello multifiltro può esprimere interazioni tra posizioni. Il paper
riporta una buona approssimazione additiva sui campioni valutati; questo
motiva la verifica su tutte le 1.024 finestre e durante l'addestramento.

Un R² elevato non basta: gli errori residui potrebbero concentrarsi proprio
vicino alla soglia decisionale. Servono errore massimo, disaccordi di
classificazione e analisi delle finestre che producono i residui maggiori.

### 3. La GRU attuale non vede tutta l'informazione richiesta dal target

In `experiment6.py:make_gru_model`, l'uscita al passo `i` dipende soltanto
da `P[:i+1]`. Per Z questo genera un'ambiguità inevitabile:

| Pattern | Prefisso osservato al passo 2 | Z[2] |
|---|---|---:|
| ABAA | ABA | 1 |
| ABAB | ABA | 2 |

La stessa GRU deterministica deve produrre la stessa uscita, ma i target
sono differenti, anche dopo la normalizzazione: le lunghezze sono uguali.
Questi valori sono stati verificati con `patterns.z_array`.

La funzione dei bordi `sp[i]` intera dipende invece dal prefisso disponibile.
Tuttavia, il codice la divide per la lunghezza totale, che non viene fornita
alla GRU prima della fine del pattern: `ABA` e `ABAB` hanno entrambi
`sp[2]=1`, ma target rispettivamente `1/3` e `1/4` allo stesso prefisso.

Per un confronto informativo: predire `sp` intera con una codifica compatibile
con il prefisso, oppure fornire la lunghezza; per Z, leggere l'intero pattern
prima di emettere la tabella. La capacità di generalizzare va misurata dopo
avere rimosso questa ambiguità. Un fallimento dell'MLP sperimentato non
dimostra un'impossibilità per tutti gli MLP: su un dominio finito di pattern
a lunghezza limitata, una rete sufficientemente grande può anche codificare
una tabella di corrispondenze. Questo non equivale ad apprendere un algoritmo
che generalizza a lunghezze nuove.

### 4. Equivalenza dello scanner e accuratezza sul task sono distinte

Ho verificato un controesempio usando le funzioni attuali di `experiment5.py`:
pattern `AB`, penalità 0 sul simbolo corretto e 1 sugli altri, testo `BCB`,
soglia cumulativa `tau=1`. Lo scanner completo accetta la finestra `CB`
all'offset 1. Lo scanner con shift costruiti a `eta=0` la salta; con `eta=1`
la conserva.

Questo non dimostra una perdita di match esatti: `CB` è approssimato. Dimostra
che la conservazione di tutte le accettazioni a soglia positiva richiede
condizioni aggiuntive e non segue dai test attuali a `tau=0`.
Con penalità non negative, `eta >= tau` è una condizione sufficiente per
la regola implementata: ogni allineamento saltato assegna al carattere che
ha provocato lo shift una penalità maggiore di `eta`, dunque di `tau`.
È una condizione sufficiente, non una caratterizzazione necessaria.

Anche lo scaling va interpretato precisamente: costo circa costante per
carattere significa costo totale circa proporzionale a `n`. Il vantaggio
proporzionale a `m` rispetto a una scansione da `n*m` lookups non dimostra
un costo totale `n/m`. Inoltre `run_scaling_benchmark` usa penalità canoniche
costruite a mano: quel risultato non isola un vantaggio dell'apprendimento.

## Quattro filoni di ricerca

### A. Che cosa determina le penalità imparate?

**Domanda:** la gerarchia dei margini riflette i bordi del pattern, la
distribuzione degli esempi negativi o entrambe?

- Tenere fisso il pattern e variare il generatore: fondo uniforme,
  near-miss di prefisso, sostituzioni bilanciate per posizione, negativi
  concentrati su una posizione scelta.
- Controllare numerosità, frequenza dei positivi, peso della loss e budget
  di addestramento. Valutare tutti i modelli sullo stesso insieme esaustivo.
- Ripetere su famiglie di pattern con diversa struttura dei bordi, separando
  per quanto possibile composizione dei simboli e auto-sovrapposizione.
- Registrare frequenze dei negativi, penalità, margini e distribuzione degli
  errori. Usare semi accoppiati e riportare variazioni tra pattern e tra semi.

**Intervento decisivo:** modificare solo la posizione dei negativi difficili.
Se si sposta anche la gerarchia dei margini, il generatore contribuisce
causalmente alla struttura appresa. Questo non esclude un ruolo dei bordi.

**Figura:** stesso pattern, generatori diversi, mappe delle penalità e
risposte alle singole sostituzioni affiancate.

### B. Come si forma la soluzione durante l'addestramento?

**Domanda:** quando viene recuperato il pattern, quando vengono separati i
near-miss e quando la rete multifiltro diventa approssimativamente additiva?

- Salvare checkpoint, incluso quello iniziale, lungo un unico training con
  stato dell'ottimizzatore continuo.
- Misurare recupero del pattern, margini, errori esaustivi e residuo additivo.
- Per il multifiltro, misurare attivazioni e contributi dei singoli filtri;
  confrontare il residuo di ogni contributo con quello della loro somma.
- Non assumere che i numeri dei filtri identifichino lo stesso ruolo tra
  semi: usare firme funzionali sulle medesime finestre.

**Analisi esatta:** sotto la distribuzione uniforme sulle finestre, calcolare
`mu = E[f(W)]`, `g_i(c) = E[f(W) | W_i=c] - mu` e
`r(W) = f(W) - mu - sum_i g_i(W_i)`.
Questa separa componente additiva e interazioni senza dipendere dalle
frequenze accidentali delle finestre nel validation set.

**Figura:** curve di prestazione, margine e quota di varianza delle
interazioni lungo gli stessi checkpoint. Le eventuali fasi di apprendimento
sono un risultato da cercare, non una conclusione da anticipare.

### C. Quali componenti della rappresentazione servono davvero?

**Domanda:** quali dettagli dei pesi modificano la funzione, e quali la
decisione finale?

- Sostituire le penalità con penalità binarie, medie per posizione o medie
  per simbolo, preservando esplicitamente il pattern bersaglio.
- Ablare un filtro alla volta e osservare quali classi di finestre cambiano.
- Confrontare la rete multifiltro con la sua proiezione additiva.
- Separare interventi a soglia fissata da interventi con soglia ricalibrata
  su un insieme dedicato: rispondono a domande diverse.

**Controlli:** filtro casuale, template esatto costruito a mano, interventi
casuali di dimensione comparabile. Un'informazione decodificabile da un
filtro non dimostra da sola che il classificatore la usi.

**Figura:** matrice intervento × classe di errore, includendo distanza di
Hamming, posizione delle sostituzioni e lunghezza del prefisso coincidente.

### D. Quando la rappresentazione diventa una procedura utilizzabile?

**Domanda:** che cosa apportano i valori appresi oltre al pattern recuperato,
e quali proprietà garantiscono l'uso corretto in uno scanner?

- Confrontare pattern noto + preprocessing classico, pattern recuperato +
  penalità binarie e penalità continue apprese, a parità di compito.
- Distinguere correttezza rispetto al classificatore, correttezza del
  matching esatto e qualità rispetto a una definizione di matching rumoroso.
- Per il rumore, derivare le etichette dalla proprietà richiesta su ogni
  finestra finale; distinguere queste etichette dai soli offset di inserimento.
- Misurare lookups, tempo, costo del preprocessing e training separatamente.
- Per tabelle neurali Z/sp, riportare baseline zero, accuratezza sui valori
  non nulli, correttezza dell'intera tabella e correttezza operativa su testi
  con sovrapposizioni. Tenere separate verifica, riparazione e predizione pura.

Solo dopo aver corretto il protocollo ricorrente ha senso aggiungere probe
sugli stati: lunghezza del bordo corrente, carattere atteso, casi di fallback.
Reset dello stato o sostituzioni tra stati possono poi verificare se queste
informazioni sono usate. Una GRU migliore dell'MLP non prova, da sola, che
esegua KMP.

## Ordine di lavoro proposto

1. Consolidare le distinzioni dell'audit e le condizioni di correttezza.
2. Costruire il valutatore esaustivo delle 1.024 finestre e il salvataggio
   dei checkpoint; studiare inizialmente ABABC e ABABA.
3. Eseguire insieme i filoni A e B: variazione controllata dei dati e
   traiettorie delle rappresentazioni, includendo baseline senza training.
4. Scegliere gli interventi del filone C in base ai meccanismi osservati.
5. Approfondire D con risultati e garanzie separati. Trattare la predizione
   neurale di Z/sp come estensione autonoma, con un protocollo corretto.

Il primo nucleo completo è quindi: **dati → traiettorie delle penalità e
delle interazioni → interventi → conseguenze algoritmiche**. Può sostenere
un lavoro interpretativo ampio mantenendo un oggetto sperimentale piccolo.
La tesi finale del paper dovrà seguire gli esiti di questo percorso.
