# File Warmup_HO

Questo primo file testa l'algoritmo di metropolis su un oscillatore armonico unidimensionale, usando come
funzione d'onda di prova una gaussiana (quindi le funzioni di prova contengono la soluzione esatta). La 
funzione "WF" calcola la funzione d'onda per un dato valore di x e del parametro variazionale alfa, che in 
questo programma è fissato a priori. La funzione "metropolis" propone una nuova posizione x, calcola la 
funzione d'onda e valuta se accettare o meno la proposta come mostrato a lezione. Le funzioni "p_energy"
e "k_energy" calcolano rispettivamente l'energia potenziale e cinetica per un dato x e la funzione di prova
scelta. La funzione "montecarlo" applica l'algoritmo di metropolis iterativamente. In una prima fase, l'algoritmo
viene applicato fino a che non si raggiunge un certo valore di accuratezza. Quando questo accade, tutti i 
contatori sono resettati e la simulazione vera e propria comincia. Il programma restituisce dei vettori che 
contengono l'energia cinetica, potenziale e totale. Inoltre vengono calcolati valor medio e deviazione standard
per l'energia ottenuta, insieme alla frazione di "proposte" accettate. In questo modo si può studiare come 
i risultati dipendono dal numero di steps, dal parametro variazionale e dal parametro delta. 

Nell'ultima versione è stato aggiunto uno schema adattivo per il parametro Delta, che stabilizza la
frazione di proposte accettate in un range definito. 

## Cosa funziona

Tutto, i risultati per l'energia e la deviazione standard sono corretti. 

# Cosa andrebbe migliorato
Si potrebbe pensare ad un modo alternativo per terminare la fase di "equilibrazione". Si potrebbe anche 
lavorare sull'autocorrelazione.

# File Warmup_Ho_2

Nel file precedente il paramtro variazionale è fissato a mano. In questo programma, la simulazione viene eseguita
per diversi valori del parametro in modo da trovare il valore che minimizza l'energia. Questo è il compito della 
funzione "variational_1". Sono stati rimossi i vettori per il plot dell'energia, che in questo caso non servono.
Il programma restituisce l'energia media e l'errore trovati nella simulazione con ogni valore del parametro variazionale
e ne fa un grafico. Anche la frazione di proposte accettate in ogni simulazione è calcolata e plottata. 

# Cosa funziona

Tutto, il valore minimo dell'energia che viene trovato concide con il valore esatto e viene trovato in corrispondenza
del valore del parametro variazionale che coincide con la soluzione esatta. In corrispondenza di questo valore,
come atteso, anche la deviazione standard è minima (ed è 0, mentre non lo è per altri valori del parametro). 
Nota: lo schema adattivo per il parametro delta non funziona troppo bene con la lunghezza della fase iniziale
stabilita con la deviazione standard... per questo userei un numero di passi fissato in futuro.

# Cosa andrebbe migliorato

 Volendo, si potrebbe usare un reweighting per la procedura variazionale, migliorando la performance. Personalmente, 
 siccome la simulazione impiega 2 secondi con un numero di punti tali da produrre varianze invisibili, non ne vedo 
 la necessità. 

Edit: è vero, non ne vedo la necessità per il caso dell'oscillatore armonico. Nel caso dell'elio, però, la procedura
diventa più lunga e impiega 2 o 3 minuti... un reweighting potrebbe essere interessante forse... nel caso, ho provato 
ad implementare la cosa nel file "Warmup_HO_Reweight"... non è andata troppo bene per ora...

