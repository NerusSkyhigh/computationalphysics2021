Prima, visitare la sezione sull'oscillatore armonico... contiene la desrizione della procedura
che si dovrà usare anche in questo caso!

# init_cond

Questa funzione posiziona le particelle nei punti identificati da un reticolo di Bravais face centered cubic con una base monoatomica. La funzione è implementata in due versioni. La prima identifica con 3 indici (variabili da 1 a 2) tutte le 8 celle. In ogni cella sono posizionati i 4 atomi che essa contiene. La seconda variante (compatibile con il decoratore @njit, a differenza della prima) posiziona le particelle "strato per strato". Le particelle sono posizionate sul piano xy nelle posizioni che gli competono e vengono adeguatamente scambiate negli strati "superiori" (easier to see in the code than to explain) 

# Test_WF_and_MC

Questo programma testa il funzionamento dell'algoritmo di Metropolis. Le particelle sono inizializzate sul reticolo FCC con la funzione initcond. La funzione "WF" calcola la funzione d'onda come stabilito nella consegna e ne restituisce il modulo quadro. La funzione "metropolis" implementa l'omonimo algoritmo. Le 3 coordinate di ogni particella sono memorizzate in un vettore. Iterando sulle coordinate di ogni particella, si produce la nuova configurazione come stabilito dall'algoritmo. Poi viene calcolato il rate di accettazione e stabilito se accettare o no la proposta. Se sì, la funzione restituisce la nuova configurazione e aumenta il numero di proposte accettate, se no mantiene la configurazione originale. La funzione "montecarlo" applica iterativamente l'algoritmo di metropolis, prima in una fase di "equilibrazione" e poi nella fase di simulazione effettiva, più lunga. Ognuna delle fasi contiene un meccanismo adattivo per il parametro delta in modo da stabilizzare il rate di accettazione. 

## Commenti

Questa parte sembra funzionare, il sampling produce effettivamente configurazioni diverse da quella di partenza e il rate di accettazione è stabile nell'intervallo fissato dal processo di adattamento di delta. Si può osservare, commentando la parte di codice del processo di adattamento, che la percentuale di accettazione è parecchio sensibile alla scelta di b e delta. Pertanto penso avere il meccanismo di adattamento sia utile per non doversi preoccupare troppo di delta. Non saprei se esista un modo per verificare la correttezza di questo codice...

# potential_energy

All'interno della funzione "montecarlo" è stata aggiunta una funzione che calcola l'energia potenziale totale, data dal potenziale di Lennard-Jones. Come prescritto, la qunatità (e il suo quadrato) viene cumulata in una variabile scalare durante la simulazione e poi vengono calcolate media e deviazione standard. La funzione "potenziale" usata semplicemente itera su tutte le coppie di particelle da contare, calcola la distanza (con PBC) e poi somma il potenziale della coppia nella variabile che poi viene ritornata. L'energia potenizale ad ogni step è immagazzinata in un vettore per permettere un plot finale.

## Commenti

La cosa sembra funzionare... chiaramente, il potenziale fluttua moltissimo fra le varie configurazioni, come atteso. Quello che mi aspetterei è un valore medio più o meno costante fra varie esecuzioni della simulazione entro l'errore statistico. Questa cosa è stata verificata, direi... Quello che mi aspetterei è che eseguendo più volte la simulazione, i valori ottenuti siano entro 1 sigma circa il 70% delle volte, e tutte le volte capitino entro 4 o massimo 5 sigma... mi sembra che la dispersione dei risultati che ho ottenuto sia compatibile con questa cosa, ma si potrebbe fare una verifica sistematica.

# total_energy

Aggiungere il calcolo dell'enegia cinetica locale... ci sto lavorando...


