#Sceglie il tipo di separatore dei file

set datafile separator ','

#Sceglie l'output immagine png

set terminal png

#Crea degli stili per le linee

set style line 1 lc rgb "red" lw 1.4
set style line 2 lc rgb "blue" lw 1.4
set style line 3 lc rgb "green" lw 1.6


#Figura per i livelli energetici

set output 'levels.png'

#Etichette degli assi x e y, titolo, griglia e legenda

set xlabel "x"
set ylabel "E"

set xrange [0:5]

set title "Energy Levels"

set grid

set key outside 

#Grafico

getValue(row,col,filename) = system('awk ''{if (NR == '.row.') print $'.col.'}'' '.filename.'')

p1 = real(getValue(2,1,"Levels.csv"))


f(x) = (x**2)/2 

plot f(x) with lines ls 3 title "Potential", \
p1 with lines ls 2 title "n=1"
