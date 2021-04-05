#Sceglie il tipo di separatore dei file

set datafile separator ','

#Sceglie l'output immagine png

set terminal png

#Crea degli stili per le linee

set style line 1 lc rgb "black" lw 2

#Figura per la cross section

set output 'c_progr_cross.png'

#Etichette degli assi x e y, titolo, griglia e legenda

set xlabel "Energy (meV)"
set ylabel "Cross Section"

#set xrange [0:10000]

set title "Total H-Kr Cross Section"

set grid

unset key

#Grafico

plot "CrossSection.csv" u ($1*5.9):2 w l ls 1 title "Potential"
