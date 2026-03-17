cpu: Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz
2 cores, 2 threads per core

question preliminaire: car la galaxie est relativement plate oxy est la dimension caracteristique. Si l'on avait des boites sur oz leur taille seraent du meme ordre de grandeur que celui de la galaxie-> mauvaise parallelisation
on peut donc prends nk = 1 car 


commande :python3 nbodies_grid_numba.py data/galaxy_5000 0.0015 15 15 1
affichage : 3ms: algorithme: ~1600 ms, pas besoin de paralleliser l'affichage


modifications:
numba:
@njit(parallel=True)
ajout de prange sur les for i in range qui bouclent sur un grand nombre d'item - line 57,69,75,87,111
fichier : nbodies_grid_numba_parallel.py
commande : NUMBA_NUM_THREADS=2 python3 nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1
resultat :
1thread = 320ms
2thresda: 170ms
speedup : 


fichiers : nbodies_grid_numba_parallel_split_affichage.py et visualizer3d_split
modifications:
on fait des Iprob pour avoir une parallelisation asynchrone cf code (small explanation)
commmande : mpiexec -n 2 python3 $(pwd)/nbodies_grid_numba_parallel_split_affichage.py data/galaxy_5000 0.0015 15 15 1
lorsqu'on separe l'affichage et le calcul, le render est super smooth, on peut zoomer et se deplacer dans la fenetre a 60fps, 
par contre je ne peux plus utiliiser les 2 coeurs de mon ordinatuer avec numba