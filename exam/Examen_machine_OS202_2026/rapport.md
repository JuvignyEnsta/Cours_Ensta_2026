# Rapport de TP : Parallélisation de N-Bodies

**Machine de test :** Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz
**Caractéristiques :** 2 coeurs physiques, 2 threads par coeur (Hyper-threading).

## 1. Question Préliminaire

**Analyse de la dimension $N_k$ :**
En observant la forme des galaxies, on constate qu'elles sont relativement "plates". La dimension selon l'axe $Oz$ est négligeable par rapport à l'étalement dans les dimensions charactérisiques $Oxy$.

- Si l'on choisissait une valeur de $N_k > 1$, les boîtes selon $Oz$ auraient une taille du même ordre de grandeur que l'épaisseur de la galaxie elle-même.
- Cela conduirait à une mauvaise parallélisation (déséquilibre de charge), car la majorité des cellules seraient vides, n'apportant aucun gain de performance pour le calcul des interactions.

## 2. Mesure du temps initial (Séquentiel)

- **Affichage (Render time) :** ~3 ms
- **Calcul (Update time) :** ~1600 ms

La partie la plus intéressante à paralléliser est donc le **calcul des trajectoires** (l'algorithme), car il consomme plus de 99% du temps CPU. L'affichage est déjà extrêmement rapide et n'est pas un goulot d'étranglement à ce stade.

## 3. Parallélisation avec Numba (Shared Memory)

### Modifications apportées

- Ajout de l'option `parallel=True` au décorateur `@njit`.
- Remplacement de `range` par `prange` sur les boucles traitant un grand nombre d'items (lignes 57, 69, 75, 87, 111).

### Résultats de performance

_Données : `galaxy_5000`, $dt=0.0015$, grille $(15, 15, 1)$_

| Configuration | Temps de calcul | Accélération (Speedup) |
| :------------ | :-------------- | :--------------------- |
| 1 Thread      | 320 ms          | 1.00x                  |
| 2 Threads     | 170 ms          | **1.88x**              |

Le speedup est quasi-linéaire sur 2 threads, ce qui est cohérent avec les 2 coeurs physiques de la machine.

Je ne peux pas faire de graphique, ni d'analyse poussée sur le speedup car je n'ai pas assez de coeurs sur mon ordinateur

## 4. Séparation de l'affichage et du calcul (MPI)

### Implémentation

J'ai utilisé `mpi4py` pour diviser le travail entre deux processus distincts :

- **Processus 0 (Master) :** Gère l'affichage SDL2 et les événements. Il utilise `Iprobe` pour vérifier de manière asynchrone si de nouvelles données sont disponibles.
- **Processus 1 (Worker) :** Effectue les calculs de trajectoires en boucle et envoie les résultats au Master.

### Constats et Comparaisons

- L'affichage est désormais smooth. On peut zoomer et déplacer la caméra à 60 FPS sans aucune saccade, car le rendu n'attend plus que le calcul soit terminé pour rafraîchir l'image.
- Performance de calcul : On constate une baisse de performance sur le calcul pur par rapport à la version Numba seule.

Sur un processeur à seulement 2 coeurs physiques, le partitionnement MPI utilise déjà les deux coeurs (un pour le Master, un pour le Worker).

1. Le coeur dédié au Master (Affichage) est "réservé" pour maintenir la fluidité.
2. Le Worker se retrouve avec un seul coeur physique disponible.
3. Par conséquent, il devient impossible d'utiliser efficacement les 2 threads de Numba pour le calcul, car cela créerait une compétition de ressources sur les mêmes coeurs, générant des changements de contexte nuisibles à la performance.

La séparation MPI permet l'asynchronisme, donc un meilleur rendu mais elle sacrifie la puissance de calcul sur une machine disposant de peu de coeurs.
