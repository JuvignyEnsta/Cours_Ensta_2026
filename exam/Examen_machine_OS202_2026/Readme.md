# Parallélisation d'un code de simulation de galaxie

## Préambule

Assurez-vous que le module numba et la visualisation d'un nuage de points avec SDL 2 et OpengL fonctionnent (voir le TP n°5)

Pour mesurer l'accélération de votre code en parallélisant avec numba, vous pouvez jouer sur la variable ```NUMBA_NUM_THREADS``` :

```shell
NUMBA_NUM_THREADS=2 python3 mon_code_numba.py
```

permet d'utiliser deux threads pour la parallélisation. Par défaut, si la variable ```NUMBA_NUM_THREADS``` n'est pas définie, numba prendra un nombre de threads égal au nombre de cœurs logiques (et non physique).

Le format pour le document peut être en markdown, pdf, open format (libreoffice) ou word.

Vous conserverez une copie des scripts python de chacune des étapes décrites dans ce présent document.

Lorsque vous enverrez votre code et votre document, envoyez les non compressés et non archivés aux adresses e-mail suivant (selon le groupe) :

  - Groupe 1 : xavier.juvigny@onera.fr
  - Groupe 2 : augustin.parret-freaud@safrangroup.com
  - Groupe 3 : alexandre.suss@onera.fr

Vous pourrez quitter la salle d'examen uniquement lorsque votre chargé de TD vous confirmera d'avoir bien reçu votre travail.

## Description du programme séquentiel

Ce programme simule une galaxie contenant un trou noir central massif et $N$ étoiles gravitant autour de ce trou noir.

Une étoile $i$ est définie par sa position $\vec{p_{i}}$, sa vitesse $\vec{v_{i}}$ et sa masse $m_{i}$. On définit également sa couleur en fonction de sa masse pour l'affichage.

La masse $m_{i}$ restera constante tout le long de la simulation, et seules la position $\vec{p_{i}}$ et $\vec{v_{i}}$ sont mis à jour à chaque itération en temps à l'aide d'un schéma en temps de *Verlet* (avec un pas de temps $\delta t$ ) :

  - Pour chaque étoile $i$
      - Calculer l'accélération $\vec{a_{i}}^{(1)}$ subie par l'étoile (par le trou noir et les autres étoiles)
      - Mettre à jour la position de l'étoile : 
      $\vec{p_{i}} \leftarrow \vec{p_{i}} + \delta t.\vec{v_{i}} + \frac{1}{2}\delta t^{2} \vec{a}_{i}^{(1)}$
      - Calculer la nouvelle accélération $\vec{a}^{(2)}_{i}$ subie par l'étoile à sa nouvelle position
      - Mettre à jour sa vitesse : 
      $\vec{v_{i}} \leftarrow \vec{v_{i}} + \frac{1}{2}\delta t.\left(\vec{a_{i}}^{(1)}+\vec{a_{i}}^{(2)}\right)$

Quant au calcul de l'accélération subie par l'étoile, on utilise les lois de la gravitation universelle de Newton :

$$
\vec{a_{i}} = \sum_{j\neq i} \mathcal{G} \frac{m_{j} (\vec{p_{j}}(t)-\vec{p_{i}}(t))}{\lVert \vec{p_{j}}(t)-\vec{p_{i}}(t)\rVert^{3}}
$$

où $\mathcal{G} = 1.560339.10^{-13}$ est la constante de gravitation universelle exprimée pour les unités de mesure suivantes :
  - les distances sont exprimées en années-lumière (la distance que parcourt la lumière en un an)
  - la masse est exprimée en *masse-solaire* (c'est à dire que notre soleil dans cette unité est de masse égale à 1, et on exprime la masse des étoiles proportionnellement à la masse de notre soleil)
  - la durée est exprimée en année (terrestre).

Puisque chaque calcul d'accélération demande le calcul de $N-1$ distances, on remarque que l'algorithme "naïf" à une complexité en $N^{2}$.

Afin d'accélérer le calcul, on décide dans un premier temps d'utiliser une grille cartésienne $C$ de $N_{i}\times N_{j}$ cellules $C_{ij}$ partitionnant l'espace de calcul en boîtes parallèles au plan $Oxy$. A chaque pas de temps :

  -  On stocke chaque étoile dans la cellule $C_{ij}$ qui la contient (en fait son indice pour optimisation dans le tableau des vitesses et le tableau des positions des étoiles);
  - Pour chacune des cellules $C_{ij}$, on calcule la masse totale des étoiles (dont le trou noir) contenues dans $C_{ij}$ ainsi que son centre de masse;
  - Puis pour chacune des cellules $C_{ij}$, on calcul l'accélération subie par chaque étoile avec des étoiles "proches", c'est à dire des étoiles se trouvant soit dans la même cellule soit dans une cellule voisine (diagonale comprise);
  - On complète le calcul de l'accélération en utilisant la masse totale et le centre de masse des cellules $C_{kl}$ qui ne sont pas voisines de la cellule $C_{ij}$.

Vous pouvez trouver le code séquentiel dans ```nbodies_grid_numba.py```.  

Dans le répertoire ```data``` vous trouverez deux jeux de données, une pour une galaxie contenant 1000 étoiles et une seconde pour une galaxie contenant 5000 étoiles. 

Vous pouvez générer un nouveau jeu de données à l'aide du script ```galaxy_generator.py```. Usage :

```shell
python3 galaxy_generator.py <nombre étoiles> <nom fichier de sortie>
```

Exemple :
```shell
python3 galaxy_generator.py 10000 data/galaxy_10000
```

Pour utiliser un jeu de donnée spécifique pour la simulation, vous pouvez passer le fichier à lire en argument. De façon général, ```nbodies_grid_numba.py``` attend en option :
   - le jeu de donnée à utiliser (par défaut, c'est ```data/galaxy_1000```)
   - le pas de temps à utiliser (défaut $\delta t=0.001$)
   - Le nombre de cellule contenue dans la grille cartésienne : $N_{i}$ $N_{j}$ et $N_{k}$ 

__Exemple d'utilisation__ :

```shell
python3 nbodies_grid_numba.py data/galaxy_5000 0.0015 15 15 1
```

lancera la simulation sur les données pour 5000 étoiles, avec un pas de temps de $\delta t=0.0015$, et une grille cartésienne avec 15 cellules en $N_{i}$, 15 cellules en $N_{j}$ et une cellule en $N_{k}$.

**Question préliminaire** : En observant la forme des galaxies, pourquoi il n'est pas intéressant de prendre une valeur pour $N_{k}$ autre que 1 (sachant que $N_{k}$ donne le nombre de cellules en $Oz$) ?

## Mesure du temps initial

Observez les temps pris pour l'affichage et le calcul. Quel est la partie de l'algorithme (le calcul des trajectoires ou l'affichage) qui est la plus intéressante à paralléliser ?

## Parallélisation en numba du code séquentiel

Dans un premier temps, rajouter l'option ```parallel=True```aux décorateurs ```njit``` et remplacer aux endroits adéquats l'instruction ```range``` par ```prange``` afin de paralléliser les boucles qui vous semble parallélisable.

Calculer l'accélération du code en fonction du nombre de threads.

## Séparation de l'affichage et du calcul

En reprenant le code que vous venez de transformer, séparez en MPI l'affichage du calcul des trajectoires des étoiles. Le processus 0 s'occupera de l'affichage et le processus 1 s'occupera du calcul.

Comparez l'accélération obtenue en fonction du nombre de threads utilisés pour le calcul avec la version précédente. Que constatez-vous ? Pourquoi ?

## Parallélisation du calcul

L'idée générale pour paralléliser à l'aide de MPI le calcul des trajectoires est d'utiliser la technique des cellules fantômes sur la grille cartésienne permettant l'optimisation du calcul des trajectoires :

   - Mettre à jour les étoiles qui sont stockées par processus à chaque pas de temps (en terme de position, de vitesse et de masse), c'est à dire les étoiles qui appartiendront à une cellule locale du processus (cellule fantôme comprise);
   - Distribuer ces étoiles sur les cellules contenues par le processus
   - Mettre à jour les tableaux contenant les masses totales et les centres de masse de chaque cellule de la grille cartésienne globale.

Travail à effectuer :

  - Mettre en œuvre la parallélisation des trajectoires avec MPI à l'aide du code obtenue dans la partie précédente;
  - Calculer l'accélération obtenue en fonction du nombre de processus **ET** de threads numba;
  - En observant que la densité d'étoiles diminue avec l'éloignement du trou noir, quel problème de performance de notre algorithme parallèle peut handicaper l'accélération ?
  - Proposer une distribution "intelligente" des cellules de notre grille cartésienne. Quel problème de performance peut alors apparaître ? (on ne demande pas de mettre en œuvre cette distribution).

## Pour aller plus loin

Le préconditionnement du calcul par une grille cartésienne n'est pas l'algorithme le plus optimal. Un algorithme très performant pour traiter un grand nombre d'étoile est l'algorithmede  Barnes-Hut, qui consiste à construire à chaque pas de temps un *quadtree* :

  - On calcul la boîte englobant l'ensemble des étoiles
  - On subdivise dans le plan $Oxy$ la boîte en quatre sous-boîtes de même dimension;
  - Puis pour chacune de ces sous-boîtes, on subdivise de nouveau en quatre de façon récursive jusqu'à ce que chaque partie crée ne contient plus qu'un nombre d'étoiles inférieur à un certain nombre $n_{s}$. 

Le calcul de l'accélération se fait pour chaque étoile de la manière suivante (la première boîte considérée est la boîte englobant le domaine de calcul):

  - Si l'étoile est à une distance de la boîte supérieure à la moitié de la dimension de la boîte, on calcule la contribution de cette boîte à l'accélération de l'étoile à l'aide de sa masse totale et de son centre de masse
  - Sinon 
     - si cette boîte contient des sous-boîtes, on calcule la contribution de chacune de ces sous-boîtes à l'accélération de l'étoile (appel récursif);
     -   Sinon si la boîte ne contient pas de sous-boîte, on calcule la contribution de chacune des étoiles contenues par cette sous-boîte à l'accélération de notre étoile.

On peut montrer que l'algorithme pour calculer l'accélération de toutes les étoiles est alors en $N\log_{2}(N)$ (au lieu de $N^{2}$).

Le code mettant en œuvre cet algorithme peut être trouvé dans ```barnes_hut_numba.py```. Il n'est donné qu'à titre indicatif et n'est pas destiné à être modifié.

- Comment distribuer les différentes boîtes et sous-boîtes parmi les processus (on s'autorise que certains processus partagent les mêmes boîtes) ?
- Proposer sur papier une façon de paralléliser à l'aide de MPI l'accélération des étoiles en utilisant une telle structure et la distribution que vous avez proposé.

On peut supposer que le nombre de processus utilisés est une puissance de quatre et que le nombre de processus est très petit devant le nombre d'étoiles...



