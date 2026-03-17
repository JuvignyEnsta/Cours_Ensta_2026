# Simulation d'un problème à N corps utilisant une approche grille pour accélerer le calcul :
#     - On crée une grille cartésienne régulière pour diviser l'espace en cellules
#     - On attribue chaque corps à une cellule de la grille
#     - On calcule le centre de masse et la masse totale de chaque cellule
#     - Le calcul de l'accélération pour chaque corps se fait :
#          - En sommant les corps de la même cellule et des cellules voisines
#          - En utilisant le centre de masse et la masse totale des cellules plus éloignées
import numpy as np
from numpy import linalg
import visualizer3d
import sys
import time
# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13

def generate_star_color(mass):
    """
    Génère une couleur pour une étoile en fonction de sa masse.
    Les étoiles massives sont bleues, les moyennes sont jaunes, les petites sont rouges.
    
    Parameters:
    -----------
    mass : float
        Masse de l'étoile en masses solaires
    
    Returns:
    --------
    color : tuple
        Couleur RGB (R, G, B) avec des valeurs entre 0 et 255
    """
    if mass > 5.0:
        # Étoiles massives: bleu-blanc
        return (150, 180, 255)
    elif mass > 2.0:
        # Étoiles moyennes-massives: blanc
        return (255, 255, 255)
    elif mass >= 1.0:
        # Étoiles comme le Soleil: jaune
        return (255, 255, 200)
    else:
        # Étoiles de faible masse: rouge-orange
        return (255, 150, 100)
    
class Grid:
    def __init__(self, box_min : np.ndarray, box_max : np.ndarray, n_cells_per_dir : np.ndarray[np.int32]):
        print (f"box min : {box_min}, box max : {box_max}")
        self.box_min = box_min
        self.box_max = box_max
        self.n_cells_per_dir = n_cells_per_dir
        self.cell_size = (box_max - box_min) / n_cells_per_dir
        # Pré-calculer les cellules voisines et lointaines pour chaque cellule
        self._precompute_cell_neighbors()

    def update_bounding_box(self, positions : np.ndarray):
        """
        Met à jour la boîte englobante du système en fonction des positions des corps.
        
        :param self: Description
        :param positions: Description
        :type positions: np.ndarray
        """
        for i in range(3):
            self.box_min[i] = min(self.box_min[i], np.min(positions[:,i]) - 1.E-6)
            self.box_max[i] = max(self.box_max[i], np.max(positions[:,i]) + 1.E-6)
        self.cell_size = (self.box_max - self.box_min) / self.n_cells_per_dir
        # Recalculer les voisins si nécessaire (structure de grille changée)
        self._precompute_cell_neighbors()

    def _precompute_cell_neighbors(self):
        """
        Pré-calcule pour chaque cellule possible ses cellules voisines (distance Chebyshev <= 1)
        et génère un mapping efficace pour les cellules lointaines.
        """
        self.neighbor_offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    self.neighbor_offsets.append((dx, dy, dz))
        
        # Pré-calculer toutes les cellules possibles comme array
        n = self.n_cells_per_dir
        self.all_cell_indices = np.array(
            [(i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])],
            dtype=np.int32
        )

    def update_indices_in_cells(self, positions : np.ndarray):
        """
        Met à jour les indices des corps dans chaque cellule de la grille.
        
        :param self: Description
        :param positions: Description
        :type positions: np.ndarray
        """
        # On le fait en vectorisé :
        ##On calculer l'indice de la cellule correspondant à chaque position des corps :
        indices = np.floor( (positions - self.box_min) / self.cell_size ).astype(int)
        # On s'assure que les indices sont dans les bornes :
        indices = np.clip(indices, 0, self.n_cells_per_dir - 1)
        # Puis on remplit le dictionnaire des contenus des cellules :
        self.cell_contents = {}
        for ibody, idx in enumerate(indices):
            key = (idx[0], idx[1], idx[2])
            if key not in self.cell_contents:
                self.cell_contents[key] = []
            self.cell_contents[key].append(ibody)

    def compute_global_mass_and_com(self, masses : np.ndarray, positions : np.ndarray):
        """
        Calcule la masse totale et le centre de masse de chaque cellule de la grille.
        
        :param self: Description
        :param masses: Description
        :type masses: np.ndarray
        :param positions: Description
        :type positions: np.ndarray
        """
        self.cell_mass = {}
        self.cell_com  = {}
        for key, body_indices in self.cell_contents.items():
            total_mass = np.sum(masses[body_indices])
            com = np.sum(positions[body_indices] * masses[body_indices][:,np.newaxis], axis=0) / total_mass
            self.cell_mass[key] = total_mass
            self.cell_com[key]  = com               
    
class NBodySystem:
    def __init__(self, filename, ncells_per_dir = 10):
        positions = []
        velocities = []
        masses    = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6,-1.E-6,-1.E-6],[1.E-6,1.E-6,1.E-6]], dtype=np.float64) # Contient les coins min et max du système
        with open(filename, "r") as fich:
            line = fich.readline() # Récupère la masse, la position et la vitesse sous forme de chaîne
            # Récupère les données numériques pour instancier un corps qu'on rajoute aux corps déjà présents :
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i]-1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i]+1.E-6)
                    
                line = fich.readline()
        
        self.positions  = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses     = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid   = Grid(self.box[0], self.box[1], ncells_per_dir)
        self.grid.update_indices_in_cells(self.positions)

    def compute_acceleration(self):
        """
        Calcul l'accélération de chaque corps en utilisant la méthode de la grille (version hautement optimisée).
        Si un corps est dans une cellule, on somme les contributions des corps dans la même cellule et les cellules voisines
        sinon on utilise le centre de masse et la masse totale des cellules plus éloignées.
        
        :param self: Description
        """
        n_bodies = self.positions.shape[0]
        accelerations = np.zeros((n_bodies, 3), dtype=np.float32)
        
        # Met à jour la grille :
        self.grid.update_bounding_box(self.positions)
        self.grid.update_indices_in_cells(self.positions)
        self.grid.compute_global_mass_and_com(self.masses, self.positions)
        
        # Calcul des indices de cellule pour tous les corps
        idx_cells = np.floor((self.positions - self.grid.box_min) / self.grid.cell_size).astype(int)
        idx_cells = np.clip(idx_cells, 0, self.grid.n_cells_per_dir - 1)
        # Convertir les clés de cellules en array numpy pour vectorisation
        cell_keys = np.array(list(self.grid.cell_contents.keys()))
        cell_coms = np.array([self.grid.cell_com[tuple(k)] for k in cell_keys], dtype=np.float32)
        cell_masses = np.array([self.grid.cell_mass[tuple(k)] for k in cell_keys], dtype=np.float32)
        # Pour chaque cellule occupée, traiter les interactions
        for key in self.grid.cell_contents:
            cell_bodies = np.array(self.grid.cell_contents[key], dtype=int)
            if len(cell_bodies) == 0:
                continue
                
            cell_idx = np.array(key)
            # 1. Interactions proches (cellules voisines) - traitement vectorisé par cellule
            for offset in self.grid.neighbor_offsets:
                neighbor_key = (cell_idx[0] + offset[0], cell_idx[1] + offset[1], cell_idx[2] + offset[2])
                # Vérifier que la cellule voisine est dans la grille
                if  (0 <= neighbor_key[0] < self.grid.n_cells_per_dir[0] and
                    0 <= neighbor_key[1] < self.grid.n_cells_per_dir[1] and
                    0 <= neighbor_key[2] < self.grid.n_cells_per_dir[2] and
                    neighbor_key in self.grid.cell_contents):                            
                            neighbor_bodies = np.array(self.grid.cell_contents[neighbor_key], dtype=int)
                            
                            # Calcul vectorisé pour toutes les paires (cell_bodies x neighbor_bodies)
                            for ibody in cell_bodies:
                                j_bodies = neighbor_bodies
                                diff = self.positions[j_bodies] - self.positions[ibody]
                                dists = linalg.norm(diff, axis=1)
                                valid = dists > 1.E-10  # Seuil ajusté pour les années-lumière
                                if np.any(valid):
                                    inv_dist3 = 1.0 / dists[valid]**3
                                    accelerations[ibody] += np.sum(
                                        (G * self.masses[j_bodies[valid], np.newaxis] * inv_dist3[:, np.newaxis]) * diff[valid],
                                        axis=0
                                    )
            
            # 2. Interactions lointaines (centre de masse) - traitement vectorisé
            # Calculer quelles cellules sont lointaines pour cette cellule (distance Chebyshev > 1)
            # Utiliser la distance de Chebyshev: max(|dx|, |dy|, |dz|)
            far_mask = ((np.abs(cell_keys[:, 0] - cell_idx[0]) > 1) |
                       (np.abs(cell_keys[:, 1] - cell_idx[1]) > 1) |
                       (np.abs(cell_keys[:, 2] - cell_idx[2]) > 1)) & \
                       ((np.abs(cell_keys[:, 0] - cell_idx[0]) + 
                         np.abs(cell_keys[:, 1] - cell_idx[1]) + 
                         np.abs(cell_keys[:, 2] - cell_idx[2])) > 1)
            
            if np.any(far_mask):
                far_coms = cell_coms[far_mask]
                far_masses = cell_masses[far_mask]
                
                # Pour chaque corps de la cellule, calculer l'accélération due aux cellules lointaines
                for ibody in cell_bodies:
                    diff = far_coms - self.positions[ibody]  # (n_far_cells, 3)
                    dists = linalg.norm(diff, axis=1)
                    inv_dist_cube = 1.0 / (dists**3)
                    accelerations[ibody] += np.sum(
                            (G * far_masses[:, np.newaxis] * inv_dist_cube[:, np.newaxis]) * diff,
                            axis=0
                        )
        return accelerations
    
    def update_positions(self, dt):
        accelerations = self.compute_acceleration()
        # Met à jour les vitesses et positions de tous les corps :
        self.positions  += self.velocities  * dt + 0.5 * accelerations * dt * dt
        self.velocities += accelerations * dt
        
system = None

def update_positions(dt):
    global system
    system.update_positions(dt)
    return system.positions

def run_simulation(filename, geometry=(800,600), ncells_per_dir=10, dt=0.001):
    # Initialise le système de corps :
    global system
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    # Initialise l'affichage graphique :
    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(pos, col, intensity,  [[system.box[0][0], system.box[1][0]], [system.box[0][1], system.box[1][1]], [system.box[0][2], system.box[1][2]]])
    visu.run(updater=update_positions, dt = dt)


filename = "data/test_data"
dt = 0.001
ncells_per_dir = np.array([10, 10, 1])
if len(sys.argv) > 1:
    filename = sys.argv[1]
if len(sys.argv) > 2:
    dt = float(sys.argv[2])
if len(sys.argv) > 3:
    ncells_per_dir = np.array([int(x) for x in sys.argv[3].split(',')])
print(f"Simulation de {filename} avec dt = {dt} et ncells_per_dir = {ncells_per_dir}")
run_simulation(filename, dt=dt, ncells_per_dir=ncells_per_dir)
