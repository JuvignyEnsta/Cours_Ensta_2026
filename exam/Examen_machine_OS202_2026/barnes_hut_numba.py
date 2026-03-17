# Simulation d'un problème à N corps utilisant l'algorithme de Barnes-Hut :
#     - Construction d'un quadtree divisant récursivement sur le plan OXY en quatre quadrants (On ne découpe jamais selon Z car la galaxie est principalement plate)
#     - Calcul du centre de masse et de la masse totale pour chaque nœud de l'arbre
#     - Utilisation d'un critère θ (theta) pour décider si on approxime une région par son centre de masse
#     - Le calcul de l'accélération pour chaque corps se fait :
#          - Si s/d < θ (où s = taille du nœud, d = distance), on utilise le centre de masse du nœud
#          - Sinon, on descend récursivement dans les enfants du nœud
import numpy as np
import visualizer3d
import sys
from numba import njit, prange, deferred_type, optional,int64,float64
from numba.experimental import jitclass
from math import sqrt

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13
NW = 0
NE = 1
SW = 2
SE = 3

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

max_bodies_per_node : int64 = 16  # Nombre maximum de corps par nœud avant subdivision

@njit(fastmath=True)
def scatter_bodies(bodies_index : np.ndarray, positions : np.ndarray, center : np.ndarray, children_bodies : list[np.ndarray],
                   children_nbodies : list[int]):
    """
    Distribue les corps dans les quadrants en fonction de leur position par rapport au centre.

    :param bodies_index: Indices des corps à distribuer
    :type bodies_index: np.ndarray
    :param positions: Positions des corps [x,y,z]
    :type positions: np.ndarray
    :param center: Centre de la région (x, y, z)
    :type center: np.ndarray
    :param children_bodies: Listes des indices des corps pour chaque quadrant
    :type children_bodies: list[np.ndarray]
    :param children_nbodies: Nombres de corps dans chaque quadrant
    :type children_nbodies: list[int]
    """
    for i in range(bodies_index.shape[0]):
        body_idx = bodies_index[i]
        pos = positions[body_idx]
        if pos[0] < center[0]:
            if pos[1] < center[1]:
                quadrant = SW  # Quadrant inférieur gauche
            else:
                quadrant = NW  # Quadrant supérieur gauche
        else:
            if pos[1] < center[1]:
                quadrant = SE  # Quadrant inférieur droit
            else:
                quadrant = NE  # Quadrant supérieur droit
        children_bodies[quadrant][children_nbodies[quadrant]] = body_idx
        children_nbodies[quadrant] += 1

@njit(fastmath=True)
def compute_local_masses_com(bodies_index, positions, masses):
    mass = 0.
    com  = np.zeros(3,dtype=np.double)
    for i in bodies_index:
        mass += masses[i]
        com += masses[i]*positions[i]
    return (mass, com)

@njit(fastmath=True)
def local_compute_acceleration(position, positions : np.ndarray, masses : np.ndarray, indices : np.ndarray) -> np.ndarray :
    """
    Calcule l'accélération gravitationnelle exercée sur une position donnée par les corps spécifiés par leurs indices.
    :param position: Position du corps cible [x,y,z]
    :type position: np.ndarray
    :param indices: Indices des corps exerçant la force
    :type indices: np.ndarray
    :param positions: Positions des différents corps [x,y,z]
    :type positions: np.ndarray
    :param masses: Tableau des masses des corps
    :type masses: np.ndarray
    :return: Accélération gravitationnelle [ax, ay, az]
    :rtype: np.ndarray
    """
    acceleration = np.zeros(3)
    for i in range(indices.shape[0]):
        body_idx = indices[i]
        direction = positions[body_idx] - position
        distance = sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2])#linalg.norm(direction)  # Ajouter une petite valeur pour éviter la division par zéro
        if distance > 1.E-10 : 
            force_magnitude = G * masses[body_idx] / (distance*distance*distance)
            acceleration += force_magnitude * direction
    return acceleration

node_type = deferred_type()
@jitclass([('center', float64[:]),('com', float64[:]),('body', int64[:])])
class QuadtreeNode:
    center : np.ndarray
    size   : float
    size_z : float
    mass   : float
    nbodies: int
    body   : np.ndarray
    com    : np.ndarray
    nw_child : optional(node_type)
    ne_child : optional(node_type)
    sw_child : optional(node_type)
    se_child : optional(node_type)
    """
    Nœud d'un quadtree pour l'algorithme de Barnes-Hut en pseudo 2D.
    Chaque région  représente une région parallépiède-rectangle avec une base carrée en OXY
    """
    def __init__(self, center : np.ndarray, size : float, size_z : float):
        global max_bodies_per_node
        self.center = center
        self.size = size
        self.size_z = size_z
        self.mass = 0.0
        self.nbodies = 0
        self.body = np.empty(max_bodies_per_node, dtype=np.int64)
        self.com = np.zeros(3)
        self.nw_child = None  # Enfant nord-ouest
        self.ne_child = None  # Enfant nord-est
        self.sw_child = None  # Enfant sud-ouest
        self.se_child = None  # Enfant sud-est

    def subdivide(self):
        """
        Subdivise un nœud en quatre parties égales sur le plan OXY 
        """
        half_size = self.size * 0.5
        quarter_size = self.size * 0.25
        self.nw_child = QuadtreeNode(self.center + np.array([-quarter_size, quarter_size, 0]), half_size,self.size_z)
        self.ne_child = QuadtreeNode(self.center + np.array([quarter_size, quarter_size, 0]), half_size,self.size_z)
        self.sw_child = QuadtreeNode(self.center + np.array([-quarter_size, -quarter_size, 0]), half_size,self.size_z)
        self.se_child = QuadtreeNode(self.center + np.array([quarter_size, -quarter_size, 0]), half_size,self.size_z)
        
    def get_quadrant(self,position : np.ndarray):
        """    

        Détermine dans quel quadrant se trouve la position passée en argument.
        La composante z n'est pas prise en compte pour cette décision.

        :param position: Tableau [x,y[,z]]
        :type position: np.ndarray
        """
        if position[0] < self.center[0]:
            if position[1] < self.center[1]:
                return self.sw_child  # Quadrant inférieur gauche
            else:
                return self.nw_child  # Quadrant supérieur gauche
        else:
            if position[1] < self.center[1]:
                return self.se_child  # Quadrant inférieur droit
            else:
                return self.ne_child  # Quadrant supérieur droit
                        
    def set_nbodies(self, n : int64):
        self.nbodies = n
        
    def get_bodies_index(self):
        return self.body[:self.nbodies]
        
    def get_mass(self) -> float:
        return self.mass
    
    def set_mass(self, m : float):
        self.mass = m
        
    def get_com(self) -> np.ndarray:
        return self.com
    
    def set_com(self, com : np.ndarray):
        self.com = com
        
    def has_children(self) -> bool:
        return self.nw_child is not None
    
    def get_size(self) -> float:
        return self.size
    
    def get_child(self, i : int):
        if   i==NW: return self.nw_child
        elif i==NE: return self.ne_child
        elif i==SW: return self.sw_child
        elif i==SE: return self.se_child
        return None

node_type.define(QuadtreeNode.class_type.instance_type)

@njit(fastmath=True)
def insert_index_in_node(node : QuadtreeNode, body_index : int64, positions : np.ndarray):
    """
    Insère un corps dans le quadtree.


        :param body_index: Index du corps à insérer
        :type body_index: int64
        :param positions: Positions des différents corps [x,y,z]
        :type positions: np.ndarray
        :param masses: Tableau des masses des corps
        :type masses: np.ndarray
    """
    global max_bodies_per_node
    if node.nbodies < max_bodies_per_node and node.nw_child is None:
        # Nœud vide, insérer le corps ici
        node.body[node.nbodies] = body_index
        node.set_nbodies(node.nbodies + 1)
    else:
        if node.nw_child is None:
            # Nœud feuille avec un corps, subdiviser
            node.subdivide()
            assert(node.nw_child is not None)
            assert(node.ne_child is not None)
            assert(node.sw_child is not None)
            assert(node.se_child is not None)
            children_bodies = [node.nw_child.body, node.ne_child.body, node.sw_child.body, node.se_child.body]
            children_nbodies : list = [0,0,0,0]
            scatter_bodies(node.body, positions, node.center, children_bodies, children_nbodies)
            node.nw_child.set_nbodies(children_nbodies[0])
            node.ne_child.set_nbodies(children_nbodies[1])
            node.sw_child.set_nbodies(children_nbodies[2])
            node.se_child.set_nbodies(children_nbodies[3])
            quadrant : QuadtreeNode = node.get_quadrant(positions[body_index])
            insert_index_in_node(quadrant, body_index, positions)
            
            # Réinitialiser le nœud actuel
            node.set_nbodies(0)
        else:
            # Nœud interne, insérer dans l'enfant approprié
            quadrant : QuadtreeNode = node.get_quadrant(positions[body_index])
            insert_index_in_node(quadrant, body_index, positions)

@njit(fastmath=True)
def update_masses_com(node : QuadtreeNode, positions : np.ndarray, masses : np.ndarray ):
    if node.has_children():
        update_masses_com(node.get_child(NW), positions, masses)
        update_masses_com(node.get_child(NE), positions, masses)
        update_masses_com(node.get_child(SW), positions, masses)
        update_masses_com(node.get_child(SE), positions, masses)
        node.set_mass(node.get_child(NW).get_mass() + node.get_child(NE).get_mass() +
                     node.get_child(SW).get_mass() + node.get_child(SE).get_mass())
        node.set_com(node.get_child(NW).get_com() + node.get_child(NE).get_com() +
                     node.get_child(SW).get_com() + node.get_child(SE).get_com())
    else:
        m, c = compute_local_masses_com(node.get_bodies_index(), positions, masses)
        node.set_mass(m)
        node.set_com(c)
        
@njit(fastmath=True)
def finalize(node : QuadtreeNode):
        global G
        if node.has_children():
            finalize(node.get_child(NW))
            finalize(node.get_child(NE))
            finalize(node.get_child(SW))
            finalize(node.get_child(SE))
        if node.get_mass() > 0:
            node.set_com(node.get_com() / node.get_mass())
            node.set_mass(G * node.get_mass())

@njit(fastmath=True)
def compute_acceleration(node : QuadtreeNode, position : np.ndarray, positions : np.ndarray, masses : np.ndarray, theta : float = 0.5) -> np.ndarray :
        """
        Calcule l'accélération gravitationnelle exercée sur une position donnée par les corps dans ce nœud.

        :param position: Position du corps cible [x,y,z]
        :type position: np.ndarray
        :param theta: Paramètre de précision pour l'approximation Barnes-Hut
        :type theta: float
        :return: Accélération gravitationnelle [ax, ay, az]
        :rtype: np.ndarray
        """
        if node.get_mass()==0. : 
            return np.zeros(3)
        # Calculer la distance entre le corps cible et le centre de masse du nœud
        direction = node.get_com() - position
        distance = sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2])#linalg.norm(direction) 
        
        # Taille du nœud (longueur d'un côté de la base carrée)
        s : float = node.get_size()
        
        if s  < theta * distance:
            # Utiliser le centre de masse du nœud pour l'approximation
            force_magnitude = node.get_mass() / (distance*distance*distance)
            return force_magnitude * direction 
        else:
            if not node.has_children():
                if node.nbodies == 0:
                    return np.zeros(3)  # Nœud vide
                else:
                    # Nœud feuille, calculer la force directement
                    return local_compute_acceleration(position, positions, masses, node.get_bodies_index())
            else:
                # Descendre dans les enfants
                acceleration  = compute_acceleration(node.get_child(NW), position, positions, masses, theta)
                acceleration += compute_acceleration(node.get_child(NE), position, positions, masses, theta)
                acceleration += compute_acceleration(node.get_child(SW), position, positions, masses, theta)
                acceleration += compute_acceleration(node.get_child(SE), position, positions, masses, theta)
        
                return acceleration

@njit(fastmath=True)
def info_node(node : QuadtreeNode):
    print(f"noeud ayant {node.nbodies} corps")
    print("center : ", node.center, " masse : ", node.mass, " com : ", node.com)
    if node.nbodies > 0:
        print("indices des corps :")
        a = node.get_bodies_index()
        for v in a : print (v);
        print()
    if node.has_children():
        print(f"A des enfants -> ")
        info_node(node.get_child(NW))
        info_node(node.get_child(NE))
        info_node(node.get_child(SW))
        info_node(node.get_child(SE))
    else:
        print(f"et est une feuille")

@njit(fastmath=True)
def build_quadtree(box : np.ndarray, positions : np.ndarray) -> QuadtreeNode:
    """
    Construit le quadtree pour le système de N corps actuel.
    """
    center = np.array([(box[0][0] + box[1][0]) / 2., (box[0][1] + box[1][1]) / 2., (box[0][2] + box[1][2]) / 2.], dtype=np.float64)
    size = float(max(box[1][0] - box[0][0], box[1][1] - box[0][1]))
    size_z = float(box[1][2] - box[0][2])
    quadtree_root = QuadtreeNode(center, size, size_z)
    for i in range(positions.shape[0]):
        insert_index_in_node(quadtree_root, i, positions)
    return quadtree_root

class NBodySystem:
    def __init__(self, filename, theta=0.5):
        """
        Initialise le système de N corps.
        
        Parameters:
        -----------
        filename : str
            Nom du fichier contenant @njit
les données des corps
        theta : float
            Paramètre de Barnes-Hut (par défaut 0.5)
        """
        positions = []
        velocities = []
        masses = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6, -1.E-6, -1.E-6], [1.E-6, 1.E-6, 1.E-6]], dtype=np.float64)
        
        with open(filename, "r") as fich:
            line = fich.readline()
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i] - 1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i] + 1.E-6)
                
                line = fich.readline()
        
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.masses = np.array(masses, dtype=np.float64)
        self.colors = [generate_star_color(m) for m in masses]
        self.theta = theta
        
@njit(parallel=True, fastmath=True)
def compute_accelerations(quadtree_root : QuadtreeNode, positions : np.ndarray, masses : np.ndarray, theta : float) -> np.ndarray :
        """
        Calcule les accélérations pour tous les corps dans le système.
        
        Returns:
        --------
        accelerations : np.ndarray
            Tableau des accélérations [ax, ay, az] pour chaque corps
        """
        accel = np.empty_like(positions,dtype=np.float64)
        for i in prange(positions.shape[0]):
            accel[i] = compute_acceleration(quadtree_root, positions[i], positions, masses, theta)
        return accel

@njit(parallel=True, fastmath=True)
def update_positions_(dt : float, box : np.ndarray, positions : np.ndarray, velocities : np.ndarray, masses : np.ndarray, theta : float):
        """
        Met à jour les positions et vitesses des corps.
        
        Parameters:
        -----------
        dt : float
            Pas de temps
        """
        # Intégration de Verlet
        quadtree_root : QuadtreeNode = build_quadtree(box, positions)
        update_masses_com(quadtree_root, positions, masses)
        finalize(quadtree_root)
        accelerations = compute_accelerations(quadtree_root, positions, masses, theta)
        for i in prange(positions.shape[0]):
            positions[i,:] += velocities[i,:] * dt + 0.5 * accelerations[i,:] * dt * dt
        quadtree_root = build_quadtree(box, positions)
        update_masses_com(quadtree_root, positions, masses)
        finalize(quadtree_root)
        accelerations2 = compute_accelerations(quadtree_root, positions, masses, theta)
        for i in prange(velocities.shape[0]):
            velocities[i,:] += 0.5*(accelerations[i,:]+accelerations2[i,:]) * dt
        return (positions, velocities)

system : NBodySystem

def update_positions(dt : float):
    """
    Fonction callback pour la visualisation.
    """
    global system
    box = system.box
    positions = system.positions
    velocities = system.velocities
    masses = system.masses
    theta = system.theta
    positions, velocities = update_positions_(dt, box, positions, velocities, masses, theta)
    system.positions = positions
    system.velocities = velocities
    return positions

def run_simulation(filename, geometry=(800, 600), theta=0.5, dt=0.001):
    """
    Lance la simulation avec visualisation.
    
    Parameters:
    -----------
    filename : str
        Fichier de données
    geometry : tuple
        Taille de la fenêtre
    theta : float
        Paramètre de Barnes-Hut
    dt : float
        Pas de temps
    """
    import timeit
    global system
    system = NBodySystem(filename, theta=theta)
    
    # Initialiser la visualisation
    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(
        pos, col, intensity,
        [[system.box[0][0], system.box[1][0]], 
         [system.box[0][1], system.box[1][1]], 
         [system.box[0][2], system.box[1][2]]]
    )
    visu.run(updater=update_positions, dt=dt)

def run():
    filename = "data/test_data"
    dt = 0.001
    theta = 0.5
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 3:
        theta = float(sys.argv[3])
    
    print(f"Simulation Barnes-Hut de {filename} avec dt = {dt} et theta = {theta}")
    
    run_simulation(filename, theta=theta, dt=dt)

if __name__ == "__main__":
    run()
