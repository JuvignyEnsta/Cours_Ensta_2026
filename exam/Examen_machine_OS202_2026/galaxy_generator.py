"""
Module pour générer une galaxie avec des étoiles en orbite stable autour d'un trou noir central.
"""
import numpy as np
import random

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
# G_SI = 6.67430e-11 m^3 kg^-1 s^-2
# 1 ly = 9.4607e15 m
# 1 M_sun = 1.98847e30 kg
# 1 an = 3.15576e7 s
# G = G_SI * (ly^3 / M_sun) / an^2
G = 1.560339e-13  # ly^3 / (M_sun * an^2)


def generate_stable_orbit(black_hole_mass, star_mass, min_radius=0.001, max_radius=1.0):
    """
    Génère une position et vitesse pour une étoile en orbite elliptique stable.
    
    Parameters:
    -----------
    black_hole_mass : float
        Masse du trou noir central (en masses solaires)
    star_mass : float
        Masse de l'étoile (en masses solaires)
    min_radius : float
        Rayon minimal de l'orbite (en années-lumière)
    max_radius : float
        Rayon maximal de l'orbite (en années-lumière)
    
    Returns:
    --------
    position : np.array
        Position 3D de l'étoile (x, y, z) en années-lumière
    velocity : np.array
        Vitesse 3D de l'étoile (vx, vy, vz) en années-lumière par an
    """
    # Masse en masses solaires (pas de conversion nécessaire)
    M_bh = black_hole_mass
    
    # Génération d'une position aléatoire dans un disque galactique
    # Distance au centre (semi-grand axe de l'orbite elliptique)
    radius = random.uniform(min_radius, max_radius)
    
    # Angle dans le plan du disque galactique
    theta = random.uniform(0, 2 * np.pi)
    
    # Inclinaison par rapport au plan (faible pour simuler un disque)
    inclination = random.gauss(0, 0.1)  # Faible inclinaison
    
    # Position dans le plan (x, y)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = radius * np.sin(inclination)
    
    position = np.array([x, y, z])
    
    # Calcul de la vitesse orbitale pour une orbite circulaire
    # v = sqrt(G * M / r) en ly/an
    r = np.linalg.norm(position)
    v_orbital = np.sqrt(G * M_bh / r)
    
    # Excentricité de l'orbite (0 = circulaire, proche de 1 = très elliptique)
    eccentricity = random.uniform(0.0, 0.7)
    
    # Ajustement de la vitesse en fonction de l'excentricité
    # Pour une orbite elliptique, la vitesse varie selon la position
    v_magnitude = v_orbital * np.sqrt(1 + eccentricity)
    
    # Direction de la vitesse (perpendiculaire au rayon dans le plan orbital)
    # Vecteur unitaire dans la direction tangentielle
    tangent = np.array([-y, x, 0])
    if np.linalg.norm(tangent) > 0:
        tangent = tangent / np.linalg.norm(tangent)
    
    # Ajout d'une composante verticale aléatoire faible
    z_component = random.gauss(0, 0.05) * v_magnitude
    
    velocity = v_magnitude * tangent
    velocity[2] += z_component
    
    return position, velocity


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
    elif mass > 1.0:
        # Étoiles comme le Soleil: jaune
        return (255, 255, 200)
    else:
        # Étoiles de faible masse: rouge-orange
        return (255, 150, 100)


def generate_galaxy(n_stars, 
                   black_hole_mass=None,
                   star_mass_range=(0.5, 10.0),
                   min_orbital_radius=0.001,
                   max_orbital_radius=1.0,
                   output_file=None):
    """
    Génère une galaxie avec n étoiles en orbite autour d'un trou noir central.
    
    Parameters:
    -----------
    n_stars : int
        Nombre d'étoiles à générer
    black_hole_mass : float, optional
        Masse du trou noir central en masses solaires (si None, générée aléatoirement)
    star_mass_range : tuple
        Plage de masses pour les étoiles (min, max) en masses solaires
    min_orbital_radius : float
        Rayon orbital minimum (en années-lumière)
    max_orbital_radius : float
        Rayon orbital maximum (en années-lumière)
    output_file : str, optional
        Nom du fichier de sortie (si None, ne sauvegarde pas)
    
    Returns:
    --------
    masses : list
        Liste des masses (trou noir + étoiles)
    positions : list
        Liste des positions
    velocities : list
        Liste des vitesses
    colors : list
        Liste des couleurs RGB
    """
    # Génération de la masse du trou noir si non spécifiée
    if black_hole_mass is None:
        black_hole_mass = random.uniform(1e5, 1e10)
    
    # Listes pour stocker les données
    masses = []
    positions = []
    velocities = []
    colors = []
    
    # Ajout du trou noir central (position et vitesse nulles)
    masses.append(black_hole_mass)
    positions.append([0.0, 0.0, 0.0])
    velocities.append([0.0, 0.0, 0.0])
    colors.append((0, 0, 0))  # Noir pour le trou noir
    
    # Génération des étoiles
    for i in range(n_stars):
        # Masse aléatoire de l'étoile
        star_mass = random.uniform(star_mass_range[0], star_mass_range[1])
        
        # Génération de l'orbite stable
        pos, vel = generate_stable_orbit(black_hole_mass, star_mass, 
                                         min_orbital_radius, max_orbital_radius)
        
        # Couleur de l'étoile
        color = generate_star_color(star_mass)
        
        # Ajout aux listes
        masses.append(star_mass)
        positions.append(pos.tolist())
        velocities.append(vel.tolist())
        colors.append(color)
    
    # Sauvegarde dans un fichier si spécifié
    if output_file is not None:
        with open(output_file, 'w') as f:
            for i in range(len(masses)):
                # Format: masse px py pz vx vy vz
                f.write(f"{masses[i]:.6e} ")
                f.write(f"{positions[i][0]:.6e} {positions[i][1]:.6e} {positions[i][2]:.6e} ")
                f.write(f"{velocities[i][0]:.6e} {velocities[i][1]:.6e} {velocities[i][2]:.6e}\n")
        
        print(f"Galaxie générée avec {n_stars} étoiles et sauvegardée dans '{output_file}'")
        print(f"Masse du trou noir central: {black_hole_mass:.2e} masses solaires")
    
    return masses, positions, velocities, colors


def main():
    """
    Fonction principale pour tester le générateur de galaxie.
    """
    import sys
    
    # Paramètres par défaut
    n_stars = 100
    output_file = "data/galaxy_100"
    
    # Lecture des arguments de ligne de commande
    if len(sys.argv) > 1:
        n_stars = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Génération de la galaxie
    masses, positions, velocities, colors = generate_galaxy(
        n_stars=n_stars,
        output_file=output_file
    )
    
    print(f"\nStatistiques de la galaxie:")
    print(f"  - Nombre total d'objets: {len(masses)}")
    print(f"  - Nombre d'étoiles: {n_stars}")
    print(f"  - Masse totale: {sum(masses):.2e} masses solaires")
    print(f"  - Masse moyenne des étoiles: {np.mean(masses[1:]):.2f} masses solaires")
    print(f"  - Distance min/max: {min(np.linalg.norm(p) for p in positions[1:]):.4f} / {max(np.linalg.norm(p) for p in positions[1:]):.4f} années-lumière")


if __name__ == "__main__":
    main()
