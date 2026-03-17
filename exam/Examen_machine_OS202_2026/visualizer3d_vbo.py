"""
Module de visualisation 3D de points lumineux
=============================================

Ce module permet de visualiser des points en 3D avec leur couleur et luminosité
en utilisant SDL2 pour la gestion de la fenêtre et OpenGL pour le rendu 3D.

Fonctionnalités :
- Affichage de points 3D avec couleur et luminosité
- Rotation de la caméra avec la souris
- Zoom avec la molette de la souris
- Mise à jour dynamique de la position des points

Utilisation :
    visualizer = Visualizer3D(points, colors, luminosities, bounds)
    visualizer.run()
    
    # Pour mettre à jour les points :
    visualizer.update_points(new_points, new_colors, new_luminosities)
"""

import numpy as np
import sdl2
import sdl2.ext
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes


class Visualizer3D:
    """
    Classe principale pour la visualisation 3D de points lumineux.
    
    Attributes:
        points (np.ndarray): Tableau (N, 3) des coordonnées xyz des points
        colors (np.ndarray): Tableau (N, 3) des couleurs RGB (valeurs 0-1)
        luminosities (np.ndarray): Tableau (N,) des luminosités (valeurs 0-1)
        bounds (tuple): ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    """
    
    def __init__(self, points, colors, luminosities, bounds):
        """
        Initialise le visualiseur 3D.
        
        Args:
            points (np.ndarray): Coordonnées des points, shape (N, 3)
            colors (np.ndarray): Couleurs RGB des points, shape (N, 3), valeurs entre 0 et 1
            luminosities (np.ndarray): Luminosités des points, shape (N,), valeurs entre 0 et 1
            bounds (tuple): Limites de l'espace ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        """
        # Stockage des données des points
        self.points = np.array(points, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        self.luminosities = np.array(luminosities, dtype=np.float32)
        self.bounds = bounds
        
        # Paramètres de la fenêtre
        self.window_width = 1024
        self.window_height = 768
        self.window = None
        self.gl_context = None
        
        # Paramètres de la caméra
        self.camera_distance = 5.0  # Distance de la caméra au centre
        self.camera_rotation_x = 0.0  # Rotation autour de l'axe X (vertical)
        self.camera_rotation_y = 0.0  # Rotation autour de l'axe Y (horizontal)
        self.zoom_factor = 1.0  # Facteur de zoom
        
        # Paramètres de contrôle de la souris
        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.3  # Sensibilité de la rotation
        
        # Flag pour contrôler la boucle principale
        self.running = False
        
        # Vertex Buffer Objects pour optimisation GPU
        self.vbo_vertices = None
        self.vbo_colors = None
        self.vbo_needs_update = True
        
        # Calcul du centre de la scène (pour centrer la visualisation)
        self.center = np.array([
            (bounds[0][0] + bounds[0][1]) / 2.0,
            (bounds[1][0] + bounds[1][1]) / 2.0,
            (bounds[2][0] + bounds[2][1]) / 2.0
        ], dtype=np.float32)
        
        # Calcul de l'échelle de la scène (pour normaliser l'affichage)
        self.scale = max(
            bounds[0][1] - bounds[0][0],
            bounds[1][1] - bounds[1][0],
            bounds[2][1] - bounds[2][0]
        )
        
        # Initialisation de SDL et OpenGL
        self._init_sdl()
        self._init_opengl()
        self._init_vbo()
    
    def _init_sdl(self):
        """
        Initialise SDL2 et crée la fenêtre avec contexte OpenGL.
        """
        # Initialisation de SDL
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError(f"Erreur SDL_Init: {sdl2.SDL_GetError()}")
        
        # Configuration des attributs OpenGL avant création de la fenêtre
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DEPTH_SIZE, 24)
        
        # Création de la fenêtre avec support OpenGL
        self.window = sdl2.SDL_CreateWindow(
            b"Visualisation 3D - Points Lumineux",
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            self.window_width,
            self.window_height,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN
        )
        
        if not self.window:
            raise RuntimeError(f"Erreur création fenêtre: {sdl2.SDL_GetError()}")
        
        # Création du contexte OpenGL
        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        if not self.gl_context:
            raise RuntimeError(f"Erreur création contexte GL: {sdl2.SDL_GetError()}")
        
        # Activation de la synchronisation verticale (VSync)
        sdl2.SDL_GL_SetSwapInterval(1)
    
    def _init_opengl(self):
        """
        Configure OpenGL pour le rendu 3D des points lumineux.
        """
        # Couleur de fond (noir)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Activation du test de profondeur pour l'affichage 3D correct
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Activation du blending pour les effets de luminosité
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Blending additif pour l'effet lumineux
        
        # Activation de l'antialiasing pour les points
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        # Configuration de la taille des points
        glPointSize(3.0)
        
        # Configuration de la matrice de projection (perspective)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect_ratio = self.window_width / self.window_height
        gluPerspective(45.0, aspect_ratio, 0.1, 100.0)
        
        # Retour à la matrice modelview pour les transformations de la scène
        glMatrixMode(GL_MODELVIEW)
    
    def _init_vbo(self):
        """
        Initialise les Vertex Buffer Objects (VBO) pour un rendu GPU optimisé.
        Les VBO permettent de stocker les données directement dans la mémoire GPU.
        """
        # Génération des buffers
        self.vbo_vertices = glGenBuffers(1)
        self.vbo_colors = glGenBuffers(1)
        
        # Initialisation des buffers avec les données
        self._update_vbo()
    
    def _update_vbo(self):
        """
        Met à jour les données dans les VBO (vertices et couleurs).
        """
        # Calcul des couleurs avec luminosité (vectorisé)
        colors_with_luminosity = (self.colors * self.luminosities[:, np.newaxis] / 255.0).astype(np.float32)
        
        # Upload des vertices dans le VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_DYNAMIC_DRAW)
        
        # Upload des couleurs dans le VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, colors_with_luminosity.nbytes, colors_with_luminosity, GL_DYNAMIC_DRAW)
        
        # Unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.vbo_needs_update = False
    
    def _setup_camera(self):
        """
        Configure la position et l'orientation de la caméra.
        Applique le zoom et les rotations définies par l'utilisateur.
        """
        glLoadIdentity()
        
        # Recul de la caméra avec zoom
        distance = self.camera_distance / self.zoom_factor
        glTranslatef(0.0, 0.0, -distance)
        
        # Application des rotations de la caméra
        glRotatef(self.camera_rotation_x, 1.0, 0.0, 0.0)  # Rotation autour de X
        glRotatef(self.camera_rotation_y, 0.0, 1.0, 0.0)  # Rotation autour de Y
        
        # Centrage de la scène
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])
    
    def _render(self):
        """
        Effectue le rendu de la scène 3D.
        Dessine tous les points avec leur couleur et luminosité.
        """
        # Effacement des buffers de couleur et de profondeur
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Configuration de la caméra
        self._setup_camera()
        
        # Mise à jour des VBO si nécessaire
        if self.vbo_needs_update:
            self._update_vbo()
        
        # Dessin des points avec VBO (rendu GPU optimisé)
        # Activation des vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        # Binding des VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glColorPointer(3, GL_FLOAT, 0, None)
        
        # Rendu en une seule opération GPU
        glDrawArrays(GL_POINTS, 0, len(self.points))
        
        # Désactivation des états
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        # Échange des buffers (double buffering)
        sdl2.SDL_GL_SwapWindow(self.window)
    
    def _handle_events(self):
        """
        Gère les événements SDL (souris, clavier, fermeture de fenêtre).
        
        Returns:
            bool: False si l'utilisateur veut quitter, True sinon
        """
        event = sdl2.SDL_Event()
        
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            # Événement de fermeture de fenêtre
            if event.type == sdl2.SDL_QUIT:
                return False
            
            # Événement clavier
            elif event.type == sdl2.SDL_KEYDOWN:
                # Touche ESC pour quitter
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    return False
            
            # Événement bouton de souris enfoncé
            elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    self.mouse_dragging = True
                    self.last_mouse_x = event.button.x
                    self.last_mouse_y = event.button.y
            
            # Événement bouton de souris relâché
            elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    self.mouse_dragging = False
            
            # Événement mouvement de souris
            elif event.type == sdl2.SDL_MOUSEMOTION:
                if self.mouse_dragging:
                    # Calcul du déplacement de la souris
                    dx = event.motion.x - self.last_mouse_x
                    dy = event.motion.y - self.last_mouse_y
                    
                    # Mise à jour des rotations de la caméra
                    self.camera_rotation_y += dx * self.mouse_sensitivity
                    self.camera_rotation_x += dy * self.mouse_sensitivity
                    
                    # Limitation de la rotation verticale pour éviter le retournement
                    self.camera_rotation_x = np.clip(self.camera_rotation_x, -89.0, 89.0)
                    
                    # Mise à jour de la dernière position de la souris
                    self.last_mouse_x = event.motion.x
                    self.last_mouse_y = event.motion.y
            
            # Événement molette de souris (zoom)
            elif event.type == sdl2.SDL_MOUSEWHEEL:
                # Ajustement du facteur de zoom
                if event.wheel.y > 0:  # Molette vers le haut = zoom avant
                    self.zoom_factor *= 1.1
                elif event.wheel.y < 0:  # Molette vers le bas = zoom arrière
                    self.zoom_factor /= 1.1
                
                # Limitation du zoom pour éviter les valeurs extrêmes
                self.zoom_factor = np.clip(self.zoom_factor, 0.1, 10.0)
        
        return True
    
    def update_points(self, points, colors=None, luminosities=None):
        """
        Met à jour les positions, couleurs et/ou luminosités des points.
        
        Cette méthode permet de modifier les points affichés sans recréer
        le visualiseur, utile pour les animations ou simulations.
        
        Args:
            points (np.ndarray): Nouvelles coordonnées des points, shape (N, 3)
            colors (np.ndarray, optional): Nouvelles couleurs, shape (N, 3)
            luminosities (np.ndarray, optional): Nouvelles luminosités, shape (N,)
        """
        self.points = np.array(points, dtype=np.float32)
        
        if colors is not None:
            self.colors = np.array(colors, dtype=np.float32)
        
        if luminosities is not None:
            self.luminosities = np.array(luminosities, dtype=np.float32)
        
        # Marquer les VBO pour mise à jour au prochain rendu
        self.vbo_needs_update = True
    
    def run(self, updater=None, dt = 0.001 ):
        """
        Lance la boucle principale de visualisation.
        
        Cette méthode bloque jusqu'à ce que l'utilisateur ferme la fenêtre
        ou appuie sur ESC.
        """
        self.running = True
        
        print("Contrôles :")
        print("  - Clic gauche + déplacement souris : rotation de la caméra")
        print("  - Molette de la souris : zoom")
        print("  - ESC ou fermeture de fenêtre : quitter")

        t1 = sdl2.SDL_GetTicks()
        #dt = 0.01  # Intervalle de temps fictif pour la mise à jour        
        # Boucle principale
        while self.running:
            # Gestion des événements
            self.running = self._handle_events()
            
            # Rendu de la scène
            self._render()

            
            # Mise à jour via la fonction updater si fournie
            t3 = sdl2.SDL_GetTicks()
            if updater is not None:
                self.update_points(updater(dt))
            # Petite pause pour ne pas surcharger le CPU
            #sdl2.SDL_Delay(10)
            t2 = sdl2.SDL_GetTicks()
            print(f"Render time: {t3 - t1} ms, Update time: {t2 - t3} ms", end='\r')
            t1 = t2
        # Nettoyage
        self.cleanup()
    
    def cleanup(self):
        """
        Libère les ressources SDL et OpenGL.
        """
        # Suppression des VBO
        if self.vbo_vertices is not None:
            glDeleteBuffers(1, [self.vbo_vertices])
        if self.vbo_colors is not None:
            glDeleteBuffers(1, [self.vbo_colors])
        
        if self.gl_context:
            sdl2.SDL_GL_DeleteContext(self.gl_context)
        
        if self.window:
            sdl2.SDL_DestroyWindow(self.window)
        
        sdl2.SDL_Quit()


def demo():
    """
    Fonction de démonstration du module.
    Crée une sphère de points colorés avec différentes luminosités.
    """
    # Nombre de points
    n_points = 1000
    
    # Génération de points aléatoires dans une sphère
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.5, 2.0, n_points)
    
    points = np.zeros((n_points, 3), dtype=np.float32)
    points[:, 0] = r * np.sin(phi) * np.cos(theta)
    points[:, 1] = r * np.sin(phi) * np.sin(theta)
    points[:, 2] = r * np.cos(phi)
    
    # Génération de couleurs aléatoires
    colors = np.random.uniform(2, 255, (n_points, 3)).astype(np.float32)
    
    # Génération de luminosités aléatoires
    luminosities = np.random.uniform(0.3, 1.0, n_points).astype(np.float32)
    
    # Définition des limites de l'espace
    bounds = ((-3, 3), (-3, 3), (-3, 3))
    
    # Création et lancement du visualiseur
    visualizer = Visualizer3D(points, colors, luminosities, bounds)
    visualizer.run()


if __name__ == "__main__":
    # Lancement de la démo
    demo()
