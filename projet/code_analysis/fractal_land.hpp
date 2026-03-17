#ifndef _FRACTAL_LAND_HPP_
# define _FRACTAL_LAND_HPP_
// Génération d'un fractal pour le coût énergie ( altitude ) de déplacement d'une fourmie
// L'algorithme prend plusieurs paramètres :
// 0. Taille : Nombre de "cases" par direction ( pair d'entier )
// 1. Nombre de graînes : Nombre de points initiaux par direction dont on détermine l'altitude à l'initialisation de l'agorithme
// 2. Déviation : degré de variation de l'altitude en fonction de la distance
// 3. Graîne aléatoire : détermine le paysage à retrouver
# include <vector>
# include <utility>

/**
 * @brief Génère un paysage fractal à l'aide d'un algorithme pseudo-aléatoire 
 * @details 
 *     Génère de façon récursive un paysage fractal à l'aide de algorithme pseudo-aléatoire :
 *        1. Créée une grille de taille \f$nbSeeds*2^{log\_size}+1\f$ cases par directions
 *        2. Génère une altitude pour les cases ayant des indices i et j multiples de 
 *           \f$2^{log\_size}\f$ de telle sorte que le gradient d'altitude entre deux points ne dépasse pas la valeur deviation
 *           On considère alors les sous-grilles ayant pour indices mimimals : \f$Ib*2^{log\_size}\f$, \f$Jb*2^{log\_size}\f$ avec
 *           Ib et Jb compris entre 0 et nbSeeds ( compris ) et de tailles \f$2^{log\_size}\f$.
 *           On note n=log_size le niveau des sous--grilles initiales.
 *        3. Pour chaque sous--grille, on génère l'altitude des points d'indices locaux multiples de \f$2^{n-1}\f$ ormi pour les
 *           coins de la grille en respectant le gradient de déviation.
 *        4. Puis on considère les sous--grilles de niveau n-1 auxquelles on reapplique l'algorithme à partir de 3 et on s'arrête dès que
 *           le niveau de la grille atteint zéro.
 * @param log_size Le logarithme base 2 de la dimension de chaque sous-grille initiale
 * @param nbSeeds  Le nombre de sous-grilles initiales par direction
 * @param deviation La valeur maximale du gradient entre deux altitudes.
 * @param seed Graîne de génération aléatoire
 * @return Un tableau contenant la carte des altitudes en fonctions des indices i et j.
 */
class fractal_land
{
public:
    using container=std::vector<double>;
    using dim_t=unsigned long;
    fractal_land( const dim_t& log_size, unsigned long nbSeeds, double deviation, int seed = 0 );
    fractal_land( const fractal_land& ) = delete;
    fractal_land( fractal_land&& land ) = default;
    ~fractal_land() = default;

    double operator () ( unsigned long i, unsigned long j ) const {
        return m_altitude[i+j*m_dimensions];
    }
    double& operator () ( unsigned long i, unsigned long j ) {
        return m_altitude[i+j*m_dimensions];
    }
    dim_t dimensions() const { return m_dimensions; }
    double* data() { return m_altitude.data(); }
    const double* data() const { return m_altitude.data(); }

private:
    void compute_subgrid( int log_subgrid_dim, int iB, int jB, double deviation, std::size_t seed );
    dim_t m_dimensions;
    container m_altitude;
};
#endif