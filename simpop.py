from __future__ import print_function, division

from isochrones.dartmouth import Dartmouth_Isochrone
dar = Dartmouth_Isochrone()

from scipy.stats import poisson, powerlaw
import pandas as pd
import numpy as np


#Get useful functions from Dan's code:
from utils import get_duration, get_a, get_delta
from utils import get_mes, get_pdet, get_pwin
from utils import get_pgeom, get_completeness

R_EARTH = 0.009171 #in solar units

P_RANGE = (50, 300)
R_RANGE = (0.75, 20)

def draw_powerlaw(alpha, rng):
    """
    Returns random variate according to x^alpha, between rng[0] and rng[1]
    """
    if alpha == -1:
        alpha = -1.0000001
    # Normalization factor
    x0, x1 = rng
    C = (alpha + 1) / (x1**(alpha + 1) - x0**(alpha + 1))
    
    u = np.random.random()
    return ((u * (alpha + 1)) / C + x0**(alpha + 1))**(1./(alpha + 1))

def draw_planet(theta):
    """
    Returns radius and period for a planet, given parameters
    """
    _, alpha, beta, _, _ = theta
    return draw_powerlaw(alpha, R_RANGE), draw_powerlaw(beta, P_RANGE)
    
def get_companion(theta, star, ic=dar, band='Kepler'):
    """
    Returns companion star and flux ratio F2/F1, given primary star and parameters

    theta : lnf0, alpha, beta, fB, gamma.  fB = binary fraction, gamma = mass-ratio
            distribution power law index

    star : row from stellar table; has .mass, .radius, .feh
    ic : isochrone
    
    Age of system is drawn randomly; occasionally a star will 
    be attempted to be simulated that is out of range of the isochrone;
    this will raise a ValueError.  

    Also, the radius of the star remains *as in the stellar catalog*.  The
    radius of the secondary is simulated.  If the secondary radius ends
    up larger than the primary, it is fixed to the same as the primary.
    """
    _, _, _, fB, gamma = theta

    # Is there a binary?  If not, just return radius
    if np.random.random() > fB:
        return star, 0.

    # Draw the mass of the secondary
    M1 = star.mass
    qmin = dar.minmass / M1
    q = draw_powerlaw(gamma, (qmin, 1))
    M2 = q*M1
    
    # Now we need more precise stellar properties
    minage, maxage = ic.agerange(M1, star.feh)
    maxage = min(maxage, ic.maxage)
    minage = max(minage, ic.minage)
    minage += 0.05
    maxage -= 0.05
    if maxage < minage:
        raise ValueError('Cannot simulate this: maxage < minage!')
    age = np.random.random() * (maxage - minage) + minage

    R1 = star.radius # This is WRONG
    R2 = ic.radius(M2, age, star.feh)
    debug_str = '({}, {}, {}, {}, agerange: ({}, {}))'.format(M1, M2, age, star.feh,
                                                             minage, maxage)
    if np.isnan(R2):
        raise ValueError('R2 is NaN:',debug_str)
    if R2 > R1:
        R2 = R1  #HACK alert!
    
    dmag = ic.mag[band](M2, age, star.feh) - ic.mag[band](M1, age, star.feh)
    flux_ratio = 10**(-0.4 * dmag)
    
    if np.isnan(flux_ratio):
        logging.warning('Flux_ratio is nan: {}'.format(debug_str))
        
    newstar = star.copy()
    newstar.mass = M2
    newstar.radius = R2
    
    return newstar, flux_ratio
    

def diluted_radius(radius, star, star2, flux_ratio):
    """
    Returns diluted radius.
    
    radius : true planet radius
    star : primary star
    star2 : secondary star
    flux_ratio : flux ratio F2/F1

    Coin flip whether planet should be around primary or secondary
    star; the *observed* radius (including dilution and "wrong-star-ness")
    """
        
    if flux_ratio==0:
        return radius, star
    
    # calculate radius correction factor; 
    #   depends on which star planet is around

    if np.random.random() < 0.5:
        # around star 1
        Xr = np.sqrt(1 + flux_ratio)
        host_star = star.copy()
    else:
        # around star 2
        Xr = star.radius / star2.radius * np.sqrt((1 + flux_ratio) / flux_ratio)
        host_star = star2.copy()
        
    return radius / Xr, host_star

    
def generate_planets(theta, stars=stlr, mes_threshold=10):
    """
    theta = (lnf0, alpha, beta, fB, gamma)
    """
    lnf0, alpha, beta, fB, gamma = theta
    
    planets = pd.DataFrame({'kepid':[], 'koi_prad':[], 'koi_period':[],
                           'koi_prad_true':[], 'koi_max_mult_ev':[]})

    n_skipped = 0
    
    for _, star in stars.iterrows():
        if np.isnan(star.radius) or np.isnan(star.mass):
            n_skipped += 1
            continue
            
        n_planets = poisson(np.exp(lnf0)).rvs()
        if n_planets == 0:
            continue
            
        try:
            star2, flux_ratio = get_companion(theta, star)
        except ValueError:
            n_skipped += 1
            continue
            #logging.warning('Skipping {}; cannot simulate binary.'.format(star.kepid))
        
        for i in range(n_planets):
            # First, figure out true & observed properties of planet
            radius, period = draw_planet(theta) 
            observed_radius, host_star = diluted_radius(radius, star, star2, flux_ratio)
            
            logging.debug('True: {:.2f}, Observed: {:.2f} ({})'.format(radius, 
                                                               observed_radius,
                                                              flux_ratio))
            
            # Then, is it detected?
            # First, geometric:
            aor = get_a(period, host_star.mass)
            if np.isnan(aor):
                raise RuntimeError('aor is nan: P={} M={}'.format(period, host_star.mass))
            #print(host_star.mass, aor)
            transit_prob = get_pgeom(aor / host_star.radius, 0.) # no ecc.
            
            if np.random.random() > transit_prob:
                continue
            
            # Then depth and MES:
            depth = get_delta(observed_radius * R_EARTH / star.radius)
            tau = get_duration(period, aor, 0.) * 24 # no ecc.
            try:
                mes = get_mes(star, period, depth, tau)
            except ValueError:
                n_skipped += 1
                #raise RuntimeError('MES is nan! {}, {}, {}'.format(depth, tau))
                
            
            if mes < mes_threshold:
                continue
            
            # Add planet to catalog
            planets = planets.append({'kepid':star.kepid,
                               'koi_prad':observed_radius,
                               'koi_period':period,
                               'koi_prad_true':radius,
                                'koi_max_mult_ev':mes}, ignore_index=True)
        
    print('{} planets generated ({} of {} stars skipped.)'.format(len(planets),
                                                                 n_skipped, len(stars)))
    return planets


if __name__=='__main__':

    import pandas as pd
    import os, sys, shutil
    from utils import stlr

    folder = sys.argv[1]
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    #Write indices of q1_q17_dr24_stellar table to folder:
    stlr.index.to_hdf(os.path.join(folder, 'inds.h5'), 'inds')
    
    #generate population with no binaries
    theta = [-0.3, -1.5, -0.8, 0.0, 0.3]
    df = generate_planets(theta, stlr)
    filename = os.path.join(folder, 'synthetic_kois_single.h5')
    df.to_hdf(filename, 'kois')
    pd.Series(theta).to_hdf(filename, 'theta')
    
    #generate population with binaries
    theta = [-0.3, -1.5, -0.8, 0.5, 0.3]
    df = generate_planets(theta, stlr)
    filename = os.path.join(folder, 'synthetic_kois_binaries.h5')
    df.to_hdf(filename, 'df')
    pd.Series(theta).to_hdf(filename, 'theta')

    
    
