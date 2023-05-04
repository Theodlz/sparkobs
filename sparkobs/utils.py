import os
import time
from urllib.parse import urlparse

import astropy.units as u
import healpy as hp
import ligo.skymap.bayestar as ligo_bayestar
import ligo.skymap.distance
import ligo.skymap.io
import ligo.skymap.moc
import ligo.skymap.postprocess
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table
from astropy_healpix import HEALPix, uniq_to_level_ipix
from mocpy import MOC
from numba import njit

LEVEL = MOC.MAX_ORDER
hp_29_area = hp.nside2pixarea(2**29)

def timeit(func):
    """Decorator to time a function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"\n{func.__name__} took {end - start} seconds")

        return result

    return wrapper

@njit
def is_overlap(a,b):
    """
    Check if 2 ranges overlap

    Parameters
    ----------
    a : tuple
        tuple of the form (upper, lower)
    b : tuple
        tuple of the form (upper, lower)

    Returns
    -------
    overlap : bool
        True if the 2 ranges overlap, False otherwise
    """

    return (a[0] <= b[1]) and (b[0] <= a[1])

@njit
def overlap_value(a,b):
    """
    Compute the "pixel area" (nb of pixels) of the overlap between 2 ranges

    Parameters
    ----------
    a : tuple
        tuple of the form (upper, lower)
    b : tuple
        tuple of the form (upper, lower)

    Returns
    -------
    overlap : int
        number of pixels in the overlap between a and b
    """

    if not is_overlap(a,b):
        return 0
    else:
        return (min(a[1], b[1]) - max(a[0], b[0]))
    
@njit
def compute_tiles_probdensity(tiles, ranges, probdensities, progress_proxy):
    """
    Compute the probability density of each tile by computing the overlap between the tile and the skymap

    Parameters
    ----------
    tiles : list of tuples
        list of tiles, each tile is a tuple of the form (field_id, upper, lower])
    ranges : list of tuples
        list of ranges, each range is a tuple of the form (upper, lower)
    probdensities : list of floats
        list of probdensities, each probdensity is a float

    Returns
    -------
    new_tiles : list of tuples
        list of tiles, each tile is a tuple of the form (field_id, upper, lower, probdensity)
    """
    new_tiles = []
    for i in range(len(tiles)):
        probs = [0]
        for j in range(len(ranges)):
            probs.append(probdensities[j] * overlap_value((tiles[i][1], tiles[i][2]), ranges[j]))
            progress_proxy.update(1)
        new_tiles.append((tiles[i][0], tiles[i][1], tiles[i][2], sum(probs) * hp_29_area))

    return new_tiles

def get_occulted(url, nside=64):
    """TO DOCUMENT, FOUND IN SKYPORTAL WITHOUT DOCSTRING"""
    m = Table.read(url, format='fits')
    ra = m.meta.get('GEO_RA', None)
    dec = m.meta.get('GEO_DEC', None)
    error = m.meta.get('GEO_RAD', 67.5)

    if (ra is None) or (dec is None) or (error is None):
        return None

    center = SkyCoord(ra * u.deg, dec * u.deg)
    radius = error * u.deg

    hpx = HEALPix(nside, 'ring', frame=ICRS())

    # Find all pixels in the circle.
    ipix = hpx.cone_search_skycoord(center, radius)

    return ipix

def uniq_to_range(uniq):
    """Convert a uniq to a healpix range (upper, lower)."""
    level, ipix = uniq_to_level_ipix(uniq)
    shift = 2 * (LEVEL - level)
    range = (ipix << shift, (ipix + 1) << shift)
    return range

@timeit
def skymap_from_url(url, level=0.9):
    """
    Load a skymap from a URL and return a dictionary with the following keys:
    - localization_name: name of the localization
    - ranges: list of healpix ranges, each range is a tuple of the form (upper, lower)
    - probdensities: list of probdensities, each probdensity is a float
    - moc: MOC object corresponding to the skymap

    Parameters
    ----------
    url : str
        URL of the skymap
    level : float
        credible region level, e.g. 0.9 for 90th credible region

    Returns
    -------
    skymap : dict
        dictionary containing the skymap
    """
    def get_col(m, name):
        try:
            col = m[name]
        except KeyError:
            return None
        else:
            return col.tolist()

    filename = os.path.basename(urlparse(url).path)

    skymap = ligo.skymap.io.read_sky_map(url, moc=True)

    nside = 128
    occulted = get_occulted(url, nside=nside)
    if occulted is not None:
        order = hp.nside2order(nside)
        skymap_flat = ligo_bayestar.rasterize(skymap, order)['PROB']
        skymap_flat = hp.reorder(skymap_flat, 'NESTED', 'RING')
        skymap_flat[occulted] = 0.0
        skymap_flat = skymap_flat / skymap_flat.sum()
        skymap_flat = hp.reorder(skymap_flat, 'RING', 'NESTED')
        skymap = ligo_bayestar.derasterize(Table([skymap_flat], names=['PROB']))

    uniq = np.asarray(get_col(skymap, 'UNIQ'))
    probdensities = np.asarray(get_col(skymap, 'PROBDENSITY'))

    # sort by probdensity in descending order
    idx = np.argsort(probdensities)[::-1]
    uniq = uniq[idx]
    probdensities = probdensities[idx]

    cum_prob = np.zeros(len(probdensities))
    for i in range(len(probdensities)):
        r = uniq_to_range(uniq[i])
        cum_prob[i] = probdensities[i] * (r[1] - r[0]) * hp_29_area

    cum_prob = np.cumsum(cum_prob)

    # find the index of the first element that is greater than level
    idx = np.where(cum_prob > level)[0][0]

    uniq = uniq[:idx]
    probdensities = probdensities[:idx]

    skymap = {
        'localization_name': filename,
        'ranges': np.asarray([uniq_to_range(u) for u in uniq]),
        'probdensities': probdensities,
    }
    skymap["moc"] = MOC.from_depth29_ranges(max_depth=29, ranges=skymap["ranges"])

    return skymap
