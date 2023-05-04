import joblib
from astropy import time as ap_time

from telescope import Telescope
from utils import skymap_from_url

# 1. Load skymap from FERMI GBM
skymap = skymap_from_url(
    url='https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2023/bn230430325/quicklook/glg_healpix_all_bn230430325.fit',
    level=0.9,
)

# 2. Load fields from file. Fields are precomputed for ZTF. TODO: add methods to compute fields for other telescopes
fields = joblib.load('data/fields_ztf.joblib')

# 3. Create config for telescope (here ZTF)
start_date = ap_time.Time('2023-04-30T01:47:19.219789', format='isot', scale='utc')
end_date = ap_time.Time('2023-05-01T01:47:19.219789', format='isot', scale='utc')

config = {
    'lon': -116.8361,
    'lat': 33.3634,
    'elevation': 1870.0,
    'diameter': 1.2,
    'fields': fields,
    'max_airmass': 2.5,
    'min_moon_angle': 30,
    'min_time_interval': 30,
    'start_date': start_date,
    'end_date': end_date,
    'filters': ['g', 'r', 'g'],
    'exposure_time' : 300,
    'primary_limit': 881,
}

# 4. Create telescope class
telescope = Telescope(config)

# 5. Compute observability
telescope.compute_observability(skymap)

# 6. Schedule observations
telescope.schedule()

# 7. Save plan
telescope.save_plan(f'plans/plan_{skymap["localization_name"].lower().replace(".fit", "")}.json')
