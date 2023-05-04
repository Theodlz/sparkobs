import os

import joblib
import pytest
from astropy import time as ap_time

from sparkobs.telescope import Telescope
from sparkobs.utils import skymap_from_url

skymap_url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2023/bn230430325/quicklook/glg_healpix_all_bn230430325.fit'
level = 0.9

start_date = ap_time.Time('2023-04-30T01:47:19.219789', format='isot', scale='utc')
end_date = ap_time.Time('2023-05-01T01:47:19.219789', format='isot', scale='utc')

fields = joblib.load('data/fields_ztf.joblib')

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

skymap = skymap_from_url(skymap_url, level)

def test_create_telescope():
    try:
        Telescope(config)
    except Exception as e:
        pytest.fail(f'Failed to create telescope: {e}')

def test_compute_observability():
    telescope = Telescope(config)
    telescope.compute_observability(skymap)
    assert len(telescope.observable_fields) == 24

def test_schedule():
    telescope = Telescope(config)
    telescope.compute_observability(skymap)
    telescope.schedule()
    assert len(telescope.plan) == 41

def test_save_plan():
    telescope = Telescope(config)
    telescope.compute_observability(skymap)
    telescope.schedule()
    telescope.save_plan('plans/test.json')
    assert os.path.exists('plans/test.json')
    os.remove('plans/test.json')

if __name__ == '__main__':
    test_save_plan()
