import os

import toml
from astropy import time as ap_time

from sparkobs.telescope import Telescope
from sparkobs.utils import skymap_from_url

start_date = ap_time.Time('2023-04-30T01:47:19.219789', format='isot', scale='utc')
end_date = ap_time.Time('2023-05-01T01:47:19.219789', format='isot', scale='utc')

config = toml.load('config/ztf.toml')
config = {
    **config,
    'max_airmass': 2.5,
    'min_moon_angle': 30,
    'min_galactic_latitude': 10,
    'min_time_interval': 30,
    'filters': ['g', 'r', 'g'],
    'exposure_time' : 300,
} # config for ZTF common to all tests

def test_LVC():
    # a gw localization from GraceDB a little over 1000 sq deg
    url = 'https://gracedb.ligo.org/api/superevents/MS230502c/files/bayestar.fits.gz,1'
    skymap = skymap_from_url(url=url, level=0.95)

    gw_config = {
        **config,
        'start_date': ap_time.Time('2023-05-02T02:35:46.000000', format='isot', scale='utc'),
        'end_date': ap_time.Time('2023-05-03T02:35:46.000000', format='isot', scale='utc'),
    }

    telescope = Telescope(gw_config)
    telescope.compute_observability(skymap)
    assert len(telescope.observable_fields) == 101

    telescope.schedule()
    assert len(telescope.plan['planned_observations']) == 87

    telescope.save_plan('plans/test.json')
    assert os.path.exists('plans/test.json')
    os.remove('plans/test.json')

def test_Fermi():
    # a glg localization from Fermi GBM that's under 500 sq deg
    url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2023/bn230430325/quicklook/glg_healpix_all_bn230430325.fit"
    skymap = skymap_from_url(url=url, level=0.95)

    fermi_config = {
        **config,
        'start_date': ap_time.Time('2023-04-30T07:47:19.000000', format='isot', scale='utc'),
        'end_date': ap_time.Time('2023-05-01T07:47:19.000000', format='isot', scale='utc'),
    }

    telescope = Telescope(fermi_config)
    telescope.compute_observability(skymap)
    assert len(telescope.observable_fields) == 31

    telescope.schedule()
    assert len(telescope.plan['planned_observations']) == 45

    telescope.save_plan('plans/test.json')
    assert os.path.exists('plans/test.json')
    os.remove('plans/test.json')

if __name__ == '__main__':
    test_LVC()
    test_Fermi()
