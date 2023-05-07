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

# (note that we didnt ask to use the secondary grid in any of these tests)

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
    assert len(telescope.observable_fields) == 50

    telescope.schedule()
    assert len(telescope.plan['planned_observations']) == 62

    stats_g = telescope.plan['stats']['g']
    assert stats_g['probability'] == 0.7948509291241069
    assert stats_g['area'] == 741.6241257405189
    assert stats_g['nb_fields'] == 23

    stats_r = telescope.plan['stats']['r']
    assert stats_r['probability'] == 0.761766616189799
    assert stats_r['area'] == 707.3737323361398
    assert stats_r['nb_fields'] == 22

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
    assert len(telescope.observable_fields) == 17

    telescope.schedule()
    assert len(telescope.plan['planned_observations']) == 34

    stats_g = telescope.plan['stats']['g']
    assert stats_g['probability'] == 0.9582597707790319
    assert stats_g['area'] == 367.7384777437539
    assert stats_g['nb_fields'] == 12

    stats_r = telescope.plan['stats']['r']
    assert stats_r['probability'] == 0.9537364841043192
    assert stats_r['area'] == 360.44055721937866
    assert stats_r['nb_fields'] == 17

    telescope.save_plan('plans/test.json')
    assert os.path.exists('plans/test.json')
    os.remove('plans/test.json')

if __name__ == '__main__':
    test_LVC()
    test_Fermi()
