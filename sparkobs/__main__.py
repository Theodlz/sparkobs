import argparse
import os

import joblib
from astropy import time as ap_time
import toml
from datetime import datetime, timedelta

from sparkobs.telescope import Telescope
from sparkobs.utils import skymap_from_url

parser = argparse.ArgumentParser()
parser.add_argument('--skymap_url', type=str, help='URL of the skymap')
parser.add_argument('--level', type=float, default=0.9, help='credible level of the skymap')
parser.add_argument('--save_to', type=str, default=None, help='path to the plan file to save')
parser.add_argument('--telescope', type=str, default=None, help='path to the telescope configuration file')
parser.add_argument('--exposure_time', type=int, default=300, help='exposure time in seconds')
parser.add_argument('--filters', type=str, nargs='+', default=['g', 'r', 'g'], help='filters to use')
parser.add_argument('--start_date', type=str, default=None, help='start date of the observation')
parser.add_argument('--end_date', type=str, default=None, help='end date of the observation')
parser.add_argument('--min_time_interval', type=int, default=30, help='minimum time interval between observations in minutes')
parser.add_argument('--max_airmass', type=float, default=2, help='maximum airmass')
parser.add_argument('--min_moon_angle', type=float, default=10, help='minimum angle between the Moon and the field in degrees')
parser.add_argument('--min_galactic_latitude', type=float, default=10, help='minimum galactic latitude in degrees')
args = parser.parse_args()

# load the telescope configuration
config = {}

if args.telescope is None or not os.path.exists(args.telescope):
    raise ValueError('Please provide a valid path to the telescope configuration file')

if args.telescope.endswith('.toml'):
    config = toml.load(args.telescope)
else:
    raise ValueError('Only TOML configuration files are supported at the moment')


config['exposure_time'] = args.exposure_time
config['filters'] = args.filters
config['max_airmass'] = args.max_airmass
config['min_moon_angle'] = args.min_moon_angle
config['min_galactic_latitude'] = args.min_galactic_latitude
config['min_time_interval'] = args.min_time_interval

if any([args.start_date is not None, args.end_date is not None]) and not all([args.start_date is not None, args.end_date is not None]):
    raise ValueError('Please provide both start_date and end_date, or neither')

if args.start_date is None:
    config['start_date'] = ap_time.Time(datetime.now(), format='datetime', scale='utc')
else:
    config['start_date'] = ap_time.Time(args.start_date, format='isot', scale='utc')

if args.end_date is None:
    config['end_date'] = ap_time.Time(datetime.now() + timedelta(days=1), format='datetime', scale='utc')
else:
    config['end_date'] = ap_time.Time(args.end_date, format='isot', scale='utc')

# load the skymap
if args.skymap_url is None:
    raise ValueError('Please provide a valid path to the skymap')

skymap = skymap_from_url(args.skymap_url, args.level)

# create the telescope
telescope = Telescope(config)

# compute the observability
telescope.compute_observability(skymap)

# schedule the observations
telescope.schedule()

# save the plan
if args.save_to is None:
    args.save_to = f'plans/{os.path.basename(args.skymap_url).split(".")[0]}.json'

telescope.save_plan(args.save_to)
