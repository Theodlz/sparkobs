import argparse
import os
from datetime import datetime, timedelta

import joblib
import toml
from astropy import time as ap_time

from sparkobs.telescope import Telescope
from sparkobs.utils import skymap_from_url

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--skymap', type=str, help='URL or Path of the skymap')
parser.add_argument('--level', type=float, default=0.95, help='credible level of the skymap')
parser.add_argument('--save_to', type=str, default=None, help='path to the plan file to save')
parser.add_argument('--telescope', type=str, default=None, help='path to the telescope configuration file')
parser.add_argument('--exposure_time', type=int, default=300, help='exposure time in seconds')
parser.add_argument('--filters', type=str, nargs='+', default=['g', 'r', 'g'], help='filters to use')
parser.add_argument('--start_date', type=str, default=None, help='start date of the observation')
parser.add_argument('--end_date', type=str, default=None, help='end date of the observation')
parser.add_argument('--min_time_interval', type=int, default=30, help='minimum time interval between observations in minutes')
parser.add_argument('--max_airmass', type=float, default=2.5, help='maximum airmass')
parser.add_argument('--min_moon_angle', type=float, default=10, help='minimum angle between the Moon and the field in degrees')
parser.add_argument('--min_galactic_latitude', type=float, default=10, help='minimum galactic latitude in degrees')
parser.add_argument('--use_primary', type=boolean_string, default=True, help='whether or not to observe the primary grid')
parser.add_argument('--use_secondary', type=bool, default=False, help='whether or not to observe the secondary grid')
parser.add_argument('--weights', type=float, nargs='+', default=[2, 1, 1], help='weights for the filters (probability, distance, airmass)')
args = parser.parse_args()

# load the telescope configuration
config = {}

# if both use_primary and use_secondary are False, raise an error
if not args.use_primary and not args.use_secondary:
    raise ValueError('Please set at least one of use_primary and use_secondary to True')

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
config['use_primary'] = args.use_primary
config['use_secondary'] = args.use_secondary

if args.end_date is not None and args.start_date is None:
    raise ValueError('Please provide a start date if you want to provide an end date')

if args.start_date is None:
    config['start_date'] = ap_time.Time(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f"), format='isot', scale='utc')
else:
    config['start_date'] = ap_time.Time(args.start_date, format='iso', scale='utc')

if args.end_date is None and args.start_date is None:
    config['end_date'] = ap_time.Time((datetime.utcnow()+ timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%f"), format='isot', scale='utc')
elif args.end_date is None and args.start_date is not None:
    config['end_date'] = config['start_date'] + timedelta(days=1)
else:
    config['end_date'] = ap_time.Time(args.end_date, format='iso', scale='utc')

if not isinstance(args.weights, list) or len(args.weights) != 3:
    raise ValueError('Please provide a list of exactly three weights: probability, distance, airmass')

# load the skymap
if args.skymap is None:
    raise ValueError('Please provide a valid path to the skymap')

skymap = skymap_from_url(args.skymap, args.level)

# create the telescope
telescope = Telescope(config)

# compute the observability
telescope.compute_observability(skymap)

# schedule the observations
telescope.schedule(weights=args.weights if args.weights is not None else [2, 1, 1])

# save the plan
if args.save_to is None:
    args.save_to = f'plans/{os.path.basename(args.skymap).split(".")[0]}.json'

telescope.save_plan(args.save_to)
