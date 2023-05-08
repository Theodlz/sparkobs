import copy
import json
import math
import os
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta

import astroplan
import astropy.units as u
import joblib
import numpy as np
import tqdm
from astropy import time as ap_time
from astropy.coordinates import EarthLocation, SkyCoord, get_moon
from mocpy import MOC
from numba.typed import List as NumbaList
from numba_progress import ProgressBar

from sparkobs.utils import angle, compute_tiles_probdensity, timeit


class Telescope:
    """
    Class representing a telescope

    Attributes
    ----------
    lon : float
        Longitude of the telescope
    lat : float
        Latitude of the telescope
    elevation : float
        Elevation of the telescope
    diameter : float
        Diameter of the telescope
    fields : dict
        Dictionary of fields
    max_airmass : float
        Maximum airmass for the fields to be observable
    min_moon_angle : float
        Minimum angle between the moon and the fields (in degrees) to be observable
    min_galactic_latitude : float
        Minimum galactic latitude (in degrees) for the fields to be observable
    min_time_interval : float
        Minimum time interval (in minutes) between observations of different fields
    start_date : `astropy.time.Time`
        Start date of the observations (UTC), format: 'YYYY-MM-DDTHH:MM:SS.SSSSSS'
    end_date : `astropy.time.Time`
        End date of the observations (UTC), format: 'YYYY-MM-DDTHH:MM:SS.SSSSSS'
    filters : list
        List of filters to use for the observations
    exposure_time : float
        Exposure time for the observations (in seconds)
    primary_limit : float
        Last field_id of the primary field (exclusive)
    """

    def __init__(self, config):
        self.lon = config['lon']
        self.lat = config['lat']
        self.elevation = config['elevation']
        self.diameter = config['diameter']
        self.location = EarthLocation.from_geodetic(self.lon * u.deg, self.lat * u.deg, self.elevation * u.m)
        self.fixed_location = True
        self.fields = config['fields']
        if isinstance(self.fields, str):
            self.fields = joblib.load(self.fields)
        self.max_airmass = config['max_airmass']
        self.min_moon_angle = config['min_moon_angle']
        self.min_galactic_latitude = config['min_galactic_latitude']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.min_time_interval = config['min_time_interval'] * 60
        self.use_primary = config.get('use_primary', True)
        self.use_secondary = config.get('use_secondary', False)
        if not self.use_primary and not self.use_secondary:
            raise ValueError("At least one of use_primary or use_secondary must be True")
        self.filters = config["filters"]
        self.exposure_time = config["exposure_time"]
        self.primary_limit = config['primary_limit']
        self.moon_dt = None
        self.dec_range = config.get('dec_range', None)

        self.adjust_dates()
        self.compute_deltaT()
        self.setup_grid()

        self.targets_cache = {}


    @property
    def observer(self):
        """Return an `astroplan.Observer` representing an observer at this
        facility, accounting for the latitude, longitude, and elevation."""
        try:
            return self._observer
        except AttributeError:
            if (
                self.lon is None
                or self.lon == ""
                or np.isnan(self.lon)
                or self.lat is None
                or self.lat == ""
                or np.isnan(self.lat)
                or self.fixed_location is False
                or self.fixed_location is None
            ):
                self._observer = None
                return self._observer

        try:
            elevation = self.elevation
            # if elevation is not specified, assume it is 0
            if (
                self.elevation is None
                or self.elevation == ""
                or np.isnan(self.elevation)
            ):
                elevation = 0

            self._observer = astroplan.Observer(
                longitude=self.lon * u.deg,
                latitude=self.lat * u.deg,
                elevation=elevation * u.m,
            )

        except Exception as e:
            print(
                f'Telescope {self.id} ("{self.name}") cannot calculate an observer: {e}'
            )
            self._observer = None

        return self._observer

    def next_twilight_evening_astronomical(self, time=None):
        """The astropy timestamp of the next evening astronomical (-18 degree)
        twilight at this site. If time=None, uses the current time."""
        observer = self.observer
        if observer is None:
            return None
        if time is None:
            time = ap_time.Time.now()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = observer.twilight_evening_astronomical(time, which='next')
            if isinstance(t.value, np.ma.core.MaskedArray):
                return None
        return t

    def next_twilight_morning_astronomical(self, time=None):
        """The astropy timestamp of the next morning astronomical (-18 degree)
        twilight at this site. If time=None, uses the current time."""
        observer = self.observer
        if observer is None:
            return None
        if time is None:
            time = ap_time.Time.now()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = observer.twilight_morning_astronomical(time, which='next')
            if isinstance(t.value, np.ma.core.MaskedArray):
                return None
        return t

    def target(self, field_ids: list):
        """Return an `astroplan.FixedTarget` representing the target of this
        observation."""
        if isinstance(field_ids, int):
            field_ids = [field_ids]
        ra = np.array([self.fields[field_id]['ra'] for field_id in field_ids])
        dec = np.array([self.fields[field_id]['dec'] for field_id in field_ids])
        return astroplan.FixedTarget(SkyCoord(ra=ra * u.deg, dec=dec * u.deg), name=str(field_ids))

    def airmass(self, field_ids: int, time: np.ndarray, below_horizon=np.inf):
        """Return the airmass of the field at a given time. Uses the Pickering
        (2002) interpolation of the Rayleigh (molecular atmosphere) airmass.

        The Pickering interpolation tends toward 38.7494 as the altitude
        approaches zero.

        Parameters
        ----------
        time : `astropy.time.Time` or list of astropy.time.Time`
            The time or times at which to calculate the airmass
        below_horizon : scalar, Numeric
            Airmass value to assign when an object is below the horizon.
            An object is "below the horizon" when its altitude is less than
            zero degrees.

        Returns
        -------
        airmass : ndarray
           The airmass of the Obj at the requested times
        """

        # the output shape should be targets x times
        output_shape = (len(field_ids), len(time))
        time = np.atleast_1d(time)
        target = self.target(field_ids)
        altitude = self.altitude(time, target).to('degree').value
        above = altitude > 0

        # use Pickering (2002) interpolation to calculate the airmass
        # The Pickering interpolation tends toward 38.7494 as the altitude
        # approaches zero.
        sinarg = np.zeros_like(altitude)
        airmass = np.ones_like(altitude) * np.inf
        sinarg[above] = altitude[above] + 244 / (165 + 47 * altitude[above] ** 1.1)
        airmass[above] = 1.0 / np.sin(np.deg2rad(sinarg[above]))

        # set objects below the horizon to an airmass of infinity
        airmass[~above] = below_horizon
        airmass = airmass.reshape(output_shape)

        return airmass

    def altitude(self, time: np.ndarray, target: astroplan.FixedTarget):
        """Return the altitude of the object at a given time.

        Parameters
        ----------
        time : `astropy.time.Time`
            The time or times at which to calculate the altitude

        Returns
        -------
        alt : `astropy.coordinates.AltAz`
           The altitude of the Obj at the requested times
        """
        time = np.atleast_1d(time)

        return self.observer.altaz(time, target, grid_times_targets=True).alt

    def moon_angle(self, field_ids, time):
        """Return the angle between the field and the moon at a given time.

        Parameters
        ----------
        field_id : str
            The field id
        time : `astropy.time.Time` or list of astropy.time.Time`
            The time or times at which to calculate the angle

        Returns
        -------
        angle : ndarray
           The angle(s) between the field and the moon at the requested times
        """
        output_shape = (len(field_ids), len(time))
        time = ap_time.Time(time)
        target = self.target(field_ids)
        angles = []

        constraint = astroplan.MoonSeparationConstraint(self.min_moon_angle * u.deg)
        moon = constraint(self.observer, target.coord, time, grid_times_targets=True)
        moon = moon.reshape(output_shape)
        return moon
    
    def galactic_latitude(self, field_ids):
        """Return the angle between the field and the galactic plane at a given time.

        Parameters
        ----------
        field_id : str
            The field id
        time : `astropy.time.Time` or list of astropy.time.Time`
            The time or times at which to calculate the angle

        Returns
        -------
        angle : ndarray
           The angle(s) between the field and the galactic plane at the requested times
        """
        output_shape = (len(field_ids),)
        target = self.target(field_ids)
        angles = []

        angles = target.coord.galactic.b.deg
        angles = np.asarray(angles).reshape(output_shape)
        return angles

    def adjust_dates(self):
        """Adjust the start and end dates to the next evening and morning
        astronomical twilight at this site.

        Parameters
        ----------
        start_date : `astropy.time.Time`
            The start date
        end_date : `astropy.time.Time`
            The end date

        Returns
        -------
        adjusted_start_date : `astropy.time.Time`
            The adjusted start date
        adjusted_end_date : `astropy.time.Time`
            The adjusted end date
        """
        next_evening_ast = self.next_twilight_evening_astronomical(self.start_date)
        adjusted_start_date = self.start_date if self.start_date > next_evening_ast else next_evening_ast

        next_morning_ast = self.next_twilight_morning_astronomical(adjusted_start_date)
        adjusted_end_date = self.end_date if self.end_date < next_morning_ast else next_morning_ast

        self.start_date = adjusted_start_date
        self.end_date = adjusted_end_date

        print()
        print('-' * 80)
        print(f'Adjusted start date: {self.start_date.isot}')
        print(f'Adjusted end date: {self.end_date.isot}')
        print('-' * 80)
        print()

    def set_dates(self, start_date, end_date):
        """Set the start and end dates.

        Parameters
        ----------
        start_date : `astropy.time.Time`
            The start date
        end_date : `astropy.time.Time`
            The end date
        """
        self.start_date = start_date
        self.end_date = end_date
        self.adjust_dates()

    def compute_deltaT(self):
        """Compute the deltaT array, which is the time interval between each observation of different fields."""
        deltaT = np.arange(0, (self.end_date - self.start_date).sec, self.exposure_time)
        self.deltaT = np.asarray([self.start_date + timedelta(seconds=delta) for delta in deltaT])
        self.deltaTjd = np.asarray([t.jd for t in self.deltaT])

    def setup_grid(self):
        if self.use_secondary == False and self.primary_limit:
            print('Using primary fields only (as secondary fields are disabled).')
            fields = {}
            for field_id in self.fields.keys():
                if field_id < self.primary_limit:
                    fields[field_id] = self.fields[field_id]
        elif self.use_primary == False and self.primary_limit:
            print('Using secondary fields only (as primary fields are disabled).')
            fields = {}
            for field_id in self.fields.keys():
                if field_id >= self.primary_limit:
                    fields[field_id] = self.fields[field_id]
        else:
            print('Using both primary and secondary fields.')
            fields = self.fields

        self.fields = fields
                

    def compute_fields_airmasses(self):
        """Compute the airmass of each field at each deltaT."""
        # we compute the airmass at the start and end of one exposure to see if the field is observable at least once at each deltaT
        # change dts to be (len(fields), dts)
        #dts = np.tile(dts, (len(self.fields), 1))
        airmasses = self.airmass(self.observable_fields.keys(), self.deltaT)
        for i, field_id in tqdm.tqdm(enumerate(self.observable_fields.keys()), desc='Computing airmasses'):
            self.observable_fields[field_id]['airmasses'] = airmasses[i]

    def compute_fields_moon_angles(self):
        """Compute the moon angle of each field at each deltaT."""
        moon_angles = self.moon_angle(self.observable_fields.keys(), self.deltaT)
        for i, field_id in tqdm.tqdm(enumerate(self.observable_fields.keys()), desc='Computing moon angles'):
            self.observable_fields[field_id]['moon_angles'] = moon_angles[i]

    def update_observable_fields(self, method):
        """Select fields that are observable at least once at each deltaT, i.e. fields with airmasses < max_airmass at least once."""
        observable_fields = {}
        if method == 'airmasses':
            for field_id in tqdm.tqdm(self.observable_fields.keys(), desc=f'Selecting observable fields (max airmass)'):
                field = self.observable_fields[field_id]
                if np.any(field['airmasses'] < self.max_airmass):
                    observable_fields[field_id] = field
        elif method == 'moon_angles':
            for field_id in tqdm.tqdm(self.observable_fields.keys(), desc='Selecting observable fields (min moon angles)'):
                field = self.observable_fields[field_id]
                if np.any(field['moon_angles']):
                    observable_fields[field_id] = field
        elif method == 'dec_range':
            if self.dec_range is None:
                self.observable_fields = self.fields if self.observable_fields in [None, {}] else self.observable_fields
                return
            for field_id in tqdm.tqdm(self.observable_fields.keys(), desc='Selecting observable fields (dec range)'):
                field = self.observable_fields[field_id]
                if field['dec'] >= self.dec_range[0] and field['dec'] <= self.dec_range[1]:
                    observable_fields[field_id] = field
        elif method == 'galactic_plane':
            galactic_latitudes = self.galactic_latitude(self.observable_fields.keys())
            for i, field_id in tqdm.tqdm(enumerate(self.observable_fields.keys()), desc='Selecting observable fields (galactic plane)'):
                field = self.observable_fields[field_id]
                if np.any(np.abs(galactic_latitudes[i]) >= self.min_galactic_latitude):
                    observable_fields[field_id] = field

        self.observable_fields = observable_fields

    def drop_tiles_from_observable_fields(self):
        """Drop the tiles from the observable fields once they are not useful anymore."""
        for field_id in self.observable_fields.keys():
            self.observable_fields[field_id].pop('tiles', None)

    def add_tiles_to_observable_fields(self):
        """Add the tiles to the observable fields once they are needed."""
        for field_id in self.observable_fields.keys():
            self.observable_fields[field_id]['tiles'] = self.fields[field_id]['tiles']

    @timeit
    def compute_observability(self, skymap):
        """Compute the observability of each field at each deltaT + exposure_time (i.e. the end of at least one observation)
        and remove fields that are never observable at any deltaT + exposure_time, and without any overlap with the skymap."""
        self.observable_fields = copy.deepcopy(self.fields)

        print()
        self.update_observable_fields('dec_range')

        print()
        self.update_observable_fields('galactic_plane')

        print()
        self.compute_fields_skymap_overlap(skymap)

        print()
        self.compute_fields_airmasses()

        print()
        self.update_observable_fields('airmasses')

        print()
        self.adjust_deltaT()

        #print()
        self.compute_fields_moon_angles()

        print()
        self.update_observable_fields('moon_angles')

        print()
        self.compute_field_probdensity(skymap)

        # we drop the fields until we are done with computing the observability to save memory
        self.drop_tiles_from_observable_fields()

        # add a score to each field set to the probdensity of the field
        for field_id in self.observable_fields.keys():
            self.observable_fields[field_id]['score'] = self.observable_fields[field_id]['probdensity']
            self.observable_fields[field_id]['last_observed'] = None
            self.observable_fields[field_id]['filter_ids'] = []
    
    def overlapping_field_tiles(self, field: dict, moc: MOC):
        """Return the overlapping field tiles between a field and a skymap's moc."""
        tiles = [MOC.from_depth29_ranges(29, ranges=np.expand_dims(tile, axis=0)) for tile in field['tiles']]
        field_tiles= np.array(tiles)[[not(tile & moc) == MOC.new_empty(max_depth=29) for tile in tiles]]
        return field_tiles
    
    def compute_fields_skymap_overlap(self, skymap, fields=None):
        """Update the observable fields to only keep the fields that overlap with the skymap."""
        no_fields_provided = fields is None
        if no_fields_provided is True:
            fields = self.observable_fields
        observable_fields = {}
        for field_id in tqdm.tqdm(fields.keys(), desc='Computing skymap observability'):
            field = fields[field_id]
            field_tiles = self.overlapping_field_tiles(field, skymap["moc"])
            if len(field_tiles) > 0:
                observable_fields[field_id] = field
        if no_fields_provided is True:
            self.observable_fields = observable_fields

        return observable_fields

    def compute_field_probdensity(self, skymap, fields=None):
        """Compute the probdensity of each observable field."""
        ranges = skymap["ranges"]
        probdensities = skymap["probdensities"]
        no_fields_provided = fields is None
        if no_fields_provided is True:
            fields = self.observable_fields

        # reformat the observable fields to a list of tiles that look like (field_id, tile[0], tile[1], probdensity) to use with numba
        tiles = NumbaList()
        for field_id in fields.keys():
            for tile in fields[field_id]["tiles"]:
                tiles.append((field_id, tile[0], tile[1]))

        new_tiles = None
        with ProgressBar(total=int(len(tiles)), desc='Computing fields probdensity') as progress:
            new_tiles = compute_tiles_probdensity(tiles, ranges, probdensities, progress)

        # reformat the tiles back to a dictionary of fields
        new_fields = {}
        for tile in new_tiles:
            field_id = tile[0]
            if field_id not in new_fields:
                new_fields[field_id] = {**self.fields[field_id], "tiles": [], "probdensity": 0, "pixel_area": 0}
            new_fields[field_id]["tiles"].append((tile[1], tile[2]))
            new_fields[field_id]["probdensity"] += tile[3]
            new_fields[field_id]["pixel_area"] += tile[4]

        if no_fields_provided is True:
            self.observable_fields = new_fields

        return new_fields
    
    def adjust_deltaT(self):
        # we look at the first idx of each field airmasses array that is below max_airmass
        # we do this for each field to determine the first time at least one field is observable
        min_idx = len(self.deltaT)
        max_idx = 0
        for i, field_id in enumerate(self.observable_fields.keys()):
            field = self.observable_fields[field_id]
            indexes = np.argwhere(field['airmasses'] < self.max_airmass)
            min_idx = indexes[0][0] if indexes[0] < min_idx else min_idx
            max_idx = indexes[-1][0] if indexes[-1] > max_idx else max_idx

        # we adjust the deltaT array to start at the first time at least one field is observable and end at the last time at least one field is observable
        self.deltaT = self.deltaT[min_idx:max_idx]
        self.deltaTjd = self.deltaTjd[min_idx:max_idx]

        # we also adjust the airmasses and moon_angles arrays of each field
        for field_id in self.observable_fields.keys():
            field = self.observable_fields[field_id]
            field['airmasses'] = field['airmasses'][min_idx:max_idx]
            if 'moon_angles' in field:
                field['moon_angles'] = field['moon_angles'][min_idx:max_idx]

        if self.deltaT.size == 0:
            raise ValueError("No time left to observe the fields with the given max_airmass")
        # we also adjust the start_date and end_date
        self.start_date = self.deltaT[0]
        self.end_date = self.deltaT[-1] + timedelta(seconds=self.exposure_time)

    #@timeit
    def reorder(self, fields: dict, t: ap_time.Time, last_filter_id, last_filter_change, weights=[2, 1, 1], field_id: int = None):
        """Reorder the fields by distance to the given field."""
        # remove the field from the dictionary to avoid computing the distance to itself
        has_current_field = field_id is not None and field_id in fields
        if has_current_field is True:
            current_field = fields[field_id] if field_id in fields else None
        if len(list(fields.keys())) == 0:
            return fields
        for field in fields.values():
            if has_current_field is True:
                field['distance'] = angle(current_field['ra'], current_field['dec'], field['ra'], field['dec'])
            else:
                field['distance'] = 0

        dts = np.asarray([t, t + timedelta(seconds=self.exposure_time)])
        airmasses = self.airmass(list(fields.keys()), dts)
        moon_angles = self.moon_angle(list(fields.keys()), dts)

        for i, field_id in enumerate(fields.keys()):
            fields[field_id]['airmass'] = airmasses[i]
            fields[field_id]['moon_angle'] = moon_angles[i]
        for field_id in fields.keys():
            fields[field_id]['last_observed_diff'] = round((t - fields[field_id]['last_observed']).sec) if fields[field_id]['last_observed'] is not None else round(self.min_time_interval + 1)
        
        max_probdensity = max([field['probdensity'] for field in fields.values()])
        max_distance = max([field['distance'] for field in fields.values()])
        max_airmass = max([max(field['airmass']) for field in fields.values() if all(field['airmass'] < self.max_airmass)])
        max_last_observed = max([field['last_observed_diff'] for field in fields.values()]) + self.min_time_interval
        
        # set the scores to 0
        for field_id in fields.keys():
            fields[field_id]["score"] = 0

        for field_id in fields.keys():
            field = fields[field_id]
            if any(field['airmass'] >= self.max_airmass):
                continue
            elif len(field['filter_ids']) == len(self.filters):
                continue
            elif field['last_observed_diff'] < self.min_time_interval and has_current_field is True:
                continue
            elif len(field['filter_ids']) > 0 and field['filter_ids'][-1] == last_filter_id and (t - last_filter_change).sec < self.min_time_interval:
                continue
            elif field['distance'] == 0 and has_current_field is True:
                continue # we don't want to observe the same field twice in a row
            elif any(field['moon_angle'] == False):
                continue
            else:
                fields[field_id]["score"] = sum([
                    (weights[0] * ((field['probdensity']) / max_probdensity)),       # the higher the better
                    (weights[1] * (((max_distance - field['distance']) / max_distance) if (max_distance > 0 and field['distance'] != 0) else 0)),  # the lower the better
                    (weights[2] * (max(field['airmass']) / max_airmass)),                     # the higher the better
                ])
                # now wen apply the penalty based on how recently the field was observed
                fields[field_id]["score"] -= fields[field_id]["score"] * ((1 - (field['last_observed_diff'] / max_last_observed)) if max_last_observed > 0 and has_current_field else 0)
                #TODO: improve on the penalty calculation
        
        # sort the fields by score
        fields = {k: v for k, v in sorted(fields.items(), key=lambda item: item[1]['score'], reverse=True)}

        # add a penalty to the fields from the secondary grid (simply move them to the end of the list)
        primary = {}
        secondary = {}
        for field_id in fields.keys():
            if field_id < self.primary_limit:
                primary[field_id] = fields[field_id]
            else:
                secondary[field_id] = fields[field_id]

        fields = {**primary, **secondary}

        # all the fields with a score equal to zero are not observable, they go to the bottom of the list after the negative score fields

        non_null_fields = {}
        null_fields = {}
        for field_id in fields.keys():
            if fields[field_id]['score'] != 0:
                non_null_fields[field_id] = fields[field_id]
            else:
                null_fields[field_id] = fields[field_id]

        fields = {**non_null_fields, **null_fields}
        
        # in a clean way, print a table with the fields and their scores
        # print(f"Field scores at time {t.iso}:")
        # print("field_id\tscore\tprobdensity\t\tairmass\t\tlast_observed_diff\t\tfilters")
        # for field_id in list(fields.keys())[:5]:
        #     field = fields[field_id]
        #     print(f"{field_id}\t\t{field['score']}\t{field['probdensity']}\t\t{field['distance']}\t\t{field['airmass']}\t{field['last_observed_diff']}\t{field['filter_ids']}")
        # time.sleep(1)
        return fields
    
    @timeit
    def schedule(self, weights=[2, 1, 1]): #TODO: I'm just experimenting with this to see what can be done and how
        # this is not a good way to schedule observations, it's clearly suboptimal so far. I want to have a look at
        # sear, greedy slew and weighted to see if I can get something better
        """Schedule observations of the fields based on their observability and probdensity (greedy algorithm).
        Fields from the primary grid are scheduled first, then fields from the secondary grid."""
        # rank the observable fields by probdensity

        max_total_obs = len(self.deltaT)
        max_obs_per_filter = round(max_total_obs / len(self.filters)) if round(max_total_obs / len(self.filters)) < len(self.observable_fields) else len(self.observable_fields)
        if max_obs_per_filter < round(self.min_time_interval / self.exposure_time):
            max_obs_per_filter = round(self.min_time_interval / self.exposure_time)
        if max_total_obs > 0 and max_obs_per_filter == 0:
            max_obs_per_filter = 1
        max_obs_per_field = len(self.filters)

        print(f"\nobservable_fields: {len(self.observable_fields)}")
        print(f"max_total_obs (possible time-wise): {max_total_obs}")
        print(f"max_obs_per_filter: {max_obs_per_filter}")
        print(f"max_obs_per_field: {max_obs_per_field}\n")

        plan_per_filter = {filter_id: {} for filter_id in range(len(self.filters))}

        # first we keep only N fields with N = max_obs_per_filter if N < len(fields) else len(fields)

        # we schedule the fields for each filter
        t = self.start_date
        id = 0
        field_id = None
        filter_id = 0
        last_filter_change = self.start_date

        fields = deepcopy(self.observable_fields)
        with ProgressBar(total=max_total_obs, desc='Scheduling observations') as progress:
            while True:
                if t + timedelta(seconds=self.exposure_time) >= self.end_date:
                    print(f"End date reached")
                    break
                if max_total_obs > 0 and id >= max_total_obs:
                    print(f"Max total obs reached")
                    break
                # reorder the fields based on the weights
                fields = self.reorder(fields=fields, t=t, last_filter_id=filter_id, last_filter_change=last_filter_change, weights=weights, field_id=field_id)

                if round((t - last_filter_change).sec) >= self.min_time_interval and max_obs_per_filter - len(plan_per_filter[filter_id]) == 0:
                    filter_id = (filter_id + 1) % len(self.filters)
                    last_filter_change = t
                    t = t + timedelta(seconds=self.exposure_time)
                    continue
                elif round((t - last_filter_change).sec) >= self.min_time_interval and len(fields) == 0:
                    filter_id = (filter_id + 1) % len(self.filters)
                    last_filter_change = t
                    t = t + timedelta(seconds=self.exposure_time)
                    continue

                if len(fields) == 0:
                    fields = deepcopy(self.observable_fields)
                    field_id = None
                    t = t + timedelta(seconds=self.exposure_time)
                    continue

                field_id = list(fields.keys())[0]
                field = fields[field_id]

                if field['score'] == 0:
                    t = t + timedelta(seconds=self.exposure_time)
                    continue

                if len(fields[field_id]['filter_ids']) > 0 and fields[field_id]['filter_ids'][-1] == filter_id: # already observed in the current filter
                    old_filter_id = filter_id
                    filter_id = (fields[field_id]['filter_ids'][-1] + 1) % len(self.filters)
                    if self.filters[old_filter_id] != self.filters[filter_id]:
                        last_filter_change = t
                    
                for f_id in range(len(self.filters)):
                    if self.filters[f_id] == self.filters[filter_id] and f_id not in fields[field_id]['filter_ids'] and len(plan_per_filter[filter_id]) < max_obs_per_filter:
                        filter_id = f_id
                        break

                self.observable_fields[field_id]['filter_ids'].append(filter_id)
                self.observable_fields[field_id]['last_observed'] = t

                fields[field_id]['filter_ids'].append(filter_id)
                fields[field_id]['last_observed'] = t

                plan_per_filter[filter_id][field_id] = {
                    'id': id,
                    'field_id': field_id,
                    'filter_id': filter_id,
                    'filt': self.filters[filter_id],
                    'obstime': t,
                    'exposure_time': self.exposure_time,
                    'probability': fields[field_id]['probdensity'],
                    'score': fields[field_id]['score'],
                }
                id += 1
                t += timedelta(seconds=self.exposure_time)
                progress.update(1)
                continue
            
        start_time = ap_time.Time(plan_per_filter[0][list(plan_per_filter[0].keys())[0]]['obstime']) if len(plan_per_filter[0]) > 0 else self.start_date
        end_time = t if t < self.end_date else self.end_date

        print()
        observations = []
        for filter_id, plan in plan_per_filter.items():
            observations.extend(plan.values())
            print(f"{self.filters[filter_id]}: {list([obs['field_id'] for obs in plan.values()])}")

        # sort the observations by obstime
        observations = sorted(observations, key=lambda k: k['obstime'])

        # create a string that shows the field sequence, like (1(r) => 2(g) => 3(i) => 4(z))
        field_sequence = []
        for obs in observations:
            if len(field_sequence) == 0:
                field_sequence.append(f"{obs['field_id']}({obs['filt']})")
            elif obs['field_id'] != field_sequence[-1].split('(')[0]:
                field_sequence.append(f" => {obs['field_id']}({obs['filt']})")
        field_sequence = ''.join(field_sequence)
        print()
        print(f"Field sequence: {field_sequence}")

        # for each observations, convert the Time object to a iso string
        for obs in observations:
            if isinstance(obs['obstime'], ap_time.Time):
                obs['obstime'] = obs['obstime'].isot
                

        self.plan = {
            'name': f"ToO_{datetime.utcnow()}_{''.join(self.filters)}_{self.exposure_time}s",
            'nb_observations': len(observations),
            'nb_fields': len(set([obs['field_id'] for obs in observations])),
            'validity_window_start': start_time.isot,
            'validity_window_end': end_time.isot,
            'total_time': sum([obs['exposure_time'] for obs in observations]),
            'planned_observations': observations,
        }

        # compute the stats
        self.get_stats()

        # add the stats to the plan
        self.plan['stats'] = self.stats

        return self.plan

    def save_plan(self, filename, plan=None):
        """Save the plan to a json file."""
        # create the directories of the path if they dont exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.plan if plan is None else plan, f, indent=4)
        print(f'\nSaved plan to "{filename}"')

    def get_stats(self):
        # get the unique filters
        filters = set([obs['filt'] for obs in self.plan['planned_observations']])
        # get the unique fields per unique filter
        fields_per_filter = {}
        for filter in filters:
            fields_per_filter[filter] = set([obs['field_id'] for obs in self.plan['planned_observations'] if obs['filt'] == filter])

        # now, we'll calculate the overlap between instrument fields in the plan and the skymap fields
        stat_per_filter = {}
        for filter in filters:
            probability = sum([self.observable_fields[field_id]['probdensity'] for field_id in fields_per_filter[filter]])
            area = sum([self.observable_fields[field_id]['pixel_area'] for field_id in fields_per_filter[filter]]) * (180/math.pi)**2
            stat_per_filter[filter] = {
                'probability': probability,
                'area': area,
                'nb_fields': len(fields_per_filter[filter]),
            }

        self.stats = stat_per_filter
        return stat_per_filter
