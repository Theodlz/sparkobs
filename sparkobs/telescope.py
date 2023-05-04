import json
import os
import warnings
from datetime import timedelta

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

from sparkobs.utils import compute_tiles_probdensity, timeit


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
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.min_time_interval = config['min_time_interval']
        self.filters = config["filters"]
        self.exposure_time = config["exposure_time"]
        self.primary_limit = config['primary_limit']
        self.moon_dt = None

        self.adjust_dates()
        self.compute_deltaT()

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

    def target(self, field_id):
        """Return an `astroplan.FixedTarget` representing the target of this
        observation."""
        try:
            field = self.fields[field_id]
        except KeyError:
            print(f'Cannot find field {field_id}')
            return None

        coord = SkyCoord(field["ra"], field["dec"], unit='deg')
        return astroplan.FixedTarget(name=field_id, coord=coord)

    def airmass(self, field_id, time, below_horizon=np.inf):
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

        output_shape = time.shape
        time = np.atleast_1d(time)
        target = self.target(field_id)
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

    def altitude(self, time, target):
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
        return self.observer.altaz(time, target).alt

    def moon_angle(self, field_id, time):
        """Return the angle between the field and the moon at a given time.

        Parameters
        ----------
        time : `astropy.time.Time` or list of astropy.time.Time`
            The time or times at which to calculate the angle
        min_angle : scalar, Numeric
            The minimum angle between the field and the moon

        Returns
        -------
        angle : ndarray
           The angle between the field and the moon at the requested times
        """
        output_shape = time.shape
        time = np.atleast_1d(time)
        target = self.target(field_id)
        angles = []

        if self.moon_dt is None or self.moon_dt.shape != time.shape:
            self.moon_dt = np.asarray([get_moon(t, self.location) for t in time])

        for i, t in enumerate(time):
            angle = self.moon_dt[i].separation(target.coord).to('degree').value
            angles.append(angle)
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
        print(f'Adjusted start date: {self.start_date.iso}')
        print(f'Adjusted end date: {self.end_date.iso}')
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
        deltaT = np.arange(0, (self.end_date - self.start_date).sec, self.min_time_interval * 60)
        self.deltaT = np.asarray([self.start_date + timedelta(seconds=delta) for delta in deltaT])

    def compute_fields_airmasses(self):
        """Compute the airmass of each field at each deltaT."""
        for field_id in tqdm.tqdm(self.fields.keys(), desc='Computing airmasses'):
            # we compute the airmass at the start and end of one exposure to see if the field is observable at least once at each deltaT
            airmasses_start = self.airmass(field_id, self.deltaT)
            airmasses_end = self.airmass(field_id, self.deltaT + timedelta(seconds=self.exposure_time))
            self.fields[field_id]['airmasses'] = np.maximum(airmasses_start, airmasses_end)

    def compute_fields_moon_angles(self):
        """Compute the moon angle of each field at each deltaT."""
        for field_id in tqdm.tqdm(self.observable_fields.keys(), desc='Computing moon angles'):
            moon_angles_start = self.moon_angle(field_id, self.deltaT)
            moon_angles_end = self.moon_angle(field_id, self.deltaT + timedelta(seconds=self.exposure_time))
            self.observable_fields[field_id]['moon_angles'] = np.minimum(moon_angles_start, moon_angles_end)

    def select_observable_fields(self):
        """Select fields that are observable at least once at each deltaT, i.e. fields with airmasses < max_airmass at least once."""
        observable_fields = {}
        for field_id in tqdm.tqdm(self.fields.keys(), desc='Selecting observable fields'):
            field = self.fields[field_id]
            if np.any(field['airmasses'] < self.max_airmass):
                observable_fields[field_id] = field 
        self.observable_fields = observable_fields

    def drop_tiles_from_observable_fields(self):
        """Drop the tiles from the observable fields once they are not useful anymore."""
        for field_id in self.observable_fields.keys():
            self.observable_fields[field_id].pop('tiles', None)

    @timeit
    def compute_observability(self, skymap):
        """Compute the observability of each field at each deltaT + exposure_time (i.e. the end of at least one observation)
        and remove fields that are never observable at any deltaT + exposure_time, and without any overlap with the skymap."""

        self.compute_fields_airmasses()

        print()
        self.select_observable_fields()

        print()
        self.compute_fields_skymap_overlap(skymap)

        print()
        self.compute_field_probdensity(skymap)

        print()
        self.compute_fields_moon_angles()

        self.drop_tiles_from_observable_fields()
    
    def overlapping_field_tiles(self,field,moc):
        """Return the overlapping field tiles between a field and a skymap's moc."""
        tiles = [MOC.from_depth29_ranges(29, ranges=np.expand_dims(tile, axis=0)) for tile in field['tiles']]
        field_tiles= np.array(tiles)[[not(tile & moc) == MOC.new_empty(max_depth=29) for tile in tiles]]
        return field_tiles
    
    def compute_fields_skymap_overlap(self, skymap):
        """Update the observable fields to only keep the fields that overlap with the skymap."""
        observable_fields = {}
        for field_id in tqdm.tqdm(self.observable_fields.keys(), desc='Computing skymap observability'):
            field = self.observable_fields[field_id]
            field_tiles = self.overlapping_field_tiles(field, skymap["moc"])
            if len(field_tiles) > 0:
                observable_fields[field_id] = field
        self.observable_fields = observable_fields

    def compute_field_probdensity(self, skymap):
        """Compute the probdensity of each observable field."""
        ranges = skymap["ranges"]
        probdensities = skymap["probdensities"]

        # reformat the observable fields to a list of tiles that look like (field_id, tile[0], tile[1], probdensity) to use with numba
        tiles = NumbaList()
        for field_id in self.observable_fields.keys():
            for tile in self.observable_fields[field_id]["tiles"]:
                tiles.append((field_id, tile[0], tile[1]))

        with ProgressBar(total=len(tiles)*len(ranges), desc='Computing fields probdensity') as progress:
            new_tiles = compute_tiles_probdensity(tiles, ranges, probdensities, progress)

        # reformat the tiles back to a dictionary of fields
        fields = {}
        for tile in new_tiles:
            field_id = tile[0]
            if field_id not in fields:
                fields[field_id] = {**self.observable_fields[field_id], "tiles": [], "probdensity": 0}
            fields[field_id]["tiles"].append((tile[0], tile[2]))
            fields[field_id]["probdensity"] += tile[3]

        self.observable_fields = fields

    @timeit
    def schedule(self):
        """Schedule observations of the fields based on their observability and probdensity (greedy algorithm).
        Fields from the primary grid are scheduled first, then fields from the secondary grid."""
        # rank the observable fields by probdensity
        fields = self.observable_fields
        fields = {k: v for k, v in sorted(fields.items(), key=lambda item: item[1]['probdensity'], reverse=True)}

        # rerank based on whether the field is in the primary or secondary grid using self.primary_limit
        primary_fields = {}
        secondary_fields = {}
        for field_id in fields.keys():
            field = fields[field_id]
            if field_id < self.primary_limit:
                primary_fields[field_id] = field
            else:
                secondary_fields[field_id] = field
        fields = {**primary_fields, **secondary_fields}

        max_exp = len(self.filters)
        # schedule the fields
        plan = []
        for i, t in enumerate(self.deltaT):
            for field_id in fields.keys():
                # if this plan hasnt been scheduled yet
                if field_id in [p["field_id"] for p in plan]:
                    continue
                field = fields[field_id]
                if field['airmasses'][i] < self.max_airmass and field['moon_angles'][i] > self.min_moon_angle:
                    # we know that its under airmass and above moon angle at this timestep for one exposure
                    # see how many exposures we can do up to max = len(self.filter)
                    # while keeping the airmass under max_airmass and the moon angle above min_moon_angle
                    # we start at max and go down to 2
                    filters = []
                    for n_exp in range(max_exp, 1, -1):
                        if i + n_exp > len(self.deltaT):
                            continue
                        dt = self.deltaT[i] + timedelta(seconds=self.exposure_time * (n_exp - 1))
                        airmass = self.airmass(field_id, dt)
                        moon_angle = self.moon_angle(field_id, dt)
                        if airmass < self.max_airmass and moon_angle > self.min_moon_angle:
                            filters = self.filters[:n_exp]
                            break
                    
                    for i in range(len(filters)):
                        plan.append({
                            "obstime": (t+timedelta(seconds=self.exposure_time*i)).iso,
                            "field_id": field_id,
                            "filt": filters[i],
                            "exposure_time": self.exposure_time,
                            "probdensity": field['probdensity'],
                        })
                    break

        self.plan = plan

    def save_plan(self, filename, plan=None):
        """Save the plan to a json file."""
        # create the directories of the path if they dont exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.plan if plan is None else plan, f, indent=4)
        print(f'\nSaved plan to "{filename}"')