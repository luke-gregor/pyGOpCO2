import numpy as np
from pylab import num2date
from itertools import ifilter
from time import struct_time
import pylab as plt


class TimeArray(np.ndarray):

    """
    The TimeArray is a class that overcomes the clumbsyness of going between
    python datenum and datetime.

    The TimeArray is based on the numpy.ndarray module, however you can also
    access arrays of any timetuple attributes as arrays. This also applies
    for datetime and string outputs.

    INPUT:  datenum array (days since 0001-01-01 00:00:00)
    OUTPUT: a TimeArray (np.ndarray)
                .* all ndarray methods
                .tm_year
                .tm_mon
                .tm_yday
                .tm_mday
                .tm_wday
                .tm_hour
                .tm_min
                .tm_sec
                .datetime
                .timetuple
                .datestr(fmt='%Y-%m-%d')
                .matlab_datenum
    """

    def __new__(cls, input_array, matlab_datenum=False):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj = obj - (366 * matlab_datenum)
        # add the new attribute to the created instance

        for tm in ifilter(lambda s: s.startswith('tm'), dir(struct_time)):
            setattr(obj, tm, cls._get_timetuple_output(obj, tm))

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

    def _get_timetuple_output(self, tm_str):
        out = [getattr(d.timetuple(), tm_str) for d in self._datetime]
        return np.array(out).squeeze()

    def datestr(self, fmt='%Y-%m-%d'):
        out = [d.strftime(fmt) for d in self._datetime]
        return np.array(out).squeeze()

    @property
    def _datetime(self):
        return np.array(num2date(self))

    @property
    def datetime(self):
        return self._datetime.squeeze()

    @property
    def timetuple(self):
        out = [d.timetuple() for d in self._datetime]
        return np.array(out).squeeze()

    @property
    def matlab_datenum(self):
        return np.array(self + 366).squeeze()


def grid_data(df, space_res, time_res, std_limit=7):
    from pylab import num2date

    def bin_coords(coord_series, res):
        hr = res / 2.
        a = np.around((coord_series + hr) / res) * res + hr
        return a

    rs = space_res
    rt = int(time_res)

    df['Lon_grd'] = bin_coords(df.Lon, rs)
    df['Lat_grd'] = bin_coords(df.Lat, rs)

    df['iLon'] = (df.loc[:, 'Lon_grd'] + 180 + rs/2) / rs
    df['iLat'] = (df.loc[:, 'Lat_grd'] + 90. + rs/2) / rs

    time = TimeArray(df.DateTime)
    df['Day'] = time.tm_mday
    df['Year'] = time.tm_year
    df['Month'] = time.tm_mon
    df['iJulDay'] = (time.tm_yday / rt).astype(int)

    onlyEQU = df.Type.str.startswith('EQU')
    df = df.loc[onlyEQU]

    group = ['Year', 'iJulDay', 'Lat_grd', 'Lon_grd']
    df_grp = df.groupby(by=group, as_index=False)
    df_std = df_grp.std()
    df_avg = df_grp.mean()

    qc_bin_std = df_std['CO2p'] < std_limit

    df = df_avg.loc[qc_bin_std]
    df.loc[:, 'CO2p_std'] = df_std.loc[qc_bin_std, 'CO2p']

    df.index = num2date(df['DateTime'])

    return df

