#!/usr/bin/env python
"""
Script takes data from Level0 to Level1a data
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime as dt

pd.options.mode.chained_assignment = None


def to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.NaN

expected_ranges = {
    'CO2x': [0, 4000],
    'Lat': [-90, 90],
    'Lon': [-180, 360],
    'Pres_atm_hpa': [800, 1200],
    'Temp_intake': [-2, 35],
    'Salt_tsg': [25, 38],
    'Pres_equ_hpa': [-10, 10],
    'Temp_equ': [-2, 35],
    'Pres_licor_hpa': [800, 1200],
}

required_columns = [
    "Type",
    "Date",
    "Time",
    "Lat",
    "Lon",
    "CO2x",
    "STD",
    "Pres_atm_hpa",
    "Pres_equ_hpa",
    "Pres_licor_hpa",
    "Temp_equ",
    "Temp_intake",
    "Salt_tsg",
    "flow_h2o_Lmin",
    "flow_licor_cm3min",
]


def concat_co2_files(directory, verbose=False, dt_fmt="%d/%m/%y%H:%M:%S"):
    """
    Imports all the pCO2 files from the given directory. Files have to meet the
    criteria of ending on *dat.txt. The data is read in as a pandas DataFrame.
    Data is indexed by date and time. You may hve to give the date and time
    format (dt_fmt='%d/%m/%y%H:%M:%S')

    You also have to standardise the headers. There should be a CSV file in the
    input directory that gives standard headers.
    See the standard_headers_template.csv for more info
    """
    def datefunc(datestr_date, datestr_time):
        # This subfunction parses dates
        try:
            datestr = datestr_date + datestr_time
            return dt.strptime(datestr, dt_fmt)
        except:
            return dt.today()

    def standard_headers(directory, df):
        file = glob(os.path.join(directory, '*header*.csv'))
        if file:
            hdrs = pd.read_csv(file[0], comment="#")

            if df.columns.size != hdrs.raw_keys.size:
                raise Exception('Not the same number of headers. '
                                'A list of the headers follows: \n'
                                '%s' % str(df.columns))
            elif all(df.columns.values == hdrs.raw_keys.values):
                df.columns = hdrs.working_keys
                return df
            else:
                for row in np.arange(df.columns.size):
                    print(df.columns[row], hdrs.raw_keys[row])
                raise Exception('Columns of config_header.csv '
                                'do not match the data files.'
                                '\nHere is a list of the columns '
                                'in the file: \n %s' % str(df.columns))
        else:

            headers = df.columns.values
            fname = os.path.join(directory, 'config_header.csv')
            fobj = open(fname, 'w')

            fobj.write("# This is a template for config_headers.\n# Change working_keys to contain the required columns shown in the list below.\n# It is easier if the other variables don't contain spaces in the working_keys.\n# These rows (with leading #) do not have to be removed.\n")
            fobj.write("# REQUIRED COLUMNS: " + required_columns[0])
            for key in required_columns[1:]:
                fobj.write(" | {}".format(key))
            fobj.write('\nworking_keys,raw_keys\n')

            for h in headers:
                fobj.write(h.replace(" ", "_").replace("/", "") + ',' + h + '\n')
            fobj.close()

            raise Exception('config_header.csv is missing in: '
                            '%s' % directory)

    # create a file list of all the files in the given directory
    # uses the wildcard to create a list of files
    file_list = glob(os.path.join(directory, '*dat.txt'))

    print('=' * 64)
    print()
    print('1. IMPORTING DATA')
    print('1.1 - Reading %d files from %s' % (len(file_list), directory))
    # file_list = file_list[:]
    data = []  # pd.DataFrame()  # pre-assigning a dataframe that stores data
    for file in file_list:
        if len(open(file).readlines()) < 2:
            continue
        if verbose:
            print('\t', file)
        # read the txt files as a csv. Just change the seperator with "\t" the
        # dates are read in as the files index. This is done using date_parser
        # and datetime.strptime. The reader has to be told to use concatenated
        # variable "DateTime" as the index column.
        df = pd.read_csv(file,
                         sep="\t",
                         parse_dates={'DateTime': [2, 3]},
                         date_parser=datefunc,
                         index_col='DateTime',
                         keep_date_col=True,
                         na_values=['-'],
                         error_bad_lines=False,
                         warn_bad_lines=True,
                         skipinitialspace=True)

        # append the temporary "df" to the pre-assigned "data"
        data.append(df)

    # sort the data by the date_time index to assure data is stored
    data = pd.concat(data)  # .reindex_axis(df.keys(), axis=1)

    data = standard_headers(directory, data)
    data['DateTime'] = data.index.to_julian_date()

    # extra catch line for random data that gets created from bad files
    data = data.loc[data['Type'] != '0']

    data.sort_index(inplace=True, axis=0)

    cols = ['STD', 'CO2x', 'Temp_intake', 'Pres_atm_hpa', 'Salt_tsg']
    data.loc[:, cols] = data.loc[:, cols].applymap(to_float)

    return data


def sanity_checker(df, debug=False):

    txt = '1.2 - Sanity check: points within ranges'
    txt += '\n|{0:.18}|{0:.14}|{0:.9}|'.format('-' * 43)
    txt += ('\n|{0:^18}|{1:^14}|{2:^9}|'
            '').format('NAME', 'RANGE', 'PCT')
    txt += '\n|{0:.18}|{0:.14}|{0:.9}|'.format('-' * 43)
    for key in expected_ranges:
        if debug:
            print(key)
        dat = df.ix[:, key]
        liml, limu = expected_ranges[key]
        i = (dat > liml) & (dat < limu)
        pct = float(i.sum()) / i.size * 100
        if pct < 95:
            key = '*' + key
        else:
            key = ' ' + key
        sd = key, liml, limu, pct
        txt += ('\n|{0:<18}| {1:=-5.0f} :{2:=-5.0f} '
                '| {3:=8.2f}|').format(*sd)
    txt += ('\n\n *Salt_tsg doesnt have to be in range for script to work')
    print(txt)


if __name__ == '__main__':

    data1 = concat_co2_files('../Level_0/2010_Gough', verbose=True)
