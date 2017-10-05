#!/usr/bin/env python
"""
Script takes data from Levelb to Level2 data.


--------------
TO DO: BUG LOG
--------------
MATLAB vs Python: There seems to be a big difference in SANAE59 Data
                  What is the cause of this error?
atmospheric fCO2: Is it processed the same way as sea?
                  Occasionally current interpolation doesn't work
                  **Correction needs to be done at atm pressure
calibration corr: accounts for the difference between the MATLAB Script
                  This then needs to be adapted to interpoate first
                  then take the mean


"""

from __future__ import print_function
import numpy as np
import pandas as pd
from . import diagnostic_plots as dp


class UnderwayCO2(pd.DataFrame):

    @property
    def ATM(self):
        return self[self.Type.str.startswith("ATM")]

    @property
    def EQU(self):
        return self[self.Type.str.startswith("EQU")]

    @property
    def STD1(self):
        return self[self.Type.str.startswith("STD1")]

    @property
    def STD2(self):
        return self[self.Type.str.startswith("STD2")]

    @property
    def STD3(self):
        return self[self.Type.str.startswith("STD3")]

    @property
    def STD4(self):
        return self[self.Type.str.startswith("STD4")]

    def check_columns(self):

        print(self.columns)


flag4 = {'CO2x':                 [-10,  600],
         'CO2x@abs.diff':        [0.0,   10],
         'flow_h2o_Lmin':        [2.0,  3.2],
         'flow_licor_cm3min':    [40.,  150],
         'Pres_atm_hpa':         [900, 1100],
         'Pres_equ_hpa@abs':     [0,     10],
         'Temp_equ':             [-2.,   32],
         'Temp_intake':          [-2.,   32],
         'Temp_equ@abs.diff':    [0.0,  .15],
         'Temp_intake@abs.diff': [0.0,  0.5],
         'Temp_warming':         [0.0,  3.0],
         'Temp_warming@outlier': [0.0,  0.3],
         'Salt_tsg':             [5.0,   40],
         }


def main(filename):
    from level0_to_level1 import concat_co2_files

    df1 = concat_co2_files('../Level_0/2009_SANAE49/')
    df2a = xCO2_to_fCO2(df1)
    # from diagnostic_plots import plot_qaulity_control, plot_co2_standards
    # plot_qaulity_control(df2a, '/Users/luke/Desktop/SANAE50_diagns.png')
    # plot_co2_standards(df2a, '/Users/luke/Desktop/SANAE50_co2plt.png')
    df2b = quality_control_flags(df2a)

    print(df2b)


def xCO2_to_fCO2(df, atm_pres_height_correction=False):
    """
    Note that this function only works if the headers have been standardised.
    This function uses the methods from the Pierrot et al. (2009) to convert
    xCO2 to pCO2, where
    """
    print('\n', '=' * 64)
    print('\n2. CONVERTING xCO2 TO fCO2:')

    assert len(df) > 0, 'Check inputs: no data passed to the function.'

    # filling missing data
    df['Salt_tsg'] = _fill_missing(df['Salt_tsg'], fill_value=34.5)

    # if equilibrator is differential make it absolute
    df['Pres_lab_hpa'] = df['Pres_equ_hpa'] + df['Pres_licor_hpa']

    # calibrate the offset
    df['CO2x_cor'] = df['CO2x'] - _calibration_offset_xCO2(df)

    # processing xCO2 for ocean measurements
    sea = df['Type'].str.startswith('EQU')
    press_corr = _press_H2O_corrected(df['Pres_lab_hpa'], df['Temp_equ'], df['Salt_tsg'])
    ideal_fact = _ideal_gas_virial_exp(df['Pres_lab_hpa'], df['Temp_equ'], df['CO2x_cor'])
    dtemp_fact = _temperature_correction(df['Temp_intake'], df['Temp_equ'])
    df['CO2p_equ'] = df.loc[sea, 'CO2x_cor'] * press_corr / 1013.25
    df['CO2p_sst'] = df['CO2p_equ'] * dtemp_fact
    df['CO2f_sst'] = df['CO2p_sst'] * ideal_fact

    # processing atmospheric xCO2
    if atm_pres_height_correction:
        press_atm = _press_height_correction(df['Pres_atm_hpa'], df['Temp_intake'])
    else:
        press_atm = df["Pres_atm_hpa"]
    press_h20_corr = _press_H2O_corrected(press_atm, df['Temp_intake'], df['Salt_tsg']) / 1013.25
    ideal_fact = _ideal_gas_virial_exp(press_atm, df['Temp_intake'], df['CO2x_cor'])
    df['CO2x_atm'] = _interpolate_atm_CO2f(df)
    df['CO2p_atm'] = df['CO2x_atm'] * press_h20_corr
    df['CO2f_atm'] = df['CO2p_atm'] * ideal_fact

    return UnderwayCO2(df)


def quality_control_flags(df, verbose=True):
    """This is a positive filter. It keeps what you define as the mask."""
    df['Temp_warming'] = df['Temp_equ'] - df['Temp_intake']
    i = df.ix[:, 'Salt_tsg'].isnull() | (df.ix[:, 'Salt_tsg'] == 0)
    df.loc[i, 'Salt_tsg'] = 34.5

    masks = [((df['Lat'] != 0) & (df['Lon'] != 0)), (df['Type'] != '0')]

    info = '\n'
    info += '=' * 64
    info += '\n'
    info += '\n3. QUALITY CONTROL'
    info += '\n|-{0:.25}|{0:.19}|{0:.15}|'.format('-' * 43)
    info += '\n| {0:^25}|{1:^19}|{2:^15}|'.format('KEYS', 'LIMIT', 'FLAGGED')
    info += '\n|-{0:.25}|{0:.19}|{0:.15}|\n'.format('-' * 43)
    for key in flag4:
        masks += _make_qc_mask(df, key, flag4[key]),
        info += ('| {0:<25}| {3:=7.2f} : {4:=-7.2f} | {1:>5} / {2:>5} |'
                 '\n').format(key, (~masks[-1]).sum(), masks[-1].size,
                              flag4[key][0], flag4[key][1])

    mask = np.array(masks).prod(0).astype(bool)

    df['CO2f_WOCE_flag'] = 0
    df.ix[mask,  'Flag_CO2f_WOCE'] = 2
    df.ix[~mask, 'Flag_CO2f_WOCE'] = 4

    df = UnderwayCO2(df)
    df.masks = masks

    if verbose:
        info += '|-{0:.25}|{0:.19}|{0:.15}|'.format('-' * 43)
        info += ('\n| {0:<25}| {3:^7} : {4:^7} | {1:>5} / {2:>5} |'
                 '').format('TOTAL', (~mask).sum(), mask.size, 'NAN', 'NAN')
        print(info)

    return df


def _fill_missing(ser, fill_value, limit=15):
    """Fill salinity for 5 points and if not then 34.5"""

    name = getattr(ser, 'name', '')
    print('2.0 - Filling missing or 0 %s values to %.2f' % (name, fill_value))
    ser[ser == 0] = np.NaN
    ser = ser.ffill(limit=limit)
    ser[ser.isnull()] = fill_value

    return ser


def _calibration_offset_xCO2(df):

    def linreg_offset(x, y):
        """
        vectorised linear regression that avoids having to do
        for loops for each linear regression.
        """
        x = x.values
        y = y.values
        n = x.shape[1]
        xmean = np.mean(x, axis=1, keepdims=True)
        ymean = np.mean(y, axis=1, keepdims=True)

        cp_xy = (x - xmean) * (y - ymean)
        cp_xx = (x - xmean)**2
        cov_xy = cp_xy.sum(1) / (n - 1)
        cov_xx = cp_xx.sum(1) / (n - 1)

        slope = cov_xy / cov_xx
        intercept = ymean.squeeze() - slope * xmean.squeeze()

        return slope, intercept

    """
    input method is whether to use a:
        1. factor: fraction of xCO2 over known standards
        2. offset: difference between xCO2 and standards
    """

    print('2.1 - Calibration factor from standards')

    # Find only STDs, but zero and span
    i = df['Type'].str.startswith('STD') & \
        ~df['Type'].str.endswith('s') & \
        ~df['Type'].str.endswith('z') & \
        ~df['Type'].str.endswith('4s-DRAIN') & \
        ~df['Type'].str.endswith('4z-DRAIN')

    names = 'STD1', 'STD2', 'STD3', 'STD4'
    stds = pd.DataFrame(np.NaN, index=df.index, columns=names)
    meas = pd.DataFrame(np.NaN, index=df.index, columns=names)

    for s in names:
        j = i & df['Type'].str.startswith(s)
        meas.loc[j, s] = df.loc[j, 'CO2x']
        stds.loc[j, s] = df.loc[j, 'STD']

    k = meas.notnull().any()
    meas = meas.loc[:, k]
    stds = stds.loc[:, k]

    offs = meas - stds
    if (abs(offs.median()) > 15).any() | (offs.shape[1] < 3):
        dp.plot_co2_standards(df)
        txt = offs.median().__repr__()
        raise Exception("Standards are not within acceptable ranges. \n"
                        "Check that the correct values have been set. \n"
                        "Median differences between measured and standards "
                        "are: \n" + txt)
    offs[abs(offs) > 10] = np.NaN
    offs.interpolate('linear', inplace=True)
    offs.bfill(inplace=True)
    stds.interpolate('linear', inplace=True)
    stds.bfill(inplace=True)

    m, c = linreg_offset(stds, offs)
    offset_correction = df['CO2x'] * m + c

    return offset_correction


def _press_height_correction(press_hPa, temp_C, height_diff=-10.):

    Tk = temp_C + 273.15
    Pa = press_hPa * 100  # pressure in Pascal
    # Correction for pressure based on sensor height
    R = 8.314  # universal gas constant (J/mol/K)
    M = 0.02897  # molar mass of air in (kg/mol) - Wikipedia
    # Density of air at a given temperature. Here we assume
    # that the air temp is the same as the intake temperature
    d = Pa / (R / M * Tk)
    g = 9.8  # gravity in (m/s2)
    h = height_diff  # height in (m)
    # correction for atmospheric
    press_height_corr_hpa = (Pa - (d * g * h)) / 100.

    return press_height_corr_hpa


def _press_H2O_corrected(press_hPa, temp_C, salt):
    print('2.2 - Water vapour factor from temp and salt')
    # Weiss and Price (1980)
    # Convert temperature C to K
    Tk = temp_C + 273.15
    S = salt
    P_atm = press_hPa / 1013.25

    # Equation comes straight from Weiss and Price (1980)
    pH2O = np.exp(24.4543 -
                  67.4509 * (100. / Tk) -
                  4.8489 * np.log(Tk / 100.) -
                  0.000544 * S)
    press_H2O_corr = (P_atm - pH2O) * 1013.25

    return press_H2O_corr


def _temperature_correction(temp_intake, temp_equ):
    print('2.3 - Temperature correction factor')
    # see the Takahashi 1993 paper for full description
    delta_temp = temp_intake - temp_equ
    temp_correct_factor = np.exp(0.0423 * delta_temp)
    return temp_correct_factor


def _ideal_gas_virial_exp(press_hPa, temp_C, xCO2_ppm):
    print('2.4 - Correcting for non-ideal behaviour of CO2')
    # Temperature in (Kelvin)
    Tk = temp_C + 273.15
    # pressure in (Pa)
    Pa = press_hPa * 100.
    # CO2 in (atm)
    xCO2 = xCO2_ppm * 1e-6
    # Universal Gas constant in (J.K-1.mol-1)
    R = 8.314
    # virial coefficient of CO2 in air (m3.mol-1)
    vca = (57.7 - 0.118 * Tk) * 1e-6
    # first virial coefficient of pure CO2 (m3.mol-1)
    bT = (-1636.75 +
          (12.0408 * Tk) -
          (3.27957e-2 * Tk**2) +
          (3.16528e-5 * Tk**3)) * 1e-6

    ideal_gas_corr = np.exp(Pa * (bT + 2 * (1 - xCO2)**2 * vca) / (R * Tk))

    return ideal_gas_corr


def _interpolate_atm_CO2f(df, lim=8):
    # Output here is different to what the scipt puts out
    # Find ATM

    df['Type3'] = df.Type.str[:3]
    i = df['Type3'] == 'ATM'
    # identify groupings of standards
    # I use the difference between only ATM samples
    df['groups'] = df.loc[i, 'DateTime'].diff() > 0.007  # +- 10min of day
    df.loc[:, 'groups'] = df.loc[:, 'groups'].ffill().cumsum()

    grps = df.groupby(['Type3', 'groups']).CO2x_cor.median()['ATM']
    stds = df.groupby(['Type3', 'groups']).CO2x_cor.std()['ATM']
    meds = grps.rolling(5, center=True).mean().ffill().bfill()
    grps.loc[((grps - meds) > 1) | (stds > 1)] = np.nan
    smth = grps.interpolate('cubic').bfill()

    df.loc[:, 'CO2x_atm'] = np.NaN
    for g in grps.index:
        df.loc[(df.groups == g) & (df.Type.str.startswith('ATM')), 'CO2x_atm'] = smth.loc[g]

    CO2x_atm = df.CO2x_atm.interpolate('linear').bfill()

    return CO2x_atm


def _make_qc_mask(df, key, lims, verbose=True):
    if '@' in key:
        keyn, command = key.split('@')
        assert keyn in df.keys(), 'key (%s) does not exist'
        dat = df[keyn]
        while command:
            if 'diff' in command:
                command = command.replace('diff', '')
                dat = dat.interpolate('time').diff()
            elif 'abs' in command:
                command = command.replace('abs', '')
                dat = abs(dat)
            elif 'outlier' in command:
                command = command.replace('outlier', '')
                a = dat
                b = a.rolling(window=240, min_periods=1).median()
                dat = abs(b - a)
            command = command.replace('.', '')
    elif key in df.keys():
        dat = df[key]

    mask = (dat >= lims[0]) & (dat < lims[1])
    mask.name = key

    return mask


if __name__ == '__main__':

    main('testfilename')
