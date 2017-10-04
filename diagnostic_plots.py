#!/usr/bin/env python
"""
Script takes data from Levelb to Level2 data.

"""

import pylab as plt
from .level1_to_level2 import UnderwayCO2
import numpy as np

try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    pass


def main():
    pass


def plot_qaulity_control(df, sname=None):

    def text_pos(ax, pos=[0.95, 0.1]):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xloc = xmin + pos[0] * (xmax - xmin)
        yloc = ymin + pos[1] * (ymax - ymin)
        return xloc, yloc

    df.index.name = ''
    plt.rcParams['axes.titlesize'] = 14.0
    plt.rcParams['font.size'] = 12

    kwdsf = dict(alpha=0.2, zorder=1)
    kwdsp = dict(marker='.',
                 alpha=0.25,
                 lw=0,
                 ms=5,
                 legend=False,
                 zorder=3,
                 color='k',
                 label='')

    # creating variable dictionary. Keys are consistent
    warming = df['Temp_equ'] - df['Temp_intake']
    warming_median = warming.rolling(window=400, min_periods=1).median()
    var = {0: df['flow_licor_cm3min'],
           1: df['flow_h2o_Lmin'],
           2: warming,
           3: abs(df['Temp_equ'].interpolate('time').diff()),
           4: abs(df['Pres_equ_hpa']),
           5: df['Pres_atm_hpa']}

    # Creating plotting variables
    # rlim1l, rlim1u, gliml, glimu, rlim2l, rlim2u
    lims = {0: np.array([0.0, 50., 55., 140, 180, 200]),
            1: np.array([1.5, 2.0, 2.5, 2.8, 3.4, 4.0]),
            2: np.array([-2., 0.0, 0.1, 0.1, 3.0, 4.0]),
            3: np.array([0.0, 0.0, 0.0, .05, 0.15, 0.3]),
            4: np.array([0.0, 0.0, 0.0, 1.0, 2.5, 5.0]),
            5: np.array([850, 900, 950, 1035, 1050, 1060])}

    title = {0: u'Licor air flow  (cm$^3$/ min)',
             1: u'Water flow  (L / min)',
             2: (u'Water warming [$\Delta$T = T$^{equ}$ - T$^{sst}$]'
                 u' showing ~10hr moving median + 0.3 (C)'),
             3: u'Equilibrator temperature changes',
             4: u'Absolute value of equilibrator pressure (hPa)',
             5: u'Atmospheric air pressure (hPa)'}

    fig, ax = plt.subplots(3, 2, figsize=[15, 15])
    ax = np.reshape(ax, -1)

    # Air flow throught the Licor
    for i in range(6):
        ylim = lims[i][[0, 5]]
        var[i].clip(*ylim).plot(ax=ax[i], ylim=ylim, **kwdsp)
        xlim = ax[i].get_xlim()
        ax[i].fill_between(xlim, *lims[i][[0, 1]], color='r', **kwdsf)
        ax[i].fill_between(xlim, *lims[i][[4, 5]], color='r', **kwdsf)
        ax[i].fill_between(xlim, *lims[i][[2, 3]], color='b', **kwdsf)
        ax[i].set_title(title[i])

        xpos = text_pos(ax[i])[0]
        ydif = np.diff(lims[i][[0, 1, 4, 5]].reshape(2, 2), 1).argmax()
        ypos = lims[i][[[0, 1], [4, 5]][ydif]].mean()
        rjct = np.ma.masked_outside(var[i], *lims[i][[1, 4]]).mask.sum()
        props = dict(facecolor=ax[i].get_axis_bgcolor(),
                     pad=12, lw=0)
        ax[i].text(xpos, ypos, 'Bad points: %d' % rjct,
                   ha='right', va='center', bbox=props)

    # Plot specific changes
    warm_out_u = (warming_median + 0.3)
    warm_out_l = (warming_median - 0.3)
    warm_out_l[warm_out_l > 3] = 3
    warm_out_u[warm_out_u > 3] = 3
    warm_out_l[warm_out_l < 0] = 0
    warm_out_u[warm_out_u < 0] = 0
    x = ax[2].get_lines()[0].get_xdata()
    ax[2].fill_between(x, warm_out_u, 3, color='y', **kwdsf)
    ax[2].fill_between(x, warm_out_l, 0, color='y', **kwdsf)
    ax[2].fill_between(x, warm_out_l, warm_out_u, color='b', **kwdsf)
    map(lambda a: a.set_xticklabels(''), ax[:4])

    # Figure changes
    fig.tight_layout()
    fig.text(0.5, .96,
             ('Underway CO$_2$ diagnostics with '
              'suggested WOCE flag 4 criteria shown in red'),
             va='bottom', ha='center', size=20)
    fig.subplots_adjust(hspace=0.22, wspace=.1,
                        top=0.94, bottom=0.04, left=0.05, right=0.95)

    if sname:
        fig.savefig(sname, dpi=300)

    return fig


def plot_co2_standards(df, sname=None):
    kwdsp = {'marker': '.', 'lw': 0, 'ms': 5, 'legend': False, 'zorder': 5}
    kwdsf = dict(alpha=0.2, zorder=1)

    dfp = UnderwayCO2(df)

    var01 = dfp.EQU.CO2x.clip(165, 605)
    var02 = dfp.ATM.CO2x.clip(165, 605)

    fig = plt.figure(figsize=[15, 15])
    ax0 = plt.subplot2grid([3, 2], [0, 0], colspan=2)
    ax1 = plt.subplot2grid([3, 2], [1, 0])
    ax2 = plt.subplot2grid([3, 2], [1, 1])
    ax3 = plt.subplot2grid([3, 2], [2, 0])
    ax4 = plt.subplot2grid([3, 2], [2, 1])

    # fCO2 & CO2atm
    var01.plot(ax=ax0, color='black', label='EQU', **kwdsp)
    var02.plot(ax=ax0, color='gray', label='ATM', **kwdsp)
    ax0.fill_between(ax0.get_xlim(), 200, 500, color='g', **kwdsf)
    ax0.fill_between(ax0.get_xlim(), 590, 700, color='r', **kwdsf)
    ax0.fill_between(ax0.get_xlim(), 0, 170, color='r', **kwdsf)
    ax0.legend(loc=1, ncol=2, fontsize=12, framealpha=1, markerscale=2)
    ax0.set_title('Uncalibrated xCO$_2$ ($\mu$atm)')
    ax0.set_ylim(160, 610)
    plt.setp(ax0.xaxis.get_majorticklabels(), rotation=0)

    kwdsp.update({'ms': 8, 'lw': 1, 'alpha': .6})
    for c, ax in enumerate([ax1, ax2, ax3, ax4]):
        std = 'STD%d' % (c + 1)
        STD = getattr(dfp, std)

        STD.CO2x.plot(ax=ax, color='k', **kwdsp)
        ax.axhline(STD.STD[5], c='g')
        # ax.set_ylim(STD.STD[5] - 7, STD.STD[5] + 7)
        ax.set_title('xCO$_2$ %s (%.2f $\mu$atm)' % (std, STD.STD[5]))

    ax1.set_xticklabels('')
    ax2.set_xticklabels('')

    fig.text(0.5, 0.96, 'Underway xCO$_2$ with calibration standards',
             va='bottom', ha='center', size=20)
    fig.subplots_adjust(hspace=0.17, wspace=.1,
                        top=0.94, bottom=0.04, left=0.05, right=0.95)

    if sname:
        fig.savefig(sname, dpi=300)

    return fig


def plot_ocean_params(df, sname=None):

    kwds = dict(marker='.', lw=0, ms=5, legend=False, zorder=5)
    df = UnderwayCO2(df)

    fig = plt.figure(figsize=[9, 6])
    ax0 = plt.subplot2grid([2, 1], [0, 0])
    ax1 = plt.subplot2grid([2, 1], [1, 0])

    var01 = df.EQU.CO2x.clip(100, 620)
    var02 = df.ATM.CO2x.clip(100, 620)
    var11 = df.EQU.Temp_intake.clip(-2, 25)
    var12 = df.EQU.Salt_tsg.clip(33, 36)

    # xCO2 and xCO2atm
    var01.plot(ax=ax0, color='black', label='xCO2$_{sea}$', **kwds)
    var02.plot(ax=ax0, color='gray', label='xCO2$_{atm}$', **kwds)
    ax0.set_ylim(160, 610)
    ax0.set_ylabel('Uncalibrated xCO$_2$ (ppm)')
    ax0.set_xticklabels([])
    ax0.legend(loc=1, ncol=2, fontsize=12, markerscale=2, frameon=1)
    plt.setp(ax0.xaxis.get_majorticklabels(), rotation=0)

    # Temperature and Salinity
    ax11 = var11.plot(ax=ax1, grid=0, color='orange', **kwds)
    ax11.set_ylabel(u'Temperature (C)', color='orange')
    ax11.set_yticklabels(ax11.get_yticks(), color='orange')
    ax12 = var12.plot(ax=ax1, grid=1, color='red', secondary_y=True, **kwds)
    ax12.yaxis.grid(False)
    ax12.set_ylabel('Salinity (ppt)', color='red', rotation=-90, va='bottom')
    plt.setp(ax12.yaxis.get_majorticklabels(), color='red')
    plt.sca(ax11)
    plt.xticks(plt.xticks()[0], rotation=45, ha='center')

    fig.text(0.5, .96, 'Underway xCO$_2$ and other oceanographic parameters',
             va='bottom', ha='center', size=16)
    fig.subplots_adjust(hspace=0.17, wspace=.1,
                        top=0.94, bottom=0.04, left=0.05, right=0.95)
    if sname:
        fig.savefig(sname, dpi=300)

    return fig


def plot_grid_params(df):

    diag_cols = ('CO2f_sst', 'CO2x', 'CO2x_cor',
                 'Pres_equ_hpa', 'Pres_licor_hpa', 'Pres_atm_hpa',
                 'Chl_conc', 'O2_ppm', 'Salt_tsg',
                 'Temp_equ', 'Temp_intake', 'cond_temp',
                 'dry_box_temp', 'flow_h2o_Lmin', 'flow_licor_cm3min',
                 'flow_vent_cm3min', 'licor_h2o_x', 'licor_temp', )
    df.ix[:, diag_cols].plot(subplots=1,
                             figsize=[15, 20],
                             layout=[6, 3],
                             marker='.',
                             lw=0,
                             ms=4)
    plt.show()


if __name__ == '__main__':
    main()
