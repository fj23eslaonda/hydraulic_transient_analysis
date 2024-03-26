import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

### Matplotlib config
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 12})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# PLOTS
def plot_results(params, z, eta, q, dzdt, pcp, plot_press, save, pathout, name, prdw_logo_path, figsize, window):
    if plot_press == True:
        fig, ax = plt.subplots(3,1,figsize=(20,10), sharex=True)

        fsize = 14

        ax[0].plot(params['tau'], eta, '-b', label='$\eta$ (sea)')
        ax[0].plot(params['tau'], z, '-r', label='$\zeta$ (reservoir)')
        ax[0].set_ylabel('surface elevation [m SWL]', fontsize=fsize)
        ax[0].tick_params(axis='y',labelsize=fsize)
        ax[0].legend(loc='SouthEast', fontsize=fsize)
        plt.grid(True)

        ax[1].plot(params['tau'], params['A'] * (q - dzdt), '-b', label='Q (pipeline)')
        ax[1].plot(params['tau'], params['A'] * q, '-r', label='$Q_R$ (influx)')
        ax[1].set_ylabel('seaward flowrate [m^3/s]', fontsize=fsize)
        ax[1].tick_params(axis='y',labelsize=fsize)
        ax[1].legend(loc='SouthEast', fontsize=0.8 * fsize)
        plt.grid(True)

        ax[2].plot(params['tau'], pcp * 1000, '-b', label='Control point')
        ax[2].set_xlabel('time [s]', fontsize=fsize)
        ax[2].set_ylabel('absolute pressure [mbar]', fontsize=fsize)
        ax[2].set_xlabel('Time [s]', fontsize=fsize)
        ax[2].tick_params(axis='x',labelsize=fsize)
        ax[2].tick_params(axis='y',labelsize=fsize)
        ax[2].legend(loc='SouthEast', fontsize=fsize)
        ax[2].grid(True)

        im = plt.imread(prdw_logo_path)
        ax = fig.add_axes(figsize)
        ax.imshow(im, aspect = 'equal', alpha = 0.5)
        ax.axis('off')

        if save == True:
            plt.savefig(pathout /f'{name}.png', dpi = 300, bbox_inches = 'tight')

    
    else:
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)

        fsize = 14

        ax[0].plot(params['tau'], eta, '-b', label='$\eta$ (sea)')
        ax[0].plot(params['tau'], z, '-r', label='$\zeta$ (reservoir)')
        ax[0].set_ylabel('surface elevation [m SWL]', fontsize=fsize)
        ax[0].tick_params(axis='y',labelsize=fsize)
        ax[0].legend(loc='SouthEast', fontsize=fsize)
        plt.grid(True)

        ax[1].plot(params['tau'], params['A'] * (q - dzdt), '-b', label='Q (pipeline)')
        ax[1].plot(params['tau'], params['A'] * q, '-r', label='$Q_R$ (influx)')
        ax[1].set_ylabel('seaward flowrate [m^3/s]', fontsize=fsize)
        ax[1].tick_params(axis='x',labelsize=fsize)
        ax[1].tick_params(axis='y',labelsize=fsize)
        ax[1].set_xlabel('Time [s]', fontsize=fsize)
        ax[1].legend(loc='SouthEast', fontsize=0.8 * fsize)
        plt.grid(True)

        im = plt.imread(prdw_logo_path)
        ax = fig.add_axes(figsize)
        ax.imshow(im, aspect = 'equal', alpha = 0.5)
        ax.axis('off')

        if save == True:
            plt.savefig(pathout / f'{name}.png', dpi = 300, bbox_inches = 'tight')
			
			
def print_outputs(params, eta, z, q, dzdt, pcp, maxdpcp):
    # OUTPUT
    print('\nNatural period Tn = {:.1f} s'.format(params['Tn']))
    print('\nMinimum level at reservoir = {:.2f} [m]'.format(min(z)))
    print('\nMaximum level at reservoir = {:.2f} [m]'.format(max(z)))
    print('\nFinal level at reservoir = {:.2f} [m]'.format(z[-1]))
    print('\nMaximum pressure oscillation at control point = {:.2f} [bar]\n'.format(maxdpcp))

    variables = [params['tau'], eta, z, q * params['A'], (q - dzdt) * params['A'], pcp]

    foutname = 'output.txt'
    headline = 'time[s] eta[m NRS] z[m NRS] Qres[m3/s] Qpipe[m3/s] Pcp[mbar]'

    with open(foutname, 'w') as fid:
        fid.write(headline + '\n')
        for i in range(len(variables)):
            line = ' '.join(['{:.2f}'.format(val) for val in variables[i]])
            fid.write(line + '\n')

def plot_initial_flowrate_eta(params, q_filename, eta, figsize, fsize):
    fig, ax = plt.subplots(2,1,figsize=figsize, sharex=True)

    ax[0].plot(q_filename['Time'], q_filename['Flowrate'], '-o')
    ax[0].set_ylabel('Flowrate [m3/s]', fontsize=fsize)
    ax[0].tick_params(axis='y',labelsize=fsize)
    ax[0].legend(loc='SouthEast', fontsize=fsize)

    ax[1].plot(params['tau'], eta, 'orange', label='$\eta$')
    ax[1].set_ylabel('eta [m]', fontsize=fsize)
    ax[1].tick_params(axis='y',labelsize=fsize)
    ax[1].legend(loc='SouthEast', fontsize=fsize)
    ax[1].tick_params(axis='x',labelsize=fsize)
    ax[1].set_xlabel('Time [s]', fontsize = fsize)
    plt.grid(True)