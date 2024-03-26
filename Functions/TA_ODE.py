import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def read_files(path_files,params, w_filename, q_filename):
# Time dependent functions

    if w_filename != '':
    # Read sea surface elevation from file
        wdata         = pd.read_csv(path_files/ w_filename, sep='\t', header=None)
        wdata.columns = ['t', 'q']
        f_wdata       = interp1d(wdata['t'], wdata['q'], kind='linear')
        eta           = f_wdata(params['tau'])
    else:
        eta = np.ones(params['ltau'])*params['swl']

    if q_filename.empty == False:
    # Read sea flowrate into the basin from file
        f_qdata       = interp1d(q_filename['Time'], q_filename['Flowrate'], kind='linear')
        q             = f_qdata(params['tau'])/(params['n']*params['A']);
    else:
        q = np.ones(params['ltau'])*params['Q']/params['A']
        q[1:int(params['dtq']/params['dt'])]= q[1:int(params['dtq']/params['dt'])]*0.5        

    return eta, q


# Define la ecuación diferencial con argumentos
def f_odePipeOsc(t,y, p, tau, eta, q, dq):
    ### eta interpolation
    eta = interp1d(tau, eta, kind='linear', fill_value='extrapolate')(t)
    
    ### q interpolation
    q  = interp1d(tau, q, kind='linear', fill_value='extrapolate')(t)

    ### dq interpolation
    dq = interp1d(tau, dq, kind='linear', fill_value='extrapolate')(t)
    
    ### Compute dy
    dy = np.zeros(2)
    dy[0] = y[1]
    dy[1] = p[0]**2*eta+dq-p[1]*abs(y[1]-q)*(y[1]-q)-p[0]**2*y[0]

    return dy

# Función de envoltura que incluye los argumentos adicionales
def f_odePipeOsc_2(t, y, p, tau, eta, q, dq):
    return f_odePipeOsc(t, y, p, tau, eta, q, dq)


def f_odePipeOsc_dif(t, y, p, tau, eta, q, dq):
    ### eta interpolation
    f_eta = interp1d(tau, eta)
    eta  = f_eta(t)
    
    ### q interpolation
    f_q  = interp1d(tau, q)
    q    = f_q(t)

    ### dq interpolation
    f_dq = interp1d(tau, dq)
    dq   = f_dq(t)
    
    ### Compute dy
    dy = np.zeros(2)
    dy[0] = np.min(y[1],q)
    dy[1] = p[0]**2*eta+dq-p[1]*abs(y[1]-q)*(y[1]-q)-p[0]**2*y[0]

    return dy

# Función de envoltura que incluye los argumentos adicionales
def f_odePipeOsc_dif_2(t, y, p, tau, eta, q, dq):
    return f_odePipeOsc_dif(t, y, p, tau, eta, q, dq)

def solve_ODE(params, eta, q, dq, y0):
    # Puntos de tiempo donde quieres obtener la solución
    t_span = (params['ti'], params['tf'])

    # ODE
    if params['disch_type'] == 0:   # If check valve is used
        # Resuelve la ecuación diferencial usando solve_ivp con la función de envoltura
        sol = solve_ivp(
            lambda t, y: f_odePipeOsc_2(t, y, p =np.asarray([params['w'], params['lmda']]), tau= params['tau'], eta= eta, q=q, dq=dq),
            t_span,
            y0,
            dense_output=True
        )
    else:
        # Resuelve la ecuación diferencial usando solve_ivp con la función de envoltura
        sol = solve_ivp(
            lambda t, y: f_odePipeOsc_dif_2(t, y, p =np.asarray([params['w'], params['lmda']]), tau= params['tau'], eta= eta, q=q, dq=dq),
            t_span,
            y0,
            dense_output=True
        )
    return sol

def print_outputs(params, eta, z, q, dzdt, pcp, maxdpcp):
    # OUTPUT
    print('\nNatural period Tn = {:.1f} s'.format(params['Tn']))
    print('\nMinimum level at reservoir = {:.2f} [m]'.format(min(z)))
    print('\nMaximum level at reservoir = {:.2f} [m]'.format(max(z)))
    print('\nFinal level at reservoir = {:.2f} [m]'.format(z[-1]))
    print('\nMaximum pressure oscillation at control point = {:.2f} [bar]\n'.format(maxdpcp))

    variables = [params['tau'], eta, z, q * params['A'], (q - dzdt) * params['A'], pcp]

    foutname = 'Results/output.txt'
    headline = 'time[s] eta[m NRS] z[m NRS] Qres[m3/s] Qpipe[m3/s] Pcp[mbar]'

    with open(foutname, 'w') as fid:
        fid.write(headline + '\n')
        for i in range(len(variables)):
            line = ' '.join(['{:.2f}'.format(val) for val in variables[i]])
            fid.write(line + '\n')
    


def pressure_at_control_point(params, z, eta, q):
    if params['disch_type'] == 1:
        dzdt = np.zeros(params['ltau'])
    elif params['disch_type'] == 0:
        dzdt                   = np.zeros(params['ltau'])
        dzdt[1:params['ltau']] = np.diff(z, axis=0) / np.diff(params['tau'], axis=0)
        dzdt[0]                = dzdt[1] 

    pcp      = (params['patm'] + ((eta - z) * (params['lcp'] / params['L']) + z - params['zcp'] + params['kav'] * ((params['alpha'] * (q - dzdt)) ** 2) / (2 * 9.81)) * 9.81 * 1025) * 1e-5
    dpcp     = pcp-pcp[params['ltau']-1]
    maxdpcp  = np.max(np.abs(dpcp))

    return dzdt, pcp, dpcp, maxdpcp

# PLOTS
def plot_results(params, z, eta, q, dzdt, pcp, plot_press, save, pathout, name, prdw_logo_path, figsize):
    if plot_press == True:
        fig, ax = plt.subplots(3,1,figsize=(20,10), sharex=True)

        fsize = 14

        z_dummy = interp1d(params['tau'], z, kind='cubic')(np.linspace(0,3600,3601))

        ax[0].plot(params['tau'], eta, '-b', label='$\eta$ (sea)')
        ax[0].plot(np.linspace(0,3600,3601), z_dummy, '-r', label='$\zeta$ (reservoir)')
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
        ax[2].set_xlabel('Tiempo [s]', fontsize=fsize)
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

        z_dummy = interp1d(params['tau'], z, kind='cubic')(np.linspace(0,3600,3601))

        ax[0].plot(params['tau'], eta, '-b', label='$\eta$ (sea)')
        ax[0].plot(np.linspace(0,3600,3601), z_dummy, '-r', label='$\zeta$ (reservoir)')
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

        im = plt.imread(prdw_logo_path)
        ax = fig.add_axes(figsize)
        ax.imshow(im, aspect = 'equal', alpha = 0.5)
        ax.axis('off')

        if save == True:
            plt.savefig(pathout / f'{name}.png', dpi = 300, bbox_inches = 'tight')

def initial_conditions (params, eta, q, conditions):

    # Level function
    funZ0 = lambda x: ((eta[0]-x)*params['g']/params['L'] + params['f'] /(2*params['D'])*np.abs(q[0]*params['alpha'])*q[0]*params['alpha'])

    # Velocity function
    funU0 = lambda x: ((eta[0]-z0)*params['g']/params['L']+params['f']/(2*params['D'])*np.abs(x)*x)

    if conditions == 'sequential_start':
        z0 = eta[0]
        u0 = q[0]*params['alpha']

    elif conditions == 'trip':
        z0 = root_scalar(funZ0, bracket=[-5, 5], method='brentq').root
        u0 = root_scalar(funU0, bracket=[-5, 5], method='brentq').root

    y01 = z0
    y02 = q[0]- u0/params['alpha']

    return y01, y02

def diff_q(params, q):
    dq                   = np.zeros(params['ltau'])
    dq[1:params['ltau']] = np.diff(q, axis=0) / np.diff(params['tau'], axis=0)
    dq[0]                = dq[1]

    return dq