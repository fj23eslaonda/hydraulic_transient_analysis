import numpy as np
import pandas as pd


def sequential_pumps_start(total_time, number_pumps, delta_time, flowrate):
    """
    Function to create an timeserie of flowrate
    due to sequetial pumps start process.

    Variables
    ------------
    total_time    : Simulation time
    number_pumps  : Number of pumps
    delta_time    : time between the pumps start
    flowrate      : total flowrate
    """
    delta_flowrate = flowrate/number_pumps

    # Create time array
    time_dummy1    = [delta_time*ix for ix in range(1,number_pumps+1)]
    time_dummy2    = [ix+1 for ix in time_dummy1]
    time_serie     = np.asarray(sorted([0] + time_dummy1 + time_dummy2 + [total_time]))

    # Initial variables
    jx = 0
    flowrate_serie = []
    initial_flowrate = 0

    # Create flowrate array
    for _ in time_serie:
        flowrate_serie.append(initial_flowrate)
        jx +=1
        if jx==2:
            initial_flowrate += delta_flowrate
            jx=0
    flowrate_serie = np.asarray(flowrate_serie)

    # Create dataframe to save time and flowrate
    df = pd.DataFrame()
    df['Time'] = time_serie
    df['Flowrate'] = -1*flowrate_serie

    return df

def abrupt_start(total_time, delta_time, flowrate, system='Intake'):
    '''
    Function to create an timeserie of flowrate
    due to abrut start process of whole system. you can select if you are working on intake or outfall system 

        Variables
    ------------
    total_time    : Simulation time
    delta_time    : time between the pumps start
    flowrate      : total flowrate
    system        : 'Intake' or 'Outfall'
    '''
    timeserie = np.asarray([0, delta_time, delta_time+1, total_time])
    if system == 'Intake':
        flowrate_serie = np.asarray([0, 0, flowrate, flowrate])*-1
    if system == 'Outfall':
        flowrate_serie = np.asarray([0, 0, flowrate, flowrate])

    df = pd.DataFrame()
    df['Time'] = timeserie
    df['Flowrate'] = flowrate_serie

    return df

def trip_pumps(total_time, delta_time, flowrate, system ='Intake'):
    '''
    Function to create an timeserie of flowrate
    due to trip process. you can select if you are working on intake or outfall system.
    
    Variables
    ------------
    total_time    : Simulation time
    delta_time    : time between the pumps start
    flowrate      : total flowrate
    system        : 'Intake' or 'Outfall'
    '''
    
    timeserie = np.asarray([0, delta_time, delta_time+1, total_time])
    if system == 'Intake':
        flowrate_serie = np.asarray([flowrate, flowrate, 0, 0])*-1
    if system == 'Outfall':
        flowrate_serie = np.asarray([flowrate, flowrate, 0, 0])

    df = pd.DataFrame()
    df['Time'] = timeserie
    df['Flowrate'] = flowrate_serie

    return df

def flowrate_timeserie(params, number_pumps, delta_time, system, serie_type):
    if serie_type == 'sequential_start':
        # sequential_pumps_start
        q_filename = sequential_pumps_start(total_time   = params['tf'], 
                                            number_pumps = number_pumps, 
                                            delta_time   = delta_time, 
                                            flowrate     = params['Q_total'])
    elif serie_type == 'abrupt_start':
        # abrupt start
        q_filename = abrupt_start(total_time = params['tf'], 
                                delta_time = delta_time, 
                                flowrate   = params['Q_total'],
                                system     = system)
    elif serie_type == 'trip':
        # trip
        q_filename = trip_pumps(total_time = params['tf'], 
                                delta_time = delta_time, 
                                flowrate   = params['Q_total'],
                                system     = system)
    return q_filename