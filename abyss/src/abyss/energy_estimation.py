import numpy as np

def get_energy_estimation(Itorque,Itorque_empty,Ithrust,sf=100.0, v=48.0):
    '''
        Calculate energy estimation for current signals of SetiTec

        Returns the energy estimation in joules [J] for a single hole, assuming a
        constant voltage of 48V and a sampling rate of 100Hz.
        e.g.

        from energy_estimation import get_energy_estimation
        import dataparser as dp
        data = dp.loadSetitecXLS(path)[-1]

        energy = get_energy_estimation(Itorque=data["I Torque (A)"].values,
                                       Itorque_empty=data["I Torque empty (A)"].values,
                                       Ithrust=data["I Thrust (A)"].values)

        Inputs:
            Itorque       : Full input vector of the "I Torque (A)"
            Itorque_empty : Full input vector of the "I Torque Empty (A)"
            Ithrust       : Full input vector of the "I Thrust (A)"
            sf            : Sampling rate.
            v             : Voltage, it could be a vector of the same size
                            as the current vectors
        Output:
            energy        : energy estimation for a single hole

    '''
    # sampling period
    T = 1 / sf
    # vector of time
    time = np.arange(0.0, len(Itorque) * T, T, dtype='float16')
    time = np.around(time, 2)
    # vector of power
    power = v * (Itorque+Itorque_empty+Ithrust)
    # replace NaNs with 0.0
    np.nan_to_num(power,copy=False)
    # energy as the integral of power over time
    energy = np.trapz(power, time[0:len(power)])

    return energy


