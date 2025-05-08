import numpy as np

def get_energy_estimation_set(Itorque,sf=100.0, v=48.0):
    '''
        Calculate energy estimation for torque current signal of SetiTec

        Returns the energy estimation in joules [J] for a single hole, assuming a
        constant voltage of 48V and a sampling rate of 100Hz.
        e.g.

        from energy_estimation import get_energy_estimation_set
        import dataparser as dp
        data = dp.loadSetitecXLS(path)[-1]

        energy = get_energy_estimation(Itorque=data["I Torque (A)"].values)

        Inputs:
            Itorque       : Full input vector of the "I Torque (A)"
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
    power = v * (Itorque)

    # replace NaNs with 0.0
    np.nan_to_num(power,copy=False)

    # energy as the integral of power over time
    energy = np.trapz(power, time[0:len(power)])

    # correction factor - regressor that includes the correction in comparison with a power analyser
    energy = 0.7521*energy+959.86

    return energy


def get_energy_estimation_lue(Ifiltorque, time, v=48.0):
    '''
        Calculate energy estimation for torque current signal of Luebbering

        Returns the energy estimation in joules [J] for a single hole, assuming a
        constant voltage of 48V and a sampling rate of 20Hz.
        e.g.

        from energy_estimation import get_energy_estimation_lue

        Ifiltorque = pd.read_excel(path).fil_torque
        time = pd.read_excel(path).timestamp/1000

        energy = get_energy_estimation_lue(Ifiltorque=Ifiltorque.values, time = time.values)

        Inputs:
            Ifiltorque    : Full input vector of the "fil_torque"
            time          : Full input vector of the "time"
            v             : Voltage, it could be a vector of the same size
                            as the current vectors
        Output:
            energy        : energy estimation for a single hole

    '''
    # vector of power
    power = Ifiltorque / 1000 * v

    # replace NaNs with 0.0
    np.nan_to_num(power, copy=False)

    # energy as the integral of power over time
    energy = np.trapz(power, time)

    # correction factor - regressor that includes the correction in comparison with a power analyser
    energy = 0.7996 * energy - 31.432

    return energy




