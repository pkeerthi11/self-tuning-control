import numpy as np
from math import ceil, floor
import pickle
import itertools
import scipy.signal as signal

from simulation import single_simulation, constants_nevado_holgado_healthy
from controller import ZeroController


if __name__ == '__main__':
    constants = constants_nevado_holgado_healthy
    constants[5] = -4
    constants[14] = 8
    constants[15] = -139.4

    simulation_time = 2000
    dt = 0.5

    c12_range = np.arange(0, -4, -0.125)
    c21_range = np.arange(0, 32, 1)

    stn_amplitude = np.zeros((len(c12_range), len(c21_range)))
    gpe_amplitude = np.zeros((len(c12_range), len(c21_range)))

    tail_len = int(ceil(800 / dt))
    z = ZeroController()
    for (i, c12), (j, c21) in itertools.product(enumerate(c12_range), enumerate(c21_range)):
        print('Running simulations for c12=%.2f, c21=%.2f' % (c12, c21))
        constants[3] = c12
        constants[4] = c21
        history = single_simulation(constants, simulation_time, dt, z)

        div = 10
        dhistory = signal.decimate(history, div, axis=0)
        fs = 1000/(dt * div)
        nyq = fs / 2
        b, a = signal.butter(5, [16/nyq, 24/nyq], btype='band')
        sstn = signal.filtfilt(b, a, dhistory[:, 0])
        sgpe = signal.filtfilt(b, a, dhistory[:, 1])

        stn_amplitude[i, j] = np.ptp(sstn[-int(floor(tail_len/div)):])
        gpe_amplitude[i, j] = np.ptp(sgpe[-int(floor(tail_len/div)):])

        with open('simulation_results/results_figure_1a', 'wb+') as results_file:
            pickle.dump({
                'stn': stn_amplitude,
                'gpe': gpe_amplitude,
                'constants': constants,
                'c12_range': c12_range,
                'c21_range': c21_range
            }, results_file)