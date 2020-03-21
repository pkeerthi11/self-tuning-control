import pickle
from math import ceil

import numpy as np

from controller import ZeroController, AdaptiveControllerFilter
from simulation import single_simulation, constants_nevado_holgado_healthy

if __name__ == "__main__":
    constants = constants_nevado_holgado_healthy
    constants[5] = -0.9
    constants[16] = 10

    simulation_time = 2000
    dt = 0.05

    plot_color = False
    print('Simulations')

    f_range = np.arange(3, 101, 0.5)
    ctx_range = np.array([10, 22])
    stn_amplitude = np.zeros((len(f_range), len(ctx_range), 4))
    gpe_amplitude = np.zeros((len(f_range), len(ctx_range), 4))

    for ci, cv in enumerate(ctx_range):
        constants[18] = cv
        for i, f in enumerate(f_range):
            print('Running simulation for cortical mean value %.1f and frequency %.1f Hz' % (cv, f))
            constants[17] = f
            z = ZeroController()
            a = AdaptiveControllerFilter(0.1, 50, dt)

            h2 = single_simulation(constants, simulation_time, dt, control_mechanism=z)
            h3 = single_simulation(constants, simulation_time, dt, control_mechanism=a, init_theta=0)
            tail_length = int(ceil(800 / dt))
            stn_amplitude[i, ci, 1] = np.ptp(h2[-tail_length:, 0])
            stn_amplitude[i, ci, 2] = np.ptp(h3[-tail_length:, 0])
            gpe_amplitude[i, ci, 1] = np.ptp(h2[-tail_length:, 1])
            gpe_amplitude[i, ci, 2] = np.ptp(h3[-tail_length:, 1])
    with open('simulation_results/results_figure_2a', 'wb+') as results_file:
        pickle.dump({
            'stn': stn_amplitude,
            'gpe': gpe_amplitude,
            'constants': constants,
            'f_range': f_range
        }, results_file)
