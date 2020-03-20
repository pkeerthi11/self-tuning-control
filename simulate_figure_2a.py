import numpy as np
from math import ceil
import pickle

from main import single_simulation, constants_nevado_holgado_healthy
from controller import ZeroController, AdaptiveControllerFilter


if __name__ == "__main__":
    constants = constants_nevado_holgado_healthy
    constants[5] = -0.9
    constants[16] = 10

    max_delay = max(constants[6:10])
    simulation_time = 2000
    dt = 0.05

    plot_color = False
    print('Simulations')
    tt = np.arange(-max_delay, simulation_time, dt)
    history = np.zeros((len(tt), 3))
    control_history = np.zeros((len(tt),))

    f_range = np.arange(3, 101, 0.5)
    ctx_range = np.array([10, 22])
    stn_amplidute = np.zeros((len(f_range), len(ctx_range), 4))
    gpe_amplidute = np.zeros((len(f_range), len(ctx_range), 4))

    for ci, cv in enumerate(ctx_range):
        constants[18] = cv
        for i, f in enumerate(f_range):
            print('Running simulation for cortical mean value %.1f and frequency %.1f Hz' % (cv, f))
            constants[17] = f
            z = ZeroController()
            a = AdaptiveControllerFilter(0.1, 50, dt)

            h2 = single_simulation(constants, max_delay, simulation_time, dt, control_mechanism=z, plot=False)
            h3 = single_simulation(constants, max_delay, simulation_time, dt, control_mechanism=a, init_theta=0,
                                   plot=False)
            tail_length = int(ceil(800/dt))
            stn_amplidute[i, ci, 1] = np.ptp(h2[-tail_length:, 0])
            stn_amplidute[i, ci, 2] = np.ptp(h3[-tail_length:, 0])
            gpe_amplidute[i, ci, 1] = np.ptp(h2[-tail_length:, 1])
            gpe_amplidute[i, ci, 2] = np.ptp(h3[-tail_length:, 1])
    with open('simulation_results/results_figure_2a', 'wb+') as results_file:
        pickle.dump({
            'stn': stn_amplidute,
            'gpe': gpe_amplidute,
            'constants': constants,
            'f_range': f_range
        }, results_file)