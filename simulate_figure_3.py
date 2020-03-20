import numpy as np
import pickle
import main as sim
import controller as ctrl

if __name__ == '__main__':
    steady_state_pad = 1000

    constants = sim.constants_nevado_holgado_healthy
    constants[3] = -3
    constants[4] = 10
    constants[5] = -0.9
    constants[14] = 5
    constants[15] = -139.4

    max_delay = max(constants[6:10])
    simulation_time = 3000
    dt = 0.01

    print('Simulations')
    tt = np.arange(-max_delay, simulation_time, dt)
    sigma = 0.19
    mi = 15
    it = 0
    tau_theta = 75
    prop_theta = 2

    p_controller = ctrl.ProportionalController(gain=prop_theta, dt=dt)
    pi_controller = ctrl.ProportionalIntegralController(proportional_gain=prop_theta, integral_gain=prop_theta/2,
                                                        dt=dt)
    a_controller = ctrl.AdaptiveController(sigma=sigma, tau_theta=tau_theta, dt=dt)
    history_adaptive = sim.single_simulation(constants, max_delay, simulation_time, dt, control_mechanism=a_controller,
                                             init_theta=it, mid_increase=mi, steady_state_pad=steady_state_pad,
                                             plot=False)
    history_proportional = sim.single_simulation(constants, max_delay, simulation_time, dt,
                                                 control_mechanism=p_controller, init_theta=prop_theta, mid_increase=mi,
                                                 steady_state_pad=steady_state_pad,
                                                 plot=False)

    print('Saving simulation results')
    with open('simulation_results/results_figure_3', 'wb+') as f:
        pickle.dump({
            'proportional': history_proportional[int(steady_state_pad / dt):],
            'adaptive': history_adaptive[int(steady_state_pad / dt):],
            'constants': constants,
            'sigma': sigma,
            'mid_increase': mi,
            'initial_theta': it,
            'simulation_time': simulation_time,
            'dt': dt,
            'tau_theta': tau_theta,
            'prop_theta': prop_theta
        }, f)
